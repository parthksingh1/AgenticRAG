"""The AgenticRAG Discord bot.

Answers questions as a slash command, with its sources.

    python -m integrations.discord_bot.bot

Discord's constraints are different from Slack's in ways that change the design.

**Three seconds to acknowledge, then fifteen minutes.** `defer()` buys the time a
RAG answer needs. Without it the interaction fails and the user sees "the
application did not respond" — with no way to tell whether their question was
received.

**2000 characters per message.** Long answers are split on paragraph boundaries
rather than cut at 2000, because a message ending mid-word reads as a bug.

**A slash command, not a mention.** Discord shows the parameter inline as you
type, so people discover what the bot accepts without being told. A
mention-triggered bot in a busy server also reads every message in every channel
it can see, which is far more access than answering questions requires.
"""

from __future__ import annotations

import os
from typing import Any

import httpx

#: Discord's hard limit. Answers longer than this are split, never truncated.
MAX_MESSAGE_CHARS = 2000

#: Leaves room for the citation footer without a second round of splitting.
SPLIT_TARGET_CHARS = 1800

#: Sources listed under an answer.
MAX_CITATIONS_SHOWN = 4

#: Discord embeds cap a field at 1024; the description is the safer place for
#: an answer of unknown length.
EMBED_DESCRIPTION_LIMIT = 4000


class AgRagClient:
    """Talks to the AgenticRAG API.

    Separated from the Discord layer so the retry and error handling can be
    tested without a gateway connection, and so the same class serves the
    slash command and any future context-menu action.
    """

    def __init__(self, *, api_url: str, api_key: str) -> None:
        """Configure the client.

        Raises:
            ValueError: when the API key is missing. Failing at startup beats
                failing on the first question somebody asks.
        """
        if not api_key:
            msg = "AGRAG_API_KEY is required"
            raise ValueError(msg)
        self._api_url = api_url.rstrip("/")
        self._api_key = api_key

    async def ask(self, question: str, *, conversation_id: str) -> dict[str, Any]:
        """Ask a question.

        Raises:
            RuntimeError: with a message fit to show a user. A traceback in a
                Discord channel helps nobody and leaks internals.
        """
        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                response = await client.post(
                    f"{self._api_url}/api/chat",
                    headers={"Authorization": f"Bearer {self._api_key}"},
                    json={"message": question, "conversation_id": conversation_id},
                )
                response.raise_for_status()
                return dict(response.json())
        except httpx.TimeoutException as exc:
            msg = "That took too long. Try a narrower question."
            raise RuntimeError(msg) from exc
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 429:
                msg = "Rate limited. Try again in a minute."
            elif exc.response.status_code in {401, 403}:
                msg = "The bot's credentials were rejected. Tell an admin."
            else:
                msg = f"The API returned {exc.response.status_code}."
            raise RuntimeError(msg) from exc
        except httpx.HTTPError as exc:
            msg = "Could not reach the API."
            raise RuntimeError(msg) from exc


def split_message(text: str, limit: int = SPLIT_TARGET_CHARS) -> list[str]:
    """Split a long answer into Discord-sized messages.

    Splits on paragraph boundaries, then on sentences, then — only if a single
    sentence is somehow longer than the limit — on the character count. A
    message that ends mid-word reads as a bug rather than as a continuation.

    Example:
        >>> split_message("short")
        ['short']
        >>> len(split_message("a" * 4000))
        3
        >>> parts = split_message("one" + chr(10) * 2 + "two", limit=5)
        >>> parts
        ['one', 'two']
    """
    if len(text) <= limit:
        return [text] if text else []

    chunks: list[str] = []
    current = ""

    for paragraph in text.split("\n\n"):
        candidate = f"{current}\n\n{paragraph}" if current else paragraph

        if len(candidate) <= limit:
            current = candidate
            continue

        if current:
            chunks.append(current)
            current = ""

        if len(paragraph) <= limit:
            current = paragraph
            continue

        # One paragraph is over the limit on its own. Sentences next.
        sentence_buffer = ""
        for sentence in paragraph.replace(". ", ".\x00").split("\x00"):
            if len(sentence_buffer) + len(sentence) <= limit:
                sentence_buffer += sentence
                continue
            if sentence_buffer:
                chunks.append(sentence_buffer.strip())
            # A single sentence longer than the limit: hard-split, because
            # there is nothing better left to split on.
            while len(sentence) > limit:
                chunks.append(sentence[:limit])
                sentence = sentence[limit:]
            sentence_buffer = sentence
        current = sentence_buffer.strip()

    if current:
        chunks.append(current)
    return [c for c in chunks if c]


def build_embed(answer: dict[str, Any]) -> dict[str, Any]:
    """Render an answer as a Discord embed.

    Example:
        >>> build_embed({"content": "hi", "citations": []})["color"]
        3447003
    """
    content = str(answer.get("content", "")).strip() or "*No answer.*"
    if len(content) > EMBED_DESCRIPTION_LIMIT:
        content = content[:EMBED_DESCRIPTION_LIMIT].rsplit(" ", 1)[0] + "… *(truncated)*"

    citations = answer.get("citations") or []
    if citations:
        lines = []
        for citation in citations[:MAX_CITATIONS_SHOWN]:
            page = f" · p.{citation['page_number']}" if citation.get("page_number") else ""
            lines.append(f"`[{citation['index']}]` {citation['document_title']}{page}")
        if len(citations) > MAX_CITATIONS_SHOWN:
            lines.append(f"*and {len(citations) - MAX_CITATIONS_SHOWN} more*")
        footer_fields = [{"name": "Sources", "value": "\n".join(lines), "inline": False}]
    else:
        # Said explicitly. An uncited answer looks identical to a cited one in a
        # chat window, and nobody checks.
        footer_fields = [
            {"name": "Sources", "value": "*None — treat with caution.*", "inline": False}
        ]

    return {
        "description": content,
        "color": 0x3498DB,
        "fields": footer_fields,
        "footer": {
            "text": f"{answer.get('model', 'unknown')} · "
            f"${float(answer.get('cost_usd', 0)):.4f} · "
            f"{int(answer.get('latency_ms', 0)) / 1000:.1f}s"
        },
    }


def build_bot() -> Any:
    """Build the Discord client with its slash command registered."""
    import discord
    from discord import app_commands

    client_api = AgRagClient(
        api_url=os.getenv("AGRAG_API_URL", "http://localhost:8000"),
        api_key=os.getenv("AGRAG_API_KEY", ""),
    )

    # The default intents, with no message-content intent. The bot answers slash
    # commands, so it never needs to read messages — and requesting an intent
    # that is not needed is access nobody should grant.
    intents = discord.Intents.default()
    bot = discord.Client(intents=intents)
    tree = app_commands.CommandTree(bot)

    @tree.command(name="ask", description="Ask a question about the indexed documents.")
    @app_commands.describe(question="What do you want to know?")
    async def ask(interaction: discord.Interaction, question: str) -> None:
        """Answer a question."""
        # Defer immediately. Discord allows 3 seconds to acknowledge and a RAG
        # answer takes ten to thirty; without this the user sees "the
        # application did not respond" and cannot tell whether it was received.
        await interaction.response.defer(thinking=True)

        # The channel is the conversation, so a follow-up in the same channel
        # has the history that makes a follow-up meaningful.
        conversation_id = f"discord-{interaction.channel_id}"

        try:
            answer = await client_api.ask(question, conversation_id=conversation_id)
        except RuntimeError as exc:
            await interaction.followup.send(f"⚠️ {exc}")
            return

        embed_data = build_embed(answer)
        embed = discord.Embed(description=embed_data["description"], color=embed_data["color"])
        for field in embed_data["fields"]:
            embed.add_field(name=field["name"], value=field["value"], inline=field["inline"])
        embed.set_footer(text=embed_data["footer"]["text"])

        await interaction.followup.send(embed=embed)

    @bot.event
    async def on_ready() -> None:
        """Register the slash commands once connected."""
        await tree.sync()
        print(f"connected as {bot.user}")

    return bot


def main() -> int:
    """Run the bot."""
    token = os.getenv("DISCORD_BOT_TOKEN", "")
    if not token:
        print("DISCORD_BOT_TOKEN is required")
        return 2

    build_bot().run(token)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
