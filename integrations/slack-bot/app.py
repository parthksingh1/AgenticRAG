"""The AgenticRAG Slack bot.

Answers questions in channels and DMs, in a thread, with its sources.

    python -m integrations.slack_bot.app

Three things shape the design, and each of them is a mistake this bot avoids
making.

**Slack requires an acknowledgement within 3 seconds.** A RAG answer takes ten to
thirty. So the bot acknowledges immediately with a placeholder and edits it as
the answer arrives — the same reason the web UI streams.

**Answers go in a thread, never in the channel.** A bot that answers inline turns
a busy channel into a transcript of everyone else's questions.

**Every request is verified.** Slack signs its requests; anything unsigned or
stale is rejected. Without that, the endpoint is an open proxy to the tenant's
corpus for anyone who finds the URL.
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import os
import time
from typing import Any

import httpx

SLACK_API = "https://slack.com/api"

#: Slack rejects a response that takes longer than this, so the acknowledgement
#: must be sent before any work begins.
ACK_DEADLINE_SECONDS = 3.0

#: Requests older than this are replayed, not live. Slack's own guidance.
SIGNATURE_TOLERANCE_SECONDS = 300

#: Slack truncates blocks past this. Answers are trimmed with a pointer rather
#: than silently cut mid-sentence.
MAX_BLOCK_CHARS = 2900

#: Sources shown under an answer. More than this and the message is longer than
#: the answer it supports.
MAX_CITATIONS_SHOWN = 4


class SlackBot:
    """Bridges Slack events to the AgenticRAG API."""

    def __init__(
        self,
        *,
        bot_token: str,
        signing_secret: str,
        api_url: str,
        api_key: str,
    ) -> None:
        """Configure the bot.

        Raises:
            ValueError: when a credential is missing. Failing at startup beats
                failing on the first message a user sends.
        """
        missing = [
            name
            for name, value in (
                ("SLACK_BOT_TOKEN", bot_token),
                ("SLACK_SIGNING_SECRET", signing_secret),
                ("AGRAG_API_KEY", api_key),
            )
            if not value
        ]
        if missing:
            msg = f"missing required configuration: {', '.join(missing)}"
            raise ValueError(msg)

        self._bot_token = bot_token
        self._signing_secret = signing_secret
        self._api_url = api_url.rstrip("/")
        self._api_key = api_key
        self._seen_events: set[str] = set()

    def verify(self, *, body: bytes, timestamp: str, signature: str) -> bool:
        """Verify Slack's request signature.

        Without this the endpoint is an open proxy to the tenant's corpus for
        anyone who discovers the URL.

        Example:
            >>> bot = SlackBot(bot_token="x", signing_secret="s", api_url="u", api_key="k")
            >>> bot.verify(body=b"{}", timestamp="0", signature="v0=bad")
            False
        """
        try:
            age = abs(time.time() - int(timestamp))
        except (TypeError, ValueError):
            return False

        if age > SIGNATURE_TOLERANCE_SECONDS:
            # A valid signature on an old request is a replay. The timestamp is
            # inside the signed base string precisely so this check is possible.
            return False

        base = b"v0:" + timestamp.encode() + b":" + body
        expected = "v0=" + hmac.new(self._signing_secret.encode(), base, hashlib.sha256).hexdigest()
        # compare_digest: a naive == leaks the position of the first differing
        # byte through timing, which is enough to forge a signature.
        return hmac.compare_digest(expected, signature)

    def is_duplicate(self, event_id: str) -> bool:
        """Whether this event has already been handled.

        Slack redelivers an event when it does not get a fast acknowledgement,
        and answering twice costs two model calls and posts two replies.

        Example:
            >>> bot = SlackBot(bot_token="x", signing_secret="s", api_url="u", api_key="k")
            >>> bot.is_duplicate("Ev1"), bot.is_duplicate("Ev1")
            (False, True)
        """
        if event_id in self._seen_events:
            return True
        self._seen_events.add(event_id)
        # Bounded, because this is a process-local cache and a redelivery
        # arrives within seconds. An unbounded set is a slow memory leak.
        if len(self._seen_events) > 5000:
            self._seen_events = set(list(self._seen_events)[-2500:])
        return False

    async def handle_event(self, payload: dict[str, Any]) -> None:
        """Handle one Slack event, answering in a thread."""
        event = payload.get("event", {})

        # Ignore the bot's own messages. Without this the bot answers itself and
        # the thread grows until the rate limit stops it.
        if event.get("bot_id") or event.get("subtype") == "bot_message":
            return

        question = _strip_mention(str(event.get("text", "")))
        if not question:
            return

        channel = str(event.get("channel", ""))
        # Reply in the existing thread if there is one, otherwise start one on
        # this message. Never in the channel.
        thread = str(event.get("thread_ts") or event.get("ts", ""))

        placeholder = await self._post(
            channel=channel, thread_ts=thread, text="_Searching the corpus…_"
        )

        try:
            answer = await self._ask(question, thread=thread)
        except Exception as exc:  # noqa: BLE001 - the user gets a message, not a silence
            await self._update(
                channel=channel,
                ts=placeholder,
                text=f":warning: Could not answer that: {str(exc)[:200]}",
            )
            return

        await self._update(channel=channel, ts=placeholder, text="", blocks=_blocks(answer))

    async def _ask(self, question: str, *, thread: str) -> dict[str, Any]:
        """Ask the API.

        The Slack thread id becomes the conversation id, so a follow-up in the
        same thread has the history a follow-up needs — "and how many can I carry
        over?" is meaningless without it.
        """
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                f"{self._api_url}/api/chat",
                headers={"Authorization": f"Bearer {self._api_key}"},
                json={"message": question, "conversation_id": f"slack-{thread}"},
            )
            response.raise_for_status()
            return dict(response.json())

    async def _post(self, *, channel: str, thread_ts: str, text: str) -> str:
        """Post a message and return its timestamp."""
        body = await self._call(
            "chat.postMessage", {"channel": channel, "thread_ts": thread_ts, "text": text}
        )
        return str(body.get("ts", ""))

    async def _update(
        self, *, channel: str, ts: str, text: str, blocks: list[dict[str, Any]] | None = None
    ) -> None:
        """Edit a previously posted message."""
        payload: dict[str, Any] = {"channel": channel, "ts": ts, "text": text or " "}
        if blocks:
            payload["blocks"] = blocks
        await self._call("chat.update", payload)

    async def _call(self, method: str, payload: dict[str, Any]) -> dict[str, Any]:
        """Call a Slack Web API method.

        Raises:
            RuntimeError: when Slack reports an error. Slack returns 200 with
                ``ok: false``, so checking the status code alone silently
                swallows every failure.
        """
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{SLACK_API}/{method}",
                headers={
                    "Authorization": f"Bearer {self._bot_token}",
                    "Content-Type": "application/json; charset=utf-8",
                },
                json=payload,
            )
            body = dict(response.json())

        if not body.get("ok"):
            msg = f"slack {method} failed: {body.get('error', 'unknown')}"
            raise RuntimeError(msg)
        return body


def _strip_mention(text: str) -> str:
    """Remove the `<@U123>` mention from a message.

    Leaving it in sends the bot's own user id to the retriever as part of the
    query, which is noise in the embedding and occasionally in the answer.

    Example:
        >>> _strip_mention("<@U0123ABC> what is the notice period?")
        'what is the notice period?'
        >>> _strip_mention("plain question")
        'plain question'
    """
    import re

    return re.sub(r"<@[A-Z0-9]+>", "", text).strip()


def _blocks(answer: dict[str, Any]) -> list[dict[str, Any]]:
    """Render an answer as Slack blocks.

    Example:
        >>> _blocks({"content": "hello", "citations": []})[0]["type"]
        'section'
    """
    content = str(answer.get("content", "")).strip() or "_No answer._"
    if len(content) > MAX_BLOCK_CHARS:
        # Trimmed with a pointer rather than cut mid-sentence, so the reader
        # knows there was more rather than assuming the answer ended there.
        content = content[:MAX_BLOCK_CHARS].rsplit(" ", 1)[0] + "… _(truncated)_"

    blocks: list[dict[str, Any]] = [
        {"type": "section", "text": {"type": "mrkdwn", "text": content}}
    ]

    citations = answer.get("citations") or []
    if citations:
        lines = []
        for citation in citations[:MAX_CITATIONS_SHOWN]:
            page = f" · p.{citation['page_number']}" if citation.get("page_number") else ""
            lines.append(f"`[{citation['index']}]` {citation['document_title']}{page}")
        if len(citations) > MAX_CITATIONS_SHOWN:
            lines.append(f"_and {len(citations) - MAX_CITATIONS_SHOWN} more_")
        blocks.append(
            {"type": "context", "elements": [{"type": "mrkdwn", "text": "\n".join(lines)}]}
        )
    else:
        # Saying so is the honest option. A Slack answer with no sources looks
        # exactly like one with them, and in a chat window nobody checks.
        blocks.append(
            {
                "type": "context",
                "elements": [{"type": "mrkdwn", "text": "_No sources — treat with caution._"}],
            }
        )

    if answer.get("model"):
        blocks.append(
            {
                "type": "context",
                "elements": [
                    {
                        "type": "mrkdwn",
                        "text": f"_{answer['model']} · ${float(answer.get('cost_usd', 0)):.4f}_",
                    }
                ],
            }
        )

    return blocks


def build_app() -> Any:
    """Build the FastAPI app that receives Slack's webhooks."""
    from fastapi import FastAPI, Header, Request
    from fastapi.responses import JSONResponse

    bot = SlackBot(
        bot_token=os.getenv("SLACK_BOT_TOKEN", ""),
        signing_secret=os.getenv("SLACK_SIGNING_SECRET", ""),
        api_url=os.getenv("AGRAG_API_URL", "http://localhost:8000"),
        api_key=os.getenv("AGRAG_API_KEY", ""),
    )

    app = FastAPI(title="AgenticRAG Slack bot")

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        """Liveness."""
        return {"status": "ok"}

    @app.post("/slack/events")
    async def events(
        request: Request,
        x_slack_signature: str = Header(default=""),
        x_slack_request_timestamp: str = Header(default=""),
    ) -> Any:
        """Receive a Slack event."""
        body = await request.body()

        if not bot.verify(
            body=body, timestamp=x_slack_request_timestamp, signature=x_slack_signature
        ):
            return JSONResponse({"error": "invalid signature"}, status_code=401)

        payload = await request.json()

        # The URL verification handshake, sent once when the endpoint is
        # registered.
        if payload.get("type") == "url_verification":
            return {"challenge": payload.get("challenge")}

        event_id = str(payload.get("event_id", ""))
        if event_id and bot.is_duplicate(event_id):
            return {"ok": True}

        # Answer in the background and acknowledge now. Slack gives 3 seconds
        # and a RAG answer takes ten to thirty; a slow acknowledgement means
        # Slack redelivers and the user gets two answers.
        asyncio.create_task(bot.handle_event(payload))  # noqa: RUF006
        return {"ok": True}

    return app


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(build_app(), host="0.0.0.0", port=int(os.getenv("PORT", "3100")))  # noqa: S104
