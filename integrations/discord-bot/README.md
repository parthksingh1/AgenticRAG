# Discord bot

`/ask <question>` — answers from your indexed corpus, with sources.

## Setup

1. Create an application at https://discord.com/developers, add a bot.
2. Invite it with the `applications.commands` and `bot` scopes.
3. Set `DISCORD_BOT_TOKEN`, `AGRAG_API_URL`, `AGRAG_API_KEY`.

```bash
pip install -r requirements.txt && python bot.py
```

## Why a slash command rather than a mention

A mention-triggered bot needs the message-content intent, which lets it read
every message in every channel it can see. That is far more access than
answering questions requires, and it is a privileged intent Discord will ask you
to justify.

A slash command also shows its parameter inline as you type, so people discover
what the bot accepts without being told.

## The three-second rule

Discord gives 3 seconds to acknowledge an interaction and 15 minutes to complete
it. `defer(thinking=True)` buys the time a RAG answer needs. Without it the
interaction fails and the user sees "the application did not respond" — with no
way to tell whether the question was even received.

## Splitting long answers

Discord caps a message at 2000 characters. Answers are split on paragraph
boundaries, then sentences, and only hard-split if a single sentence exceeds the
limit — a message that ends mid-word reads as a bug rather than a continuation.

## Conversation history

The channel id becomes the conversation id, so a follow-up in the same channel
has the history a follow-up needs.
