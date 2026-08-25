# Slack bot

Answers questions in Slack, in a thread, with sources.

## Setup

1. Create a Slack app; enable **Event Subscriptions** and subscribe to
   `app_mention` and `message.im`.
2. Add the `chat:write` and `app_mentions:read` bot scopes.
3. Point the request URL at `https://your-host/slack/events`.
4. Set `SLACK_BOT_TOKEN`, `SLACK_SIGNING_SECRET`, `AGRAG_API_URL`, `AGRAG_API_KEY`.

```bash
docker build -t agrag-slack . && docker run -p 3100:3100 --env-file .env agrag-slack
```

## Three things it gets right

**It acknowledges in under 3 seconds.** Slack redelivers an event that is not
acknowledged in time, so a bot that waits for the answer before replying gets
the same question twice and posts two answers. This one posts a placeholder
immediately and edits it.

**It replies in a thread, never in the channel.** A bot that answers inline turns
a busy channel into a transcript of everyone else's questions.

**It verifies every request.** Slack signs `v0:{timestamp}:{body}` with the
signing secret, and the timestamp inside the signed string is what makes a
captured request unusable after five minutes. Without verification the endpoint
is an open proxy to your corpus for anyone who finds the URL.

## Threads carry conversation history

The Slack thread id becomes the conversation id, so a follow-up in the same
thread works: "and how many can I carry over?" is meaningless without the turn
before it.

## When there are no sources

The bot says so explicitly. An uncited answer in a chat window looks exactly
like a cited one, and nobody checks.
