"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import {
  ArrowUpRight,
  Brain,
  ChevronDown,
  Copy,
  RefreshCw,
  Send,
  Sparkles,
  Square,
  ThumbsDown,
  ThumbsUp,
} from "lucide-react";
import { streamAnswer, type Citation } from "@/lib/api";
import { cn } from "@/lib/cn";
import { CitationSheet } from "@/components/citation-sheet";
import { Inspector, emptyRun, type RunState } from "@/components/inspector";

/** Read a string off an untrusted event payload, or undefined. */
function str(value: unknown): string | undefined {
  return typeof value === "string" ? value : undefined;
}

/** Read a number off an untrusted event payload, or undefined. */
function num(value: unknown): number | undefined {
  return typeof value === "number" ? value : undefined;
}

interface Turn {
  id: string;
  role: "user" | "assistant";
  content: string;
  citations: Citation[];
  thinking: string[];
  streaming?: boolean;
  error?: string;
  meta?: { model?: string; latencyMs?: number; costUsd?: number; cacheHit?: string | null };
}

/** The chat surface: streaming answers, visible reasoning, inline citations. */
export function Chat() {
  const [turns, setTurns] = useState<Turn[]>([]);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const [openCitation, setOpenCitation] = useState<Citation | null>(null);
  const [run, setRun] = useState<RunState | null>(null);
  const abortRef = useRef<AbortController | null>(null);
  const conversationRef = useRef<string | undefined>(undefined);
  const endRef = useRef<HTMLDivElement>(null);

  // Only follow the stream when the user is already at the bottom. Yanking the
  // viewport down while somebody is reading an earlier answer is worse than not
  // scrolling at all.
  const scrollLockedRef = useRef(true);
  useEffect(() => {
    if (scrollLockedRef.current) endRef.current?.scrollIntoView({ block: "end" });
  }, [turns]);

  const send = useCallback(
    async (question: string) => {
      if (!question.trim() || busy) return;

      const userTurn: Turn = {
        id: crypto.randomUUID(),
        role: "user",
        content: question,
        citations: [],
        thinking: [],
      };
      const assistantTurn: Turn = {
        id: crypto.randomUUID(),
        role: "assistant",
        content: "",
        citations: [],
        thinking: [],
        streaming: true,
      };

      setTurns((prev) => [...prev, userTurn, assistantTurn]);
      setInput("");
      setBusy(true);
      // Input guardrails are the first thing the graph runs, and the stream
      // opening at all is proof they passed.
      setRun({ ...emptyRun(), stages: { ...emptyRun().stages, guardrails_in: "active" } });
      scrollLockedRef.current = true;

      const controller = new AbortController();
      abortRef.current = controller;

      const patch = (fn: (turn: Turn) => Turn) =>
        setTurns((prev) => prev.map((t) => (t.id === assistantTurn.id ? fn(t) : t)));

      try {
        for await (const event of streamAnswer(
          { message: question, conversation_id: conversationRef.current, include_thinking: true },
          controller.signal,
        )) {
          const data = event.data;
          switch (event.type) {
            case "token":
              patch((t) => ({ ...t, content: t.content + (str(data.text) ?? "") }));
              setRun((r) =>
                r && r.stages.generate !== "active" && r.stages.generate !== "done"
                  ? {
                      ...r,
                      stages: {
                        ...r.stages,
                        // Tokens arriving means everything upstream finished.
                        // Retrieval is only claimed if citations were seen; a
                        // question routed away from retrieval skips it.
                        route: "done",
                        retrieve: r.citations > 0 ? "done" : "skipped",
                        rerank: r.citations > 0 ? "done" : "skipped",
                        generate: "active",
                      },
                    }
                  : r,
              );
              break;
            case "thinking": {
              const step = str(data.text);
              if (step) patch((t) => ({ ...t, thinking: [...t.thinking, step] }));
              setRun((r) =>
                r
                  ? {
                      ...r,
                      thinkingSteps: r.thinkingSteps + 1,
                      stages: { ...r.stages, guardrails_in: "done", route: "active" },
                    }
                  : r,
              );
              break;
            }
            case "citations": {
              const cites = (data.citations as Citation[]) ?? [];
              patch((t) => ({ ...t, citations: cites }));
              setRun((r) =>
                r
                  ? {
                      ...r,
                      citations: cites.length,
                      stages: {
                        ...r.stages,
                        guardrails_in: "done",
                        route: "done",
                        retrieve: "done",
                        rerank: "done",
                      },
                    }
                  : r,
              );
              break;
            }
            case "done":
              conversationRef.current = str(data.conversation_id) ?? conversationRef.current;
              patch((t) => ({
                ...t,
                streaming: false,
                meta: {
                  model: str(data.model),
                  latencyMs: num(data.latency_ms),
                  costUsd: num(data.cost_usd),
                  cacheHit: str(data.cache_hit) ?? null,
                },
              }));
              setRun((r) => {
                if (!r) return r;
                // stop_reason tells us whether this ended as an answer or as a
                // refusal at the output guardrail; the pipeline reads
                // differently in each case and should not claim otherwise.
                const blocked = str(data.stop_reason) === "guardrail_block";
                return {
                  ...r,
                  model: str(data.model),
                  latencyMs: num(data.latency_ms),
                  costUsd: num(data.cost_usd),
                  cacheHit: str(data.cache_hit) ?? null,
                  blocked,
                  stages: {
                    ...r.stages,
                    guardrails_in: blocked ? "done" : r.stages.guardrails_in,
                    route: blocked ? "skipped" : "done",
                    retrieve: blocked ? "skipped" : r.stages.retrieve === "pending" ? "skipped" : "done",
                    rerank: blocked ? "skipped" : r.stages.rerank === "pending" ? "skipped" : "done",
                    generate: blocked ? "skipped" : "done",
                    cite: r.citations > 0 ? "done" : "skipped",
                    guardrails_out: "done",
                  },
                };
              });
              break;
            case "error":
              patch((t) => ({
                ...t,
                streaming: false,
                error: str(data.message) ?? "Something went wrong.",
              }));
              break;
          }
        }
      } catch (error) {
        // An abort is the user pressing stop, not a failure. Keeping the partial
        // answer is the useful behaviour — they stopped it because they had
        // read enough.
        const aborted = error instanceof Error && error.name === "AbortError";
        patch((t) => ({
          ...t,
          streaming: false,
          error: aborted ? undefined : "The connection dropped before the answer finished.",
        }));
      } finally {
        patch((t) => ({ ...t, streaming: false }));
        setBusy(false);
        abortRef.current = null;
      }
    },
    [busy],
  );

  function stop() {
    abortRef.current?.abort();
  }

  function regenerate(index: number) {
    const question = turns[index - 1];
    if (!question) return;
    setTurns((prev) => prev.slice(0, index));
    void send(question.content);
  }

  return (
    <div className="flex h-dvh min-w-0 flex-1">
      <div className="flex min-w-0 flex-1 flex-col">
      <header className="sticky top-0 z-10 flex items-center justify-between border-b border-line bg-bg/85 px-5 py-3 backdrop-blur-md">
        <div className="flex items-center gap-2.5">
          <h1 className="text-[13.5px] font-semibold tracking-tight">Chat</h1>
          <span className="hidden items-center gap-1.5 rounded-full border border-line px-2 py-0.5 text-[10.5px] text-muted sm:inline-flex">
            <span
              aria-hidden
              className={cn(
                "h-1.5 w-1.5 rounded-full",
                busy ? "animate-pulse bg-amber-500" : "bg-emerald-500",
              )}
            />
            {busy ? "Generating" : "Ready"}
          </span>
        </div>
        {turns.length > 0 && (
          <button
            type="button"
            onClick={() => {
              conversationRef.current = undefined;
              setTurns([]);
            }}
            className="rounded-lg border border-line px-2.5 py-1.5 text-[11.5px] text-muted transition-colors hover:border-accent/40 hover:text-fg"
          >
            New conversation
          </button>
        )}
      </header>

      <div
        className="flex-1 overflow-y-auto px-5 [scrollbar-gutter:stable]"
        onScroll={(e) => {
          const el = e.currentTarget;
          scrollLockedRef.current = el.scrollHeight - el.scrollTop - el.clientHeight < 80;
        }}
      >
        <div className="mx-auto w-full max-w-3xl py-7">
          {turns.length === 0 ? <Empty onPick={send} /> : null}

          {turns.map((turn, index) => (
            <Message
              key={turn.id}
              turn={turn}
              onCitationClick={setOpenCitation}
              onRegenerate={turn.role === "assistant" ? () => regenerate(index) : undefined}
            />
          ))}
          <div ref={endRef} />
        </div>
      </div>

      <Composer
        value={input}
        busy={busy}
        onChange={setInput}
        onSubmit={() => void send(input)}
        onStop={stop}
      />

      <CitationSheet citation={openCitation} onClose={() => setOpenCitation(null)} />
      </div>

      <Inspector run={run} busy={busy} />
    </div>
  );
}

function Empty({ onPick }: { onPick: (q: string) => void }) {
  const suggestions = [
    {
      q: "What is the carry-over limit for annual leave, and when does it expire?",
      note: "A direct lookup, answered with an inline citation",
    },
    {
      q: "I spent $5,000 two months ago. Who approves it and can I still claim it?",
      note: "Two sections no single chunk contains",
    },
    {
      q: "If I have 26 leave days and use 11, how many are left?",
      note: "Retrieval plus a tool call",
    },
    {
      q: "What is the company's dental insurance provider?",
      note: "Not in the corpus — the right answer says so",
    },
  ];

  return (
    <div className="animate-fade-up py-14">
      <div className="mx-auto max-w-xl text-center">
        <span
          aria-hidden
          className="accent-gradient mx-auto mb-5 grid h-12 w-12 place-items-center rounded-2xl text-white shadow-md"
        >
          <Sparkles size={22} />
        </span>
        <h2 className="text-[26px] font-semibold leading-tight tracking-[-0.02em]">
          Ask the corpus something
        </h2>
        <p className="mx-auto mt-2.5 max-w-md text-[13.5px] leading-relaxed text-muted">
          Every answer cites the passage it came from. The last suggestion is
          deliberately unanswerable — a system worth trusting says so rather than
          inventing a provider.
        </p>
      </div>

      <div className="mx-auto mt-9 grid max-w-2xl gap-2.5 sm:grid-cols-2">
        {suggestions.map(({ q, note }) => (
          <button
            key={q}
            type="button"
            onClick={() => onPick(q)}
            className="surface group p-3.5 text-left transition-all hover:-translate-y-px hover:border-accent/35 hover:shadow-md"
          >
            <span className="block text-[13px] font-medium leading-snug">{q}</span>
            <span className="mt-2 flex items-center gap-1.5 text-[11.5px] text-muted">
              <ArrowUpRight
                size={12}
                aria-hidden
                className="shrink-0 transition-transform group-hover:translate-x-0.5 group-hover:-translate-y-0.5"
              />
              {note}
            </span>
          </button>
        ))}
      </div>
    </div>
  );
}

function Message({
  turn,
  onCitationClick,
  onRegenerate,
}: {
  turn: Turn;
  onCitationClick: (c: Citation) => void;
  onRegenerate?: () => void;
}) {
  const [copied, setCopied] = useState(false);
  const [vote, setVote] = useState<"up" | "down" | null>(null);

  if (turn.role === "user") {
    return (
      <div className="mb-7 flex justify-end">
        <div className="accent-gradient max-w-[80%] animate-fade-up rounded-[18px] rounded-br-[6px] px-4 py-2.5 text-[13.5px] leading-relaxed text-white shadow-md">
          {turn.content}
        </div>
      </div>
    );
  }

  return (
    <article className="mb-8">
      {turn.thinking.length > 0 && <Thinking steps={turn.thinking} live={turn.streaming} />}

      {turn.error ? (
        <p className="rounded-lg border border-red-500/30 bg-red-500/5 px-3 py-2 text-sm text-red-500">
          {turn.error}
        </p>
      ) : (
        <div className="prose-answer animate-fade-up text-[14.5px] leading-[1.7]">
          <ReactMarkdown
            remarkPlugins={[remarkGfm]}
            components={{
              // Inline [n] markers become buttons that open the source. Rendered
              // by walking the text nodes rather than with a regex over the raw
              // markdown, so a [1] inside a code block stays literal.
              p: ({ children }) => <p>{linkCitations(children, turn.citations, onCitationClick)}</p>,
              li: ({ children }) => (
                <li>{linkCitations(children, turn.citations, onCitationClick)}</li>
              ),
            }}
          >
            {turn.content}
          </ReactMarkdown>
          {turn.streaming && (
            <span className="ml-0.5 inline-block h-4 w-[2px] animate-pulse bg-accent align-middle" />
          )}
        </div>
      )}

      {turn.citations.length > 0 && (
        <ul className="mt-3 flex flex-wrap gap-1.5">
          {turn.citations.map((c) => (
            <li key={c.chunk_id}>
              <button
                type="button"
                onClick={() => onCitationClick(c)}
                className="surface px-2.5 py-1.5 text-[11.5px] text-muted transition-all hover:-translate-y-px hover:border-accent/40 hover:text-fg hover:shadow-md"
              >
                <span className="font-medium text-accent">[{c.index}]</span> {c.document_title}
                {c.page_number ? ` · p.${c.page_number}` : ""}
              </button>
            </li>
          ))}
        </ul>
      )}

      {!turn.streaming && (
        <div className="mt-3 flex items-center gap-1 text-muted">
          <IconButton
            label={copied ? "Copied" : "Copy answer"}
            onClick={() => {
              void navigator.clipboard.writeText(turn.content);
              setCopied(true);
              setTimeout(() => setCopied(false), 1500);
            }}
          >
            <Copy size={14} />
          </IconButton>
          {onRegenerate && (
            <IconButton label="Regenerate" onClick={onRegenerate}>
              <RefreshCw size={14} />
            </IconButton>
          )}
          <IconButton
            label="Good answer"
            active={vote === "up"}
            onClick={() => setVote(vote === "up" ? null : "up")}
          >
            <ThumbsUp size={14} />
          </IconButton>
          <IconButton
            label="Bad answer"
            active={vote === "down"}
            onClick={() => setVote(vote === "down" ? null : "down")}
          >
            <ThumbsDown size={14} />
          </IconButton>

          {turn.meta && (
            <span className="ml-2 text-[11px] tabular-nums">
              {turn.meta.model}
              {turn.meta.latencyMs ? ` · ${(turn.meta.latencyMs / 1000).toFixed(1)}s` : ""}
              {turn.meta.costUsd ? ` · $${turn.meta.costUsd.toFixed(4)}` : ""}
              {turn.meta.cacheHit ? ` · ${turn.meta.cacheHit} cache hit` : ""}
            </span>
          )}
        </div>
      )}
    </article>
  );
}

function Thinking({ steps, live }: { steps: string[]; live?: boolean }) {
  const [open, setOpen] = useState(false);

  return (
    <details
      open={open}
      onToggle={(e) => setOpen((e.currentTarget as HTMLDetailsElement).open)}
      className="surface mb-3.5 overflow-hidden"
    >
      <summary className="flex cursor-pointer list-none items-center gap-2 px-3 py-2.5 text-[11.5px] font-medium text-muted transition-colors hover:text-fg">
        <Brain size={13} aria-hidden />
        <span>
          {live ? "Thinking" : `${steps.length} reasoning ${steps.length === 1 ? "step" : "steps"}`}
        </span>
        <ChevronDown
          size={13}
          aria-hidden
          className={cn("ml-auto transition-transform", open && "rotate-180")}
        />
      </summary>
      <ol className="space-y-1.5 border-t border-line px-3 py-2.5 text-[11.5px] leading-relaxed text-muted">
        {steps.map((step, i) => (
          <li key={i} className="flex gap-2">
            <span className="tabular-nums opacity-60">{i + 1}.</span>
            <span>{step}</span>
          </li>
        ))}
      </ol>
    </details>
  );
}

function Composer({
  value,
  busy,
  onChange,
  onSubmit,
  onStop,
}: {
  value: string;
  busy: boolean;
  onChange: (v: string) => void;
  onSubmit: () => void;
  onStop: () => void;
}) {
  const ref = useRef<HTMLTextAreaElement>(null);

  // Grow with the content up to a cap. A textarea that grows without limit
  // eventually leaves no room for the conversation it belongs to.
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = `${Math.min(el.scrollHeight, 200)}px`;
  }, [value]);

  return (
    <div className="bg-gradient-to-t from-bg via-bg to-transparent px-5 pb-4 pt-3">
      <div className="surface-raised mx-auto flex w-full max-w-3xl items-end gap-2 rounded-[20px] p-2 transition-shadow focus-within:border-accent/40 focus-within:shadow-lg">
        <textarea
          ref={ref}
          rows={1}
          value={value}
          onChange={(e) => onChange(e.target.value)}
          onKeyDown={(e) => {
            // Enter sends, Shift+Enter breaks the line. The reverse traps people
            // who type a multi-line question out of habit.
            if (e.key === "Enter" && !e.shiftKey) {
              e.preventDefault();
              onSubmit();
            }
          }}
          placeholder="Ask about the documents…"
          aria-label="Your question"
          className="max-h-[200px] flex-1 resize-none bg-transparent px-2.5 py-2 text-[13.5px] leading-relaxed outline-none placeholder:text-muted/80"
        />
        <button
          type="button"
          onClick={busy ? onStop : onSubmit}
          disabled={!busy && !value.trim()}
          aria-label={busy ? "Stop generating" : "Send"}
          className="accent-gradient grid h-9 w-9 shrink-0 place-items-center rounded-[14px] text-white shadow-sm transition-all hover:brightness-110 disabled:opacity-35 disabled:shadow-none"
        >
          {busy ? <Square size={15} fill="currentColor" /> : <Send size={16} />}
        </button>
      </div>
      <p className="mx-auto mt-2.5 max-w-3xl text-center text-[11px] text-muted/80">
        Answers are grounded in the indexed corpus and cite their sources.
      </p>
    </div>
  );
}

function IconButton({
  label,
  active,
  onClick,
  children,
}: {
  label: string;
  active?: boolean;
  onClick: () => void;
  children: React.ReactNode;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      title={label}
      aria-label={label}
      aria-pressed={active}
      className={cn(
        "rounded-md p-1.5 transition-colors hover:bg-line/60 hover:text-fg",
        active && "text-accent",
      )}
    >
      {children}
    </button>
  );
}

/**
 * Replace `[n]` in rendered text with a button that opens the source.
 *
 * Operates on already-rendered children rather than on the raw markdown, so a
 * literal `[1]` inside a code block or a link is left alone — replacing it in
 * the source string would corrupt both.
 */
function linkCitations(
  children: React.ReactNode,
  citations: Citation[],
  onClick: (c: Citation) => void,
): React.ReactNode {
  if (citations.length === 0) return children;

  return Array.isArray(children)
    ? children.map((child, i) =>
        typeof child === "string" ? (
          <span key={i}>{split(child)}</span>
        ) : (
          <span key={i}>{child}</span>
        ),
      )
    : typeof children === "string"
      ? split(children)
      : children;

  function split(text: string): React.ReactNode[] {
    const parts: React.ReactNode[] = [];
    const pattern = /\[(\d+)\]/g;
    let last = 0;
    let match: RegExpExecArray | null;

    while ((match = pattern.exec(text)) !== null) {
      const index = Number(match[1]);
      const citation = citations.find((c) => c.index === index);
      parts.push(text.slice(last, match.index));
      last = match.index + match[0].length;

      // A marker with no matching citation stays as literal text. Rendering a
      // dead button would tell the reader a source exists when it does not.
      if (!citation) {
        parts.push(match[0]);
        continue;
      }

      parts.push(
        <button
          key={`${match.index}-${index}`}
          type="button"
          onClick={() => onClick(citation)}
          title={citation.document_title}
          className="mx-0.5 rounded bg-accent/10 px-1 align-super text-[0.7em] font-medium text-accent hover:bg-accent/20"
        >
          {index}
        </button>,
      );
    }

    parts.push(text.slice(last));
    return parts;
  }
}
