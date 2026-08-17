"use client";

import { useEffect, useRef } from "react";
import { ExternalLink, X } from "lucide-react";
import type { Citation } from "@/lib/api";

/**
 * The source panel opened by a citation.
 *
 * Focus moves into the panel on open and returns to whatever opened it on close.
 * Without that, a keyboard user opens the panel and their focus is still behind
 * it, tabbing through a conversation they can no longer see.
 */
export function CitationSheet({
  citation,
  onClose,
}: {
  citation: Citation | null;
  onClose: () => void;
}) {
  const panelRef = useRef<HTMLDivElement>(null);
  const returnFocusRef = useRef<HTMLElement | null>(null);

  useEffect(() => {
    if (!citation) return;
    returnFocusRef.current = document.activeElement as HTMLElement | null;
    panelRef.current?.focus();

    function onKey(e: KeyboardEvent) {
      if (e.key === "Escape") onClose();
    }
    document.addEventListener("keydown", onKey);
    return () => {
      document.removeEventListener("keydown", onKey);
      returnFocusRef.current?.focus();
    };
  }, [citation, onClose]);

  if (!citation) return null;

  return (
    <>
      <div
        className="fixed inset-0 z-40 bg-black/30 backdrop-blur-[1px]"
        onClick={onClose}
        aria-hidden
      />
      <div
        ref={panelRef}
        role="dialog"
        aria-modal="true"
        aria-label={`Source: ${citation.document_title}`}
        tabIndex={-1}
        className="fixed right-0 top-0 z-50 flex h-dvh w-full max-w-md flex-col border-l border-line bg-bg shadow-2xl outline-none"
      >
        <header className="flex items-start justify-between gap-3 border-b border-line px-4 py-3">
          <div className="min-w-0">
            <p className="text-xs text-muted">Source [{citation.index}]</p>
            <h2 className="truncate text-sm font-medium">{citation.document_title}</h2>
            {citation.section_path.length > 0 && (
              <p className="mt-0.5 truncate text-xs text-muted">
                {citation.section_path.join(" › ")}
              </p>
            )}
          </div>
          <button
            type="button"
            onClick={onClose}
            aria-label="Close source panel"
            className="rounded-md p-1.5 text-muted hover:bg-line/60 hover:text-fg"
          >
            <X size={16} />
          </button>
        </header>

        <div className="flex-1 overflow-y-auto px-4 py-4">
          <blockquote className="border-l-2 border-accent/60 bg-card px-3 py-2 text-sm leading-relaxed">
            {citation.snippet}
          </blockquote>

          <dl className="mt-4 space-y-1 text-xs text-muted">
            {citation.page_number != null && (
              <div className="flex justify-between">
                <dt>Page</dt>
                <dd className="tabular-nums">{citation.page_number}</dd>
              </div>
            )}
            {citation.score != null && (
              <div className="flex justify-between">
                <dt>Retrieval score</dt>
                <dd className="tabular-nums">{citation.score.toFixed(4)}</dd>
              </div>
            )}
            <div className="flex justify-between">
              <dt>Chunk</dt>
              <dd className="truncate font-mono">{citation.chunk_id}</dd>
            </div>
          </dl>
        </div>

        <footer className="border-t border-line px-4 py-3">
          <a
            href={`/documents/${citation.document_id}`}
            className="inline-flex items-center gap-1.5 text-sm text-accent hover:underline"
          >
            Open the document <ExternalLink size={13} aria-hidden />
          </a>
        </footer>
      </div>
    </>
  );
}
