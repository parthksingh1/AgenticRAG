"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { FileText, Trash2, Upload } from "lucide-react";
import { api, type Doc } from "@/lib/api";
import { cn } from "@/lib/cn";

/** Terminal statuses. Anything else is still moving and worth polling for. */
const SETTLED = new Set(["indexed", "failed"]);

const STATUS_STYLE: Record<string, string> = {
  indexed: "text-emerald-500 bg-emerald-500/10",
  failed: "text-red-500 bg-red-500/10",
  queued: "text-amber-500 bg-amber-500/10",
  parsing: "text-blue-500 bg-blue-500/10",
  chunking: "text-blue-500 bg-blue-500/10",
  embedding: "text-blue-500 bg-blue-500/10",
};

export default function DocumentsPage() {
  const [documents, setDocuments] = useState<Doc[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [dragging, setDragging] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  const refresh = useCallback(async () => {
    try {
      setDocuments(await api.documents());
      setError(null);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Could not load documents.");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  // Poll only while something is still processing. A fixed interval keeps
  // hitting the API forever on an idle tab for no reason.
  useEffect(() => {
    const pending = documents.some((d) => !SETTLED.has(d.status));
    if (!pending) return;
    const timer = setInterval(() => void refresh(), 3000);
    return () => clearInterval(timer);
  }, [documents, refresh]);

  async function upload(files: FileList | null) {
    if (!files?.length) return;
    for (const file of Array.from(files)) {
      try {
        await uploadOne(file);
      } catch (e) {
        setError(`${file.name}: ${e instanceof Error ? e.message : "upload failed"}`);
      }
    }
    void refresh();
  }

  return (
    <div className="flex h-dvh flex-col">
      <header className="flex items-center justify-between border-b border-line px-5 py-3">
        <h1 className="text-sm font-semibold">Documents</h1>
        <button
          type="button"
          onClick={() => inputRef.current?.click()}
          className="flex items-center gap-1.5 rounded-lg bg-accent px-3 py-1.5 text-xs font-medium text-white"
        >
          <Upload size={13} aria-hidden /> Upload
        </button>
        <input
          ref={inputRef}
          type="file"
          multiple
          className="sr-only"
          onChange={(e) => void upload(e.target.files)}
          accept=".pdf,.docx,.pptx,.txt,.md,.html,.csv,.xlsx,.png,.jpg"
        />
      </header>

      <div
        className="flex-1 overflow-y-auto px-5 py-5"
        onDragOver={(e) => {
          e.preventDefault();
          setDragging(true);
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={(e) => {
          e.preventDefault();
          setDragging(false);
          void upload(e.dataTransfer.files);
        }}
      >
        <div className="mx-auto w-full max-w-4xl">
          <div
            className={cn(
              "mb-5 rounded-xl border-2 border-dashed border-line p-8 text-center transition-colors",
              dragging && "border-accent bg-accent/5",
            )}
          >
            <FileText size={22} className="mx-auto text-muted" aria-hidden />
            <p className="mt-2 text-sm">Drop files here, or use Upload</p>
            <p className="mt-1 text-xs text-muted">
              PDF, DOCX, PPTX, HTML, Markdown, CSV, XLSX and images
            </p>
          </div>

          {error && (
            <p className="mb-4 rounded-lg border border-red-500/30 bg-red-500/5 px-3 py-2 text-sm text-red-500">
              {error}
            </p>
          )}

          {loading ? (
            <p className="text-sm text-muted">Loading…</p>
          ) : documents.length === 0 ? (
            <p className="text-sm text-muted">
              Nothing indexed yet. Run <code className="text-fg">make seed</code> for the demo
              corpus.
            </p>
          ) : (
            <ul className="divide-y divide-line rounded-xl border border-line">
              {documents.map((doc) => (
                <li key={doc.id} className="flex items-center gap-3 px-4 py-3">
                  <FileText size={16} className="shrink-0 text-muted" aria-hidden />
                  <div className="min-w-0 flex-1">
                    <p className="truncate text-sm font-medium">{doc.title}</p>
                    <p className="truncate text-xs text-muted">
                      {doc.chunk_count} chunks
                      {doc.page_count ? ` · ${doc.page_count} pages` : ""}
                      {doc.byte_size ? ` · ${formatBytes(doc.byte_size)}` : ""}
                      {doc.error_message ? ` · ${doc.error_message}` : ""}
                    </p>
                  </div>
                  <span
                    className={cn(
                      "shrink-0 rounded-full px-2 py-0.5 text-[11px] font-medium",
                      STATUS_STYLE[doc.status] ?? "bg-line text-muted",
                    )}
                  >
                    {doc.status}
                  </span>
                  <button
                    type="button"
                    aria-label={`Delete ${doc.title}`}
                    onClick={async () => {
                      await api.deleteDocument(doc.id);
                      void refresh();
                    }}
                    className="rounded-md p-1.5 text-muted hover:bg-line/60 hover:text-red-500"
                  >
                    <Trash2 size={14} />
                  </button>
                </li>
              ))}
            </ul>
          )}
        </div>
      </div>
    </div>
  );
}

/**
 * The three-step upload: ask for a slot, PUT the bytes straight to storage,
 * then confirm.
 *
 * The bytes never pass through the API. Proxying a 200MB PDF through a request
 * handler ties up a worker for the whole transfer and caps the file size at
 * whatever the platform's request limit happens to be.
 */
async function uploadOne(file: File): Promise<void> {
  const slot = await fetch("/api/backend/documents/upload", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      filename: file.name,
      content_type: file.type || "application/octet-stream",
      size_bytes: file.size,
    }),
  });
  if (!slot.ok) throw new Error((await slot.json().catch(() => ({}))).detail ?? "could not start");

  const { document_id, upload_url, required_headers } = await slot.json();

  const put = await fetch(upload_url, {
    method: "PUT",
    body: file,
    headers: required_headers ?? { "Content-Type": file.type },
  });
  if (!put.ok) throw new Error("the upload to storage failed");

  const confirm = await fetch(`/api/backend/documents/${document_id}/confirm`, { method: "POST" });
  if (!confirm.ok) throw new Error("the upload could not be confirmed");
}

function formatBytes(bytes: number): string {
  const units = ["B", "KB", "MB", "GB"];
  let value = bytes;
  let unit = 0;
  while (value >= 1024 && unit < units.length - 1) {
    value /= 1024;
    unit += 1;
  }
  return `${value.toFixed(unit === 0 ? 0 : 1)} ${units[unit]}`;
}
