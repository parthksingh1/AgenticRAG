"use client";

import { useEffect, useRef, useState } from "react";

/**
 * The knowledge-graph explorer.
 *
 * Cytoscape is imported dynamically inside an effect because it touches
 * `document` at module scope, which throws during server rendering. The layout
 * is also the expensive part of the page, and there is no reason to ship it to
 * someone who never opens this route.
 */
export default function GraphPage() {
  const containerRef = useRef<HTMLDivElement>(null);
  const [status, setStatus] = useState<"loading" | "ready" | "empty" | "error">("loading");
  const [message, setMessage] = useState("");

  useEffect(() => {
    let destroyed = false;
    let instance: { destroy: () => void } | null = null;

    async function draw() {
      try {
        const [{ default: cytoscape }, response] = await Promise.all([
          import("cytoscape"),
          fetch("/api/backend/graph/neighbourhood?limit=150"),
        ]);
        if (destroyed) return;

        if (!response.ok) throw new Error(`The graph API returned ${response.status}.`);
        const body = (await response.json()) as {
          nodes: { id: string; label: string; type: string }[];
          edges: { source: string; target: string; label: string }[];
        };

        if (!body.nodes?.length) {
          setStatus("empty");
          return;
        }

        instance = cytoscape({
          container: containerRef.current,
          elements: [
            ...body.nodes.map((n) => ({ data: { id: n.id, label: n.label, type: n.type } })),
            ...body.edges.map((e) => ({
              data: { source: e.source, target: e.target, label: e.label },
            })),
          ],
          style: [
            {
              selector: "node",
              style: {
                "background-color": "#818cf8",
                label: "data(label)",
                "font-size": "9px",
                color: "#9ca3af",
                "text-valign": "bottom",
                "text-margin-y": 4,
                width: 18,
                height: 18,
              },
            },
            {
              selector: "edge",
              style: {
                width: 1,
                "line-color": "#374151",
                "target-arrow-color": "#374151",
                "target-arrow-shape": "triangle",
                "curve-style": "bezier",
                label: "data(label)",
                "font-size": "7px",
                color: "#6b7280",
              },
            },
          ],
          // cose rather than a grid: the point of the view is which entities
          // cluster together, and a grid destroys exactly that information.
          layout: { name: "cose", animate: false, nodeRepulsion: () => 8000 },
        });
        setStatus("ready");
      } catch (e) {
        if (destroyed) return;
        setMessage(e instanceof Error ? e.message : "The graph could not be loaded.");
        setStatus("error");
      }
    }

    void draw();
    return () => {
      destroyed = true;
      instance?.destroy();
    };
  }, []);

  return (
    <div className="flex h-dvh flex-col">
      <header className="border-b border-line px-5 py-3">
        <h1 className="text-sm font-semibold">Knowledge graph</h1>
        <p className="mt-0.5 text-xs text-muted">
          Entities and relations extracted during ingestion. Answers multi-hop
          questions that chunk retrieval cannot.
        </p>
      </header>
      <div className="relative flex-1">
        <div ref={containerRef} className="absolute inset-0" />
        {status !== "ready" && (
          <p className="absolute inset-0 grid place-items-center text-sm text-muted">
            {status === "loading" && "Loading the graph…"}
            {status === "empty" &&
              "No graph yet. Enable graph extraction for the workspace and re-ingest."}
            {status === "error" && message}
          </p>
        )}
      </div>
    </div>
  );
}
