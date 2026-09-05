/**
 * The demo-mode banner.
 *
 * Deliberately not dismissible and deliberately at the top of every page. A
 * demo running on fixtures that does not say so is a misrepresentation, and the
 * person most likely to be misled by it is someone evaluating the project. The
 * cost of the banner is a strip of colour; the cost of omitting it is that
 * nothing else on the page can be taken at face value.
 */

import { isDemo } from "@/lib/demo";

export function DemoBanner() {
  if (!isDemo) return null;

  return (
    <div className="border-b border-amber-500/30 bg-amber-500/10 px-4 py-2 text-xs text-amber-900 dark:text-amber-200">
      <span className="font-semibold">Demo build.</span>{" "}
      Fixture data, no backend and no model calls — the interface is real, the answers are
      canned, and the dashboard figures are illustrative rather than measured.{" "}
      <a
        href="https://github.com/parthksingh1/AgenticRAG#running-it"
        className="underline underline-offset-2 hover:no-underline"
        target="_blank"
        rel="noreferrer"
      >
        Run the real thing with <code>docker compose up</code>
      </a>
      .
    </div>
  );
}
