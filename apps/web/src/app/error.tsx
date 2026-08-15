"use client";

/**
 * The route error boundary.
 *
 * Shows the message rather than a generic apology: "Could not reach the API"
 * tells someone to check whether the backend is running, which "Something went
 * wrong" does not.
 */
export default function Error({ error, reset }: { error: Error; reset: () => void }) {
  return (
    <div className="grid h-dvh place-items-center px-6">
      <div className="max-w-md text-center">
        <h1 className="text-lg font-medium">This page failed to load</h1>
        <p className="mt-2 text-sm text-muted">{error.message}</p>
        <button
          type="button"
          onClick={reset}
          className="mt-5 rounded-lg bg-accent px-4 py-2 text-sm text-white"
        >
          Try again
        </button>
      </div>
    </div>
  );
}
