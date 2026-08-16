import Link from "next/link";

export default function NotFound() {
  return (
    <div className="grid h-dvh place-items-center px-6 text-center">
      <div>
        <h1 className="text-lg font-medium">Not found</h1>
        <p className="mt-2 text-sm text-muted">That page does not exist.</p>
        <Link href="/" className="mt-5 inline-block text-sm text-accent hover:underline">
          Back to chat
        </Link>
      </div>
    </div>
  );
}
