import type { NextConfig } from "next";

const config: NextConfig = {
  reactStrictMode: true,
  // The API is a separate service. Proxying through Next rather than calling it
  // from the browser keeps the session cookie same-origin and means no CORS
  // configuration has to be kept in step across two deployments.
  async rewrites() {
    const api = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";
    return [
      { source: "/api/backend/:path*", destination: `${api}/api/:path*` },
      { source: "/v1/:path*", destination: `${api}/v1/:path*` },
    ];
  },
  // Standalone output: the Docker runner copies .next/standalone and runs
  // server.js, which is what keeps the production image small.
  //
  // Vercel builds its own serverless output and warns that standalone is
  // ignored there, so it is switched off when Vercel sets VERCEL=1. Leaving it
  // on is not fatal, but a build that prints a warning about its own
  // configuration invites the reader to wonder what else was left unchecked.
  ...(process.env.VERCEL ? {} : { output: "standalone" as const }),
};

export default config;
