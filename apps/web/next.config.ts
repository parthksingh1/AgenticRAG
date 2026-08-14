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
  output: "standalone",
};

export default config;
