import type { Metadata } from "next";
import "./globals.css";
import { Nav } from "@/components/nav";

export const metadata: Metadata = {
  title: "AgenticRAG",
  description: "Multi-tenant agentic RAG with evaluation-gated deployments.",
};

/**
 * The application shell.
 *
 * The theme is applied by an inline script that runs before paint. Reading the
 * preference in an effect instead would render the light theme first and then
 * repaint dark, which is the flash every dark-mode implementation gets wrong
 * once.
 */
export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" suppressHydrationWarning>
      <head>
        <script
          dangerouslySetInnerHTML={{
            __html: `(function(){try{
              var stored = localStorage.getItem('theme');
              var dark = stored ? stored === 'dark'
                : window.matchMedia('(prefers-color-scheme: dark)').matches;
              if (dark) document.documentElement.classList.add('dark');
            }catch(e){}})();`,
          }}
        />
      </head>
      <body className="min-h-dvh antialiased">
        {/* Keyboard users should not have to tab through the whole sidebar to
            reach the conversation. */}
        <a
          href="#main"
          className="sr-only focus:not-sr-only focus:absolute focus:left-4 focus:top-4 focus:z-50 focus:rounded-md focus:bg-accent focus:px-3 focus:py-2 focus:text-white"
        >
          Skip to content
        </a>
        <div className="flex min-h-dvh">
          <Nav />
          <main id="main" className="flex min-w-0 flex-1 flex-col">
            {children}
          </main>
        </div>
      </body>
    </html>
  );
}
