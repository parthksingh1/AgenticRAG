"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import {
  BarChart3,
  FileText,
  FlaskConical,
  MessageSquare,
  Moon,
  Share2,
  Sun,
} from "lucide-react";
import { useEffect, useState } from "react";
import { cn } from "@/lib/cn";

const LINKS = [
  { href: "/", label: "Chat", icon: MessageSquare },
  { href: "/documents", label: "Documents", icon: FileText },
  { href: "/playground", label: "Playground", icon: FlaskConical },
  { href: "/graph", label: "Graph", icon: Share2 },
  { href: "/admin", label: "Admin", icon: BarChart3 },
] as const;

/** The primary navigation, with a theme toggle. */
export function Nav() {
  const pathname = usePathname();

  return (
    <nav
      aria-label="Primary"
      className="flex w-16 shrink-0 flex-col items-center gap-0.5 border-r border-line bg-card py-4 sm:w-60 sm:items-stretch sm:px-3"
    >
      <div className="mb-6 flex items-center gap-2.5 px-1.5 sm:px-2">
        <span
          aria-hidden
          className="accent-gradient grid h-8 w-8 shrink-0 place-items-center rounded-[10px] text-[13px] font-bold text-white shadow-sm"
        >
          AR
        </span>
        <span className="hidden min-w-0 sm:block">
          <span className="block truncate text-[13.5px] font-semibold leading-tight tracking-tight">
            AgenticRAG
          </span>
          <span className="block truncate text-[11px] leading-tight text-muted">
            Evaluation-gated RAG
          </span>
        </span>
      </div>

      <p className="mb-1 hidden px-2.5 text-[10.5px] font-semibold uppercase tracking-[0.08em] text-muted/70 sm:block">
        Workspace
      </p>

      {LINKS.map(({ href, label, icon: Icon }) => {
        const active = href === "/" ? pathname === "/" : pathname.startsWith(href);
        return (
          <Link
            key={href}
            href={href}
            aria-current={active ? "page" : undefined}
            className={cn(
              "group relative flex items-center gap-3 rounded-[10px] px-2.5 py-[9px] text-[13.5px] transition-all",
              active
                ? "bg-accent/[0.09] font-medium text-accent shadow-[inset_0_0_0_1px_rgb(var(--accent)/0.14)]"
                : "text-muted hover:bg-line/60 hover:text-fg",
            )}
          >
            {/* A left rail on the active item. The tint alone is legible but
                not locating; the rail is what the eye returns to. */}
            <span
              aria-hidden
              className={cn(
                "absolute left-0 top-1/2 h-4 w-[2.5px] -translate-y-1/2 rounded-r-full transition-opacity",
                active ? "accent-gradient opacity-100" : "opacity-0",
              )}
            />
            <Icon size={16.5} aria-hidden className="shrink-0" />
            <span className="hidden sm:inline">{label}</span>
            <span className="sr-only sm:hidden">{label}</span>
          </Link>
        );
      })}

      <div className="mt-auto border-t border-line pt-2">
        <ThemeToggle />
      </div>
    </nav>
  );
}

/**
 * Light/dark toggle.
 *
 * Mounted state is tracked so the button renders nothing on the server: the
 * server cannot know the user's stored preference, so rendering an icon there
 * guarantees a hydration mismatch on half of all page loads.
 */
function ThemeToggle() {
  const [mounted, setMounted] = useState(false);
  const [dark, setDark] = useState(false);

  useEffect(() => {
    setMounted(true);
    setDark(document.documentElement.classList.contains("dark"));
  }, []);

  function toggle() {
    const next = !dark;
    setDark(next);
    document.documentElement.classList.toggle("dark", next);
    try {
      localStorage.setItem("theme", next ? "dark" : "light");
    } catch {
      // Private browsing blocks storage. The toggle still works for this
      // session; it just will not be remembered.
    }
  }

  return (
    <button
      type="button"
      onClick={toggle}
      aria-label={dark ? "Switch to light theme" : "Switch to dark theme"}
      className="flex w-full items-center gap-3 rounded-[10px] px-2.5 py-[9px] text-[13.5px] text-muted transition-colors hover:bg-line/60 hover:text-fg"
    >
      {mounted ? (
        dark ? (
          <Sun size={16.5} aria-hidden />
        ) : (
          <Moon size={16.5} aria-hidden />
        )
      ) : (
        <span className="block h-[16.5px] w-[16.5px]" />
      )}
      <span className="hidden sm:inline">{mounted && dark ? "Light" : "Dark"}</span>
    </button>
  );
}
