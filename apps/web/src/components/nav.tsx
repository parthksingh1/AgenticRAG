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
      className="flex w-14 shrink-0 flex-col items-center gap-1 border-r border-line bg-card py-3 sm:w-52 sm:items-stretch sm:px-3"
    >
      <div className="mb-4 px-2 text-sm font-semibold tracking-tight">
        <span className="hidden sm:inline">AgenticRAG</span>
        <span className="sm:hidden" aria-hidden>
          AR
        </span>
      </div>

      {LINKS.map(({ href, label, icon: Icon }) => {
        const active = href === "/" ? pathname === "/" : pathname.startsWith(href);
        return (
          <Link
            key={href}
            href={href}
            aria-current={active ? "page" : undefined}
            className={cn(
              "flex items-center gap-3 rounded-lg px-2.5 py-2 text-sm transition-colors",
              active
                ? "bg-accent/10 font-medium text-accent"
                : "text-muted hover:bg-line/50 hover:text-fg",
            )}
          >
            <Icon size={17} aria-hidden />
            <span className="hidden sm:inline">{label}</span>
            <span className="sr-only sm:hidden">{label}</span>
          </Link>
        );
      })}

      <div className="mt-auto">
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
      className="flex w-full items-center gap-3 rounded-lg px-2.5 py-2 text-sm text-muted transition-colors hover:bg-line/50 hover:text-fg"
    >
      {mounted ? (
        dark ? (
          <Sun size={17} aria-hidden />
        ) : (
          <Moon size={17} aria-hidden />
        )
      ) : (
        <span className="block h-[17px] w-[17px]" />
      )}
      <span className="hidden sm:inline">{mounted && dark ? "Light" : "Dark"}</span>
    </button>
  );
}
