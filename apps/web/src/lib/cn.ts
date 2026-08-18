import { clsx, type ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";

/**
 * Merge class names, letting a later Tailwind class win over an earlier one.
 *
 * Plain concatenation leaves both `p-2` and `p-4` in the string and the winner
 * depends on stylesheet order, which is not something a component author should
 * have to reason about.
 */
export function cn(...inputs: ClassValue[]): string {
  return twMerge(clsx(inputs));
}
