/**
 * Text utility functions - shared between frontend and backend
 */

/**
 * Counts words in a text string
 *
 * @param text - Text to count words in
 * @returns Number of words (non-empty strings separated by whitespace)
 *
 * @example
 * ```typescript
 * countWords("Hello world") // 2
 * countWords("  Hello   world  ") // 2
 * countWords("") // 0
 * ```
 */
export function countWords(text: string): number {
  if (!text || typeof text !== "string") return 0;
  return text
    .trim()
    .split(/\s+/)
    .filter((w) => w.length > 0).length;
}
