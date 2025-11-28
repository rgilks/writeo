/**
 * Text utility functions
 */

/**
 * Pluralizes a word based on count
 * @param count - The count to check
 * @param singular - The singular form of the word
 * @returns The pluralized word (adds 's' if count !== 1)
 */
export function pluralize(count: number, singular: string): string {
  return count === 1 ? singular : `${singular}s`;
}
