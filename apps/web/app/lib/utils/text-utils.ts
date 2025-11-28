/**
 * Text utility functions
 */

/**
 * Pluralizes a word based on count
 * @param count - The count to check
 * @param singular - The singular form of the word
 * @param plural - Optional custom plural form (for irregular plurals)
 * @returns The singular form if count is 1, otherwise the plural form
 */
export function pluralize(count: number, singular: string, plural?: string): string {
  return count === 1 ? singular : (plural ?? `${singular}s`);
}
