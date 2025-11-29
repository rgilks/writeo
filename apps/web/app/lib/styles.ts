/**
 * Shared style utilities and constants
 * Provides consistent access to CSS variables and common style patterns
 */

/**
 * Spacing values from CSS variables (8pt grid system)
 */
export const spacing = {
  xs: "var(--spacing-xs)",
  sm: "var(--spacing-sm)",
  md: "var(--spacing-md)",
  lg: "var(--spacing-lg)",
  xl: "var(--spacing-xl)",
  "2xl": "var(--spacing-2xl)",
  "3xl": "var(--spacing-3xl)",
} as const;

/**
 * Color values from CSS variables
 */
export const colors = {
  primary: "var(--primary-color)",
  primaryHover: "var(--primary-hover)",
  secondaryAccent: "var(--secondary-accent)",
  warmAccent: "var(--warm-accent)",
  error: "var(--error-color)",
  textPrimary: "var(--text-primary)",
  textSecondary: "var(--text-secondary)",
  bgPrimary: "var(--bg-primary)",
  bgSecondary: "var(--bg-secondary)",
  bgTertiary: "var(--bg-tertiary)",
  border: "var(--border-color)",
  success: "var(--success-color)",
  successBg: "var(--success-bg)",
  successText: "var(--success-text)",
  successBorder: "var(--success-border)",
  errorBg: "var(--error-bg)",
  errorText: "var(--error-text)",
  errorBorder: "var(--error-border)",
  primaryBgLight: "var(--primary-bg-light)",
  primaryBorderLight: "var(--primary-border-light)",
  successBgLight: "var(--success-bg-light)",
  successBorderLight: "var(--success-border-light)",
} as const;

/**
 * Common style patterns for reuse across components
 */
export const commonStyles = {
  container: {
    padding: spacing.lg,
    backgroundColor: colors.bgSecondary,
    borderRadius: "var(--border-radius-lg)",
  },
  text: {
    fontSize: "16px",
    lineHeight: "1.6",
    color: colors.textPrimary,
  },
  textSecondary: {
    fontSize: "16px",
    lineHeight: "1.6",
    color: colors.textSecondary,
  },
  button: {
    fontSize: "14px",
    padding: `${spacing.sm} ${spacing.md}`,
    minHeight: "44px",
  },
  focusArea: {
    padding: spacing.md,
    backgroundColor: "rgba(102, 126, 234, 0.1)",
    borderLeft: "4px solid var(--primary-color)",
    borderRadius: spacing.xs,
    fontSize: "14px",
    marginBottom: spacing.md,
    lineHeight: "1.5",
  },
} as const;

/**
 * Helper to create style objects with CSS variables
 */
export function createStyle<T extends Record<string, string | number>>(base: T): T {
  return base;
}
