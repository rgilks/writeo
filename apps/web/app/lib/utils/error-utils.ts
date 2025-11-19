import type { LanguageToolError } from "@writeo/shared";

/**
 * Group errors by type
 */
export function groupErrorsByType(
  errors: LanguageToolError[]
): Record<string, LanguageToolError[]> {
  const grouped: Record<string, LanguageToolError[]> = {};

  errors.forEach((error) => {
    const errorType = error.errorType || error.category || "Other";
    if (!grouped[errorType]) {
      grouped[errorType] = [];
    }
    grouped[errorType].push(error);
  });

  return grouped;
}

/**
 * Get error count by type
 */
export function getErrorCountByType(errors: LanguageToolError[]): Record<string, number> {
  const counts: Record<string, number> = {};

  errors.forEach((error) => {
    const errorType = error.errorType || error.category || "Other";
    counts[errorType] = (counts[errorType] || 0) + 1;
  });

  return counts;
}

/**
 * Get error severity color
 */
export function getErrorSeverityColor(error: LanguageToolError): string {
  if (error.highConfidence === false) {
    return "#f59e0b"; // Amber for experimental/low confidence
  }

  if (error.severity === "error") {
    return "#ef4444"; // Red for errors
  }

  if (error.severity === "warning") {
    return "#f59e0b"; // Amber for warnings
  }

  return "#6b7280"; // Grey for info/other
}

/**
 * Get error category icon (simple text representation)
 */
export function getErrorCategoryIcon(category: string): string {
  const icons: Record<string, string> = {
    GRAMMAR: "📝",
    SPELLING: "✏️",
    TYPOS: "🔤",
    STYLE: "✨",
    PUNCTUATION: ".",
    OTHER: "⚠️",
  };

  return icons[category.toUpperCase()] || icons.OTHER;
}

/**
 * Format error message for display
 */
export function formatErrorMessage(error: LanguageToolError): string {
  if (error.message) {
    return error.message;
  }

  if (error.errorType) {
    return `${error.errorType} error`;
  }

  if (error.category) {
    return `${error.category} issue`;
  }

  return "Error detected";
}

/**
 * Get learning tip for error type
 */
export function getLearningTipForErrorType(errorType: string): string | null {
  const tips: Record<string, string> = {
    GRAMMAR: "Review grammar rules and practice with similar sentences.",
    SPELLING: "Use a dictionary or spell-checker to verify spelling.",
    TYPOS: "Read your text carefully to catch typos.",
    STYLE: "Consider using more varied sentence structures and vocabulary.",
    PUNCTUATION: "Review punctuation rules, especially commas and periods.",
  };

  return tips[errorType.toUpperCase()] || null;
}
