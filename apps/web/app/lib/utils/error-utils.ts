import type { LanguageToolError } from "@writeo/shared";

export function getErrorType(error: LanguageToolError): string {
  return error.errorType || error.category || "Other";
}

export function groupErrorsByType(
  errors: LanguageToolError[],
): Record<string, LanguageToolError[]> {
  return errors.reduce(
    (grouped, error) => {
      const errorType = getErrorType(error);
      if (!grouped[errorType]) {
        grouped[errorType] = [];
      }
      grouped[errorType].push(error);
      return grouped;
    },
    {} as Record<string, LanguageToolError[]>,
  );
}

export function getErrorCountByType(errors: LanguageToolError[]): Record<string, number> {
  return errors.reduce(
    (counts, error) => {
      const errorType = getErrorType(error);
      counts[errorType] = (counts[errorType] || 0) + 1;
      return counts;
    },
    {} as Record<string, number>,
  );
}

export function getErrorSeverityColor(error: LanguageToolError): string {
  if (error.highConfidence === false) {
    return "#f59e0b";
  }

  switch (error.severity) {
    case "error":
      return "#ef4444";
    case "warning":
      return "#f59e0b";
    default:
      return "#6b7280";
  }
}

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

export function formatErrorMessage(error: LanguageToolError): string {
  if (error.message) {
    return error.message;
  }

  const type = error.errorType || error.category;
  if (type) {
    return `${type} ${error.errorType ? "error" : "issue"}`;
  }

  return "Error detected";
}

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
