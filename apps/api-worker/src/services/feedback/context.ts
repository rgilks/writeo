/**
 * Context building utilities for feedback prompts
 */

export function buildEssayContext(essayScores?: {
  overall?: number;
  dimensions?: { TA?: number; CC?: number; Vocab?: number; Grammar?: number; Overall?: number };
  label?: string;
}): string {
  if (!essayScores) return "";
  return `\n\nAssessment Results:
- Overall Score: ${essayScores.overall ?? essayScores.dimensions?.Overall ?? "N/A"} / 9.0
- CEFR Level: ${essayScores.label ?? "N/A"}
- Task Achievement (TA): ${essayScores.dimensions?.TA ?? "N/A"} / 9.0
- Coherence & Cohesion (CC): ${essayScores.dimensions?.CC ?? "N/A"} / 9.0
- Vocabulary: ${essayScores.dimensions?.Vocab ?? "N/A"} / 9.0
- Grammar: ${essayScores.dimensions?.Grammar ?? "N/A"} / 9.0`;
}

export function buildGrammarContext(
  languageToolErrors?: Array<{
    message: string;
    category: string;
    suggestions?: string[];
  }>,
  llmErrors?: Array<{
    message: string;
    category: string;
    suggestions?: string[];
  }>
): string {
  const allErrors = [...(languageToolErrors || []), ...(llmErrors || [])];
  if (allErrors.length > 0) {
    const errorSummary = allErrors
      .slice(0, 10)
      .map((err, idx) => {
        const source = languageToolErrors?.includes(err as any) ? "LanguageTool" : "AI Assessment";
        return `${idx + 1}. [${source}] ${err.message} (${err.category})${err.suggestions ? ` - Suggestions: ${err.suggestions.slice(0, 2).join(", ")}` : ""}`;
      })
      .join("\n");
    return `\n\nGrammar & Language Issues Found (${allErrors.length} total):\n${errorSummary}${allErrors.length > 10 ? `\n... and ${allErrors.length - 10} more issues` : ""}`;
  } else if (languageToolErrors !== undefined || llmErrors !== undefined) {
    return "\n\nGrammar & Language: No errors detected by LanguageTool or AI assessment.";
  }
  return "";
}

export function buildRelevanceContext(relevanceCheck?: {
  addressesQuestion: boolean;
  score: number;
  threshold: number;
}): string {
  if (!relevanceCheck) return "";
  return `\n\nAnswer Relevance: ${relevanceCheck.addressesQuestion ? "The answer addresses the question" : "The answer may not fully address the question"} (similarity score: ${relevanceCheck.score.toFixed(2)}, threshold: ${relevanceCheck.threshold}).`;
}

export function buildTeacherScoreContext(essayScores?: {
  overall?: number;
  dimensions?: { TA?: number; CC?: number; Vocab?: number; Grammar?: number; Overall?: number };
}): string {
  if (!essayScores) return "";
  const overall = essayScores.overall ?? essayScores.dimensions?.Overall ?? 0;
  const lowestDim = Object.entries(essayScores.dimensions || {})
    .filter(([k]) => k !== "Overall")
    .sort(([, a], [, b]) => (a ?? 0) - (b ?? 0))[0];
  return `\n\nStudent's performance:\n- Overall score: ${overall.toFixed(1)} / 9.0${lowestDim ? `\n- Weakest area: ${lowestDim[0]} (${lowestDim[1]?.toFixed(1)} / 9.0)` : ""}`;
}

export function buildTeacherErrorContext(
  languageToolErrors?: Array<{ errorType?: string; category: string }>,
  llmErrors?: Array<{ errorType?: string; category: string }>
): string {
  const allErrors = [...(languageToolErrors || []), ...(llmErrors || [])];
  if (allErrors.length > 0) {
    const errorTypes = new Map<string, number>();
    allErrors.forEach((err) => {
      const type = err.errorType || err.category || "Other";
      errorTypes.set(type, (errorTypes.get(type) || 0) + 1);
    });
    const topErrorTypes = Array.from(errorTypes.entries())
      .sort((a, b) => b[1] - a[1])
      .slice(0, 2)
      .map(([type, count]) => `${type} (${count})`)
      .join(", ");
    return `\n\nFound ${allErrors.length} grammar/language issue${allErrors.length > 1 ? "s" : ""}${topErrorTypes ? `, mostly: ${topErrorTypes}` : ""}.`;
  }
  return "";
}

export function buildTeacherRelevanceContext(relevanceCheck?: {
  addressesQuestion: boolean;
  score: number;
}): string {
  if (!relevanceCheck) return "";
  const relevancePercent = (relevanceCheck.score * 100).toFixed(0);
  if (relevanceCheck.addressesQuestion) {
    return `\n\nThe answer addresses the question well (relevance: ${relevancePercent}%).`;
  }
  return `\n\nThe answer may not fully address the question (relevance: ${relevancePercent}% - consider if all parts of the question were answered).`;
}

export function buildWordCountContext(answerText: string): string {
  const wordCount = answerText.split(/\s+/).filter((w) => w.length > 0).length;
  return wordCount > 0 ? `\n\nEssay length: ${wordCount} word${wordCount !== 1 ? "s" : ""}.` : "";
}

export function getLowestDimension(essayScores?: {
  dimensions?: { TA?: number; CC?: number; Vocab?: number; Grammar?: number; Overall?: number };
}): [string, number] | undefined {
  if (!essayScores?.dimensions) return undefined;
  return Object.entries(essayScores.dimensions)
    .filter(([k]) => k !== "Overall")
    .sort(([, a], [, b]) => (a ?? 0) - (b ?? 0))[0] as [string, number] | undefined;
}

export function getFocusArea(dimension?: string): string {
  if (!dimension) return "this area";
  const simpleNames: Record<string, string> = {
    TA: "answering all parts of the question completely and developing your ideas fully",
    CC: "organizing your ideas into clear paragraphs and connecting them smoothly with linking words",
    Vocab: "using a wider range of vocabulary and avoiding repetition",
    Grammar: "improving grammar accuracy and using more varied sentence structures",
  };
  return simpleNames[dimension] || "this area";
}
