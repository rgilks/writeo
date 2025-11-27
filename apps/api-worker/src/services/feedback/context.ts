/**
 * Context building utilities for feedback prompts
 */

import { countWords } from "@writeo/shared";
import type { EssayScores, FeedbackError, RelevanceCheck } from "./types";

export function buildEssayContext(essayScores?: EssayScores): string {
  const normalized = normalizeEssayScores(essayScores);
  if (!normalized) return "";
  return `\n\nAssessment Results:
- Overall Score: ${normalized.overall ?? "N/A"} / 9.0
- CEFR Level: ${normalized.label ?? "N/A"}
- Task Achievement (TA): ${normalized.dimensions?.TA ?? "N/A"} / 9.0
- Coherence & Cohesion (CC): ${normalized.dimensions?.CC ?? "N/A"} / 9.0
- Vocabulary: ${normalized.dimensions?.Vocab ?? "N/A"} / 9.0
- Grammar: ${normalized.dimensions?.Grammar ?? "N/A"} / 9.0`;
}

export function buildGrammarContext(
  languageToolErrors?: FeedbackError[],
  llmErrors?: FeedbackError[],
): string {
  const allErrors = [...(languageToolErrors || []), ...(llmErrors || [])];
  if (allErrors.length > 0) {
    const errorSummary = allErrors
      .slice(0, 10)
      .map((err, idx) => {
        const source = languageToolErrors?.includes(err as any) ? "LanguageTool" : "AI Assessment";
        return `${idx + 1}. [${source}] ${err.message} (${err.category})${
          err.suggestions ? ` - Suggestions: ${err.suggestions.slice(0, 2).join(", ")}` : ""
        }`;
      })
      .join("\n");
    return `\n\nGrammar & Language Issues Found (${allErrors.length} total):\n${errorSummary}${
      allErrors.length > 10 ? `\n... and ${allErrors.length - 10} more issues` : ""
    }`;
  } else if (languageToolErrors !== undefined || llmErrors !== undefined) {
    return "\n\nGrammar & Language: No errors detected by LanguageTool or AI assessment.";
  }
  return "";
}

export function buildRelevanceContext(relevanceCheck?: RelevanceCheck): string {
  if (!relevanceCheck) return "";
  const thresholdText =
    relevanceCheck.threshold !== undefined
      ? `, threshold: ${relevanceCheck.threshold.toFixed(2)}`
      : "";
  return `\n\nAnswer Relevance: ${
    relevanceCheck.addressesQuestion
      ? "The answer addresses the question"
      : "The answer may not fully address the question"
  } (similarity score: ${relevanceCheck.score.toFixed(2)}${thresholdText}).`;
}

export function buildTeacherScoreContext(essayScores?: EssayScores): string {
  const normalized = normalizeEssayScores(essayScores);
  if (!normalized) return "";
  const overall = normalized.overall ?? 0;
  const lowestDim = Object.entries(normalized.dimensions || {}).sort(
    ([, a], [, b]) => (a ?? 0) - (b ?? 0),
  )[0];
  return `\n\nStudent's performance:\n- Overall score: ${overall.toFixed(1)} / 9.0${
    lowestDim ? `\n- Weakest area: ${lowestDim[0]} (${lowestDim[1]?.toFixed(1)} / 9.0)` : ""
  }`;
}

export function buildTeacherErrorContext(
  languageToolErrors?: Array<Pick<FeedbackError, "errorType" | "category">>,
  llmErrors?: Array<Pick<FeedbackError, "errorType" | "category">>,
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
    return `\n\nFound ${allErrors.length} grammar/language issue${allErrors.length > 1 ? "s" : ""}${
      topErrorTypes ? `, mostly: ${topErrorTypes}` : ""
    }.`;
  }
  return "";
}

export function buildTeacherRelevanceContext(
  relevanceCheck?: Pick<RelevanceCheck, "addressesQuestion" | "score">,
): string {
  if (!relevanceCheck) return "";
  const relevancePercent = (relevanceCheck.score * 100).toFixed(0);
  if (relevanceCheck.addressesQuestion) {
    return `\n\nThe answer addresses the question well (relevance: ${relevancePercent}%).`;
  }
  return `\n\nThe answer may not fully address the question (relevance: ${relevancePercent}% - consider if all parts of the question were answered).`;
}

export function buildWordCountContext(answerText: string): string {
  const wordCount = countWords(answerText);
  return wordCount > 0 ? `\n\nEssay length: ${wordCount} word${wordCount !== 1 ? "s" : ""}.` : "";
}

export function getLowestDimension(essayScores?: EssayScores): [string, number] | undefined {
  const normalized = normalizeEssayScores(essayScores);
  if (!normalized?.dimensions) return undefined;
  return Object.entries(normalized.dimensions).sort(([, a], [, b]) => (a ?? 0) - (b ?? 0))[0] as
    | [string, number]
    | undefined;
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

type LegacyDimensions = EssayScores["dimensions"] & {
  Overall?: number;
};

export function normalizeEssayScores(essayScores?: EssayScores): EssayScores | undefined {
  if (!essayScores) return undefined;
  const dimensions = essayScores.dimensions as LegacyDimensions | undefined;
  const normalizedDimensions = dimensions
    ? {
        TA: dimensions.TA,
        CC: dimensions.CC,
        Vocab: dimensions.Vocab,
        Grammar: dimensions.Grammar,
      }
    : undefined;

  const overall = essayScores.overall ?? dimensions?.Overall;

  return {
    ...essayScores,
    overall,
    dimensions: normalizedDimensions,
  };
}
