import type {
  LanguageToolError,
  LanguageToolMatch,
  LanguageToolResponse,
  LanguageToolReplacement,
} from "@writeo/shared";
import { MAX_ESSAY_LENGTH, MAX_QUESTION_LENGTH } from "./constants";

// Confidence calculation constants
const BASE_CONFIDENCE = 0.5;
const HIGH_CONFIDENCE_THRESHOLD = 0.8;
const MEDIUM_CONFIDENCE_THRESHOLD = 0.6;
const TENSE_ERROR_THRESHOLD = 0.7;

// Confidence adjustments
const CATEGORY_BONUS = {
  GRAMMAR_OR_SPELLING: 0.3,
  TYPOGRAPHY: 0.2,
  STYLE_OR_PUNCTUATION: 0.1,
};
const TENSE_CONSISTENCY_BONUS = 0.25;
const TENSE_EXPLICIT_BONUS = 0.15;
const STRONG_PAST_INDICATOR_BONUS = 0.1;
const SUBJECT_VERB_BONUS = 0.1;
const ERROR_TYPE_BONUS = 0.15;
const WARNING_TYPE_PENALTY = 0.1;
const NO_SUGGESTIONS_PENALTY = 0.2;
const SINGLE_SUGGESTION_BONUS = 0.1;
const MULTIPLE_SUGGESTIONS_PENALTY = 0.1;
const RULE_TYPE_BONUS = 0.05;

// Position validation constants
const CONTEXT_WINDOW_SIZE = 100;
const MAX_SEARCH_DISTANCE = 200;
const MAX_WORD_BOUNDARY_DISTANCE = 5;
const MAX_EXPANSION_DISTANCE = 20;
const FUZZY_SEARCH_WINDOW = 50;

// Tense detection arrays
const PRESENT_TENSE_VERBS = [
  "go",
  "come",
  "see",
  "do",
  "have",
  "eat",
  "take",
  "make",
  "get",
  "give",
  "say",
  "know",
  "visit",
  "enjoy",
  "play",
  "talk",
] as const;

const PAST_TENSE_VERBS = [
  "went",
  "came",
  "saw",
  "did",
  "had",
  "ate",
  "took",
  "made",
  "got",
  "gave",
  "said",
  "knew",
  "visited",
  "enjoyed",
  "played",
  "talked",
] as const;

const PAST_TENSE_INDICATORS = [
  "yesterday",
  "last week",
  "last month",
  "last year",
  "last weekend",
  "ago",
  "was",
  "were",
  "went",
  "came",
  "saw",
  "did",
  "had",
  "ate",
  "took",
  "made",
  "got",
  "gave",
  "said",
  "knew",
  "visited",
  "enjoyed",
  "played",
  "talked",
  "before",
  "earlier",
  "previously",
  "then",
] as const;

const STRONG_PAST_INDICATORS = [
  "yesterday",
  "last week",
  "last month",
  "ago",
  "was",
  "were",
] as const;

const ESSAY_TRUNCATION_NOTICE =
  "\n\n[... essay continues but truncated for feedback generation ...]";

/**
 * Extracts and normalizes the category from a LanguageTool match.
 */
function getCategory(match: LanguageToolMatch): string {
  return (match.rule?.category?.id || match.rule?.category?.name || "UNKNOWN").toUpperCase();
}

/**
 * Truncates essay text to the maximum allowed length for API processing.
 *
 * If the text exceeds the limit, it's truncated and a continuation notice is appended.
 *
 * @param text - The essay text to truncate
 * @returns Truncated text with continuation notice if needed
 */
export function truncateEssayText(text: string): string {
  return text.length > MAX_ESSAY_LENGTH
    ? text.slice(0, MAX_ESSAY_LENGTH) + ESSAY_TRUNCATION_NOTICE
    : text;
}

/**
 * Truncates question text to the maximum allowed length.
 *
 * @param text - The question text to truncate
 * @returns Truncated text with ellipsis if needed
 */
export function truncateQuestionText(text: string): string {
  return text.length > MAX_QUESTION_LENGTH ? text.slice(0, MAX_QUESTION_LENGTH) + "..." : text;
}

/**
 * Generates structured feedback (error type, explanation, example) from a LanguageTool match.
 */
export function generateStructuredFeedback(
  match: LanguageToolMatch,
  errorText: string,
): {
  errorType: string;
  explanation: string;
  example: string;
} {
  const category = getCategory(match);

  const ruleId = match.rule?.id || "";
  const message = match.message || match.shortMessage || "";
  const suggestions =
    match.replacements?.slice(0, 3).map((r: LanguageToolReplacement) => r.value) || [];
  const firstSuggestion = suggestions[0];

  let errorType = "Grammar error";
  let explanation = message || "There may be an error here.";
  let example = "";

  if (category === "GRAMMAR") {
    if (
      ruleId.includes("SUBJECT_VERB") ||
      message.toLowerCase().includes("subject") ||
      message.toLowerCase().includes("verb")
    ) {
      errorType = "Subject-verb agreement";
      explanation = "The verb doesn't match the subject in number (singular/plural).";
    } else if (
      ruleId.includes("ARTICLE") ||
      message.toLowerCase().includes("article") ||
      message.toLowerCase().includes("a/an/the")
    ) {
      errorType = "Article use";
      explanation = "The article (a, an, the) may not be correct here.";
    } else if (ruleId.includes("PREPOSITION") || message.toLowerCase().includes("preposition")) {
      errorType = "Preposition use";
      explanation = "The preposition may not be correct for this context.";
    } else if (ruleId.includes("TENSE") || message.toLowerCase().includes("tense")) {
      errorType = "Verb tense";
      explanation = "The verb tense may not match the time frame being described.";
    } else {
      errorType = "Grammar";
      explanation = message || "There is a grammar error here.";
    }
  } else if (category === "SPELLING") {
    errorType = "Spelling";
    explanation = "This word may be misspelled.";
  } else if (category === "TYPOGRAPHY" || category === "TYPO") {
    errorType = "Typo";
    explanation = "There may be a typographical error here.";
  } else if (category === "STYLE") {
    errorType = "Style";
    explanation = "This may not follow the preferred writing style.";
  } else if (category === "PUNCTUATION") {
    errorType = "Punctuation";
    explanation = "The punctuation may be incorrect.";
  }

  if (firstSuggestion && errorText) {
    example = `${errorText} → ${firstSuggestion}`;
  } else if (firstSuggestion) {
    example = `Try: "${firstSuggestion}"`;
  }

  return { errorType, explanation, example };
}

/**
 * Checks if a LanguageTool match represents a tense consistency error.
 * Detects mismatches between present tense verbs and past tense context indicators.
 */
export function isTenseConsistencyError(match: LanguageToolMatch, fullText: string): boolean {
  const ruleId = (match.rule?.id || "").toUpperCase();
  const message = (match.message || "").toLowerCase();
  const category = getCategory(match);

  if (
    ruleId.includes("TENSE") ||
    message.includes("tense") ||
    (category === "GRAMMAR" && message.includes("verb"))
  ) {
    return true;
  }

  const errorText = fullText
    .substring(match.offset || 0, (match.offset || 0) + (match.length || 0))
    .toLowerCase();

  const errorTextLower = errorText.trim();
  if (
    PRESENT_TENSE_VERBS.some(
      (verb) =>
        errorTextLower === verb ||
        errorTextLower.startsWith(verb + " ") ||
        errorTextLower.endsWith(" " + verb),
    )
  ) {
    const contextStart = Math.max(0, (match.offset || 0) - CONTEXT_WINDOW_SIZE);
    const contextEnd = Math.min(
      fullText.length,
      (match.offset || 0) + (match.length || 0) + CONTEXT_WINDOW_SIZE,
    );
    const context = fullText.substring(contextStart, contextEnd).toLowerCase();

    const hasPastTenseNearby = PAST_TENSE_VERBS.some((verb) => context.includes(verb));
    const hasPastIndicator = PAST_TENSE_INDICATORS.some((indicator) => context.includes(indicator));

    if (hasPastIndicator || hasPastTenseNearby) {
      return true;
    }
  }

  return false;
}

/**
 * Calculates a confidence score (0-1) for a LanguageTool match based on category,
 * context, and other factors. Higher scores indicate more reliable error detection.
 */
export function calculateErrorConfidence(match: LanguageToolMatch, fullText?: string): number {
  let confidence = BASE_CONFIDENCE;

  const category = getCategory(match);

  if (category === "GRAMMAR" || category === "SPELLING") {
    confidence += CATEGORY_BONUS.GRAMMAR_OR_SPELLING;
  } else if (category === "TYPOGRAPHY" || category === "TYPO") {
    confidence += CATEGORY_BONUS.TYPOGRAPHY;
  } else if (category === "STYLE" || category === "PUNCTUATION") {
    confidence += CATEGORY_BONUS.STYLE_OR_PUNCTUATION;
  }

  if (fullText && isTenseConsistencyError(match, fullText)) {
    confidence += TENSE_CONSISTENCY_BONUS;

    const ruleId = (match.rule?.id || "").toUpperCase();
    const message = (match.message || "").toLowerCase();
    if (ruleId.includes("TENSE") || ruleId.includes("PAST") || message.includes("tense")) {
      confidence += TENSE_EXPLICIT_BONUS;
    }

    const contextStart = Math.max(0, (match.offset || 0) - CONTEXT_WINDOW_SIZE);
    const contextEnd = Math.min(
      fullText.length,
      (match.offset || 0) + (match.length || 0) + CONTEXT_WINDOW_SIZE,
    );
    const context = fullText.substring(contextStart, contextEnd).toLowerCase();
    if (STRONG_PAST_INDICATORS.some((indicator) => context.includes(indicator))) {
      confidence += STRONG_PAST_INDICATOR_BONUS;
    }
  }

  const ruleId = (match.rule?.id || "").toUpperCase();
  const message = (match.message || "").toLowerCase();

  if (
    ruleId.includes("SUBJECT_VERB") ||
    ruleId.includes("AGREEMENT") ||
    (message.includes("subject") && message.includes("verb"))
  ) {
    confidence += SUBJECT_VERB_BONUS;
  }

  if (match.issueType === "error") {
    confidence += ERROR_TYPE_BONUS;
  } else {
    confidence += WARNING_TYPE_PENALTY;
  }

  const suggestionCount = match.replacements?.length || 0;
  if (suggestionCount === 0) {
    confidence += NO_SUGGESTIONS_PENALTY;
  } else if (suggestionCount === 1) {
    confidence += SINGLE_SUGGESTION_BONUS;
  } else if (suggestionCount > 3) {
    confidence += MULTIPLE_SUGGESTIONS_PENALTY;
  }

  const ruleType = match.rule?.type || "";
  if (ruleType === "spelling" || ruleType === "grammar") {
    confidence += RULE_TYPE_BONUS;
  }

  return Math.max(0, Math.min(1, confidence));
}

/**
 * Finds the start of a word boundary at or before the given position.
 */
function findWordStart(text: string, position: number): number {
  if (position <= 0) return 0;
  if (position >= text.length) return text.length;

  let pos = position;

  const charAtPos = text[pos];
  if (charAtPos !== undefined && /\w/.test(charAtPos)) {
    // Move backwards to word start
    while (pos > 0) {
      const prevChar = text[pos - 1];
      if (prevChar !== undefined && /\w/.test(prevChar)) {
        pos--;
      } else {
        break;
      }
    }
    return pos;
  }

  // Look for nearest word: try backwards first, then forwards
  let backPos = pos;
  while (backPos > 0) {
    const prevChar = text[backPos - 1];
    if (prevChar !== undefined && !/\w/.test(prevChar)) {
      backPos--;
    } else {
      break;
    }
  }
  if (backPos > 0) {
    const prevChar = text[backPos - 1];
    if (prevChar !== undefined && /\w/.test(prevChar)) {
      while (backPos > 0) {
        const charBefore = text[backPos - 1];
        if (charBefore !== undefined && /\w/.test(charBefore)) {
          backPos--;
        } else {
          break;
        }
      }
      return backPos;
    }
  }

  let forwardPos = pos;
  while (forwardPos < text.length) {
    const charAtForward = text[forwardPos];
    if (charAtForward !== undefined && !/\w/.test(charAtForward)) {
      forwardPos++;
    } else {
      break;
    }
  }
  return forwardPos < text.length ? forwardPos : pos;
}

/**
 * Finds the end of a word boundary at or after the given position.
 */
function findWordEnd(text: string, position: number): number {
  if (position >= text.length) return text.length;

  let pos = position;

  const charAtPos = text[pos];
  if (charAtPos !== undefined && /\w/.test(charAtPos)) {
    // Move forwards to word end
    while (pos < text.length) {
      const char = text[pos];
      if (char !== undefined && /\w/.test(char)) {
        pos++;
      } else {
        break;
      }
    }
    return pos;
  }

  // Look for nearest word: try forwards first, then backwards
  let forwardPos = pos;
  while (forwardPos < text.length) {
    const char = text[forwardPos];
    if (char !== undefined && !/\w/.test(char)) {
      forwardPos++;
    } else {
      break;
    }
  }
  if (forwardPos < text.length) {
    const char = text[forwardPos];
    if (char !== undefined && /\w/.test(char)) {
      while (forwardPos < text.length) {
        const charAtForward = text[forwardPos];
        if (charAtForward !== undefined && /\w/.test(charAtForward)) {
          forwardPos++;
        } else {
          break;
        }
      }
      return forwardPos;
    }
  }

  let backPos = pos;
  while (backPos > 0) {
    const prevChar = text[backPos - 1];
    if (prevChar !== undefined && !/\w/.test(prevChar)) {
      backPos--;
    } else {
      break;
    }
  }
  if (backPos > 0) {
    const prevChar = text[backPos - 1];
    if (prevChar !== undefined && /\w/.test(prevChar)) {
      while (backPos < text.length) {
        const charAtBack = text[backPos];
        if (charAtBack !== undefined && /\w/.test(charAtBack)) {
          backPos++;
        } else {
          break;
        }
      }
      return backPos;
    }
  }

  return pos;
}

/**
 * Finds a text snippet in the full text using fuzzy matching
 * Returns the position if found, or null if not found
 */
/**
 * Finds a text snippet in the full text using fuzzy matching.
 * Returns the position if found, or null if not found.
 */
function findTextSnippet(
  snippet: string,
  fullText: string,
  expectedPosition: number,
  maxDistance: number = CONTEXT_WINDOW_SIZE,
): { start: number; end: number } | null {
  if (!snippet || snippet.trim().length === 0) return null;

  const normalizedSnippet = snippet.trim().toLowerCase();
  const searchStart = Math.max(0, expectedPosition - maxDistance);
  const searchEnd = Math.min(fullText.length, expectedPosition + maxDistance + snippet.length);
  const searchArea = fullText.substring(searchStart, searchEnd);

  const exactIndex = searchArea.toLowerCase().indexOf(normalizedSnippet);
  if (exactIndex !== -1) {
    return {
      start: searchStart + exactIndex,
      end: searchStart + exactIndex + normalizedSnippet.length,
    };
  }

  // Fuzzy match: find words that match
  const snippetWords = normalizedSnippet.split(/\s+/).filter((w) => w.length > 0);
  if (snippetWords.length === 0) return null;

  const firstWord = snippetWords[0];
  if (!firstWord) return null;
  const firstWordIndex = searchArea.toLowerCase().indexOf(firstWord);
  if (firstWordIndex === -1) return null;

  const foundStart = searchStart + firstWordIndex;
  const foundEnd = findWordEnd(fullText, foundStart + firstWord.length);

  const foundText = fullText.substring(foundStart, foundEnd).toLowerCase();
  if (foundText.includes(firstWord)) {
    let currentEnd = foundEnd;
    for (let i = 1; i < snippetWords.length; i++) {
      const nextWord = snippetWords[i];
      if (!nextWord) break;
      const nextIndex = fullText
        .substring(currentEnd, currentEnd + FUZZY_SEARCH_WINDOW)
        .toLowerCase()
        .indexOf(nextWord);
      if (nextIndex !== -1) {
        currentEnd = currentEnd + nextIndex + nextWord.length;
      } else {
        break;
      }
    }
    return { start: foundStart, end: currentEnd };
  }

  return null;
}

/**
 * Finds text in fullText using the error text and context words (before/after).
 * This is more reliable than trusting LLM-calculated positions.
 *
 * @param errorText - The error text to find
 * @param wordBefore - Optional word that should appear before the error
 * @param wordAfter - Optional word that should appear after the error
 * @param fullText - The full text to search in
 * @returns Position of the error text, or null if not found
 */
export function findTextWithContext(
  errorText: string,
  wordBefore: string | null,
  wordAfter: string | null,
  fullText: string,
): { start: number; end: number } | null {
  if (!errorText || errorText.trim().length === 0) return null;

  const normalizedErrorText = errorText.trim();
  const normalizedFullText = fullText.toLowerCase();
  const normalizedErrorLower = normalizedErrorText.toLowerCase();

  if (wordBefore || wordAfter) {
    const beforeLower = wordBefore ? wordBefore.trim().toLowerCase() : null;
    const afterLower = wordAfter ? wordAfter.trim().toLowerCase() : null;

    let searchStart = 0;
    while (true) {
      const foundIndex = normalizedFullText.indexOf(normalizedErrorLower, searchStart);
      if (foundIndex === -1) break;

      let beforeMatches = true;
      if (beforeLower) {
        const beforeArea = normalizedFullText.substring(
          Math.max(0, foundIndex - FUZZY_SEARCH_WINDOW),
          foundIndex,
        );
        beforeMatches = beforeArea.includes(beforeLower);
      }

      let afterMatches = true;
      if (afterLower && beforeMatches) {
        const afterStart = foundIndex + normalizedErrorLower.length;
        const afterArea = normalizedFullText.substring(
          afterStart,
          afterStart + FUZZY_SEARCH_WINDOW,
        );
        afterMatches = afterArea.includes(afterLower);
      }

      if (beforeMatches && afterMatches) {
        return {
          start: foundIndex,
          end: foundIndex + normalizedErrorText.length,
        };
      }

      searchStart = foundIndex + 1;
    }
  }
  const simpleIndex = normalizedFullText.indexOf(normalizedErrorLower);
  if (simpleIndex !== -1) {
    return {
      start: simpleIndex,
      end: simpleIndex + normalizedErrorText.length,
    };
  }

  return null;
}

/**
 * Validates and corrects error positions to align with word boundaries
 * and ensure they match the actual text
 */
export function validateAndCorrectErrorPosition(
  error: {
    start: number;
    end: number;
    errorText?: string; // Optional: the text snippet that should be at this position
    message?: string;
    errorType?: string;
  },
  fullText: string,
): { start: number; end: number; valid: boolean } {
  // Basic bounds checking
  if (error.start < 0 || error.end > fullText.length || error.start >= error.end) {
    return { start: 0, end: 0, valid: false };
  }

  let start = error.start;
  let end = error.end;

  // If errorText is provided, try to find it in the text
  if (error.errorText && error.errorText.trim().length > 0) {
    const found = findTextSnippet(error.errorText, fullText, start, MAX_SEARCH_DISTANCE);
    if (found) {
      start = found.start;
      end = found.end;
    }
  }

  const currentText = fullText.substring(start, end);

  // Check if position splits a word (word char before start AND at start, or before end AND at end)
  const charBeforeStart = start > 0 ? fullText[start - 1] : undefined;
  const charAtStart = fullText[start];
  const charBeforeEnd = end > 0 ? fullText[end - 1] : undefined;
  const charAtEnd = end < fullText.length ? fullText[end] : undefined;
  const splitsWordAtStart =
    charBeforeStart !== undefined &&
    charAtStart !== undefined &&
    /\w/.test(charBeforeStart) &&
    /\w/.test(charAtStart);
  const splitsWordAtEnd =
    charBeforeEnd !== undefined &&
    charAtEnd !== undefined &&
    /\w/.test(charBeforeEnd) &&
    /\w/.test(charAtEnd);
  const splitsWord = splitsWordAtStart || splitsWordAtEnd;

  const wordStart = findWordStart(fullText, start);
  const wordEnd = findWordEnd(fullText, end);

  // Align to word boundaries if splitting a word, close to boundaries, or no complete word
  const distanceToWordStart = Math.abs(start - wordStart);
  const distanceToWordEnd = Math.abs(end - wordEnd);
  const hasCompleteWord = /\w+/.test(currentText.trim());

  if (
    splitsWord ||
    (distanceToWordStart <= MAX_WORD_BOUNDARY_DISTANCE &&
      distanceToWordEnd <= MAX_WORD_BOUNDARY_DISTANCE) ||
    !hasCompleteWord
  ) {
    if (wordStart <= start + MAX_EXPANSION_DISTANCE && wordEnd >= end - MAX_EXPANSION_DISTANCE) {
      start = wordStart;
      end = wordEnd;
    }
  }

  if (start >= end) {
    const newStart = findWordStart(fullText, error.start);
    const newEnd = findWordEnd(fullText, error.start + 1);
    if (newStart < newEnd) {
      start = newStart;
      end = newEnd;
    } else {
      return { start: 0, end: 0, valid: false };
    }
  }

  const finalText = fullText.substring(start, end).trim();
  if (finalText.length === 0) {
    return { start: 0, end: 0, valid: false };
  }

  const hasWordChar = /\w/.test(finalText);
  if (!hasWordChar && error.errorType !== "Punctuation") {
    return { start: 0, end: 0, valid: false };
  }

  return { start, end, valid: true };
}

/**
 * Transforms a LanguageTool API response into our standardized error format.
 *
 * This function processes LanguageTool matches, calculates confidence scores,
 * validates error positions, and enriches errors with structured feedback.
 *
 * @param ltResponse - The LanguageTool API response
 * @param fullText - Optional full text for position validation and context-aware confidence calculation
 * @returns Array of standardized LanguageToolError objects
 */
export function transformLanguageToolResponse(
  ltResponse: LanguageToolResponse,
  fullText?: string,
): LanguageToolError[] {
  if (!ltResponse?.matches || !Array.isArray(ltResponse.matches)) {
    return [];
  }

  const errors: LanguageToolError[] = [];
  for (const match of ltResponse.matches) {
    const start = match.offset || 0;
    const length = match.length || 0;
    const end = start + length;

    // Validate and correct position if fullText is available
    let validatedStart = start;
    let validatedEnd = end;
    if (fullText) {
      const errorText =
        match.context?.text?.substring(
          match.context.offset || 0,
          (match.context.offset || 0) + (match.context.length || 0),
        ) || "";

      const validated = validateAndCorrectErrorPosition(
        {
          start,
          end,
          errorText: errorText || fullText.substring(start, end),
          errorType: getCategory(match),
        },
        fullText,
      );

      if (!validated.valid) {
        // Skip invalid positions
        continue;
      }

      validatedStart = validated.start;
      validatedEnd = validated.end;
    }

    const confidenceScore = calculateErrorConfidence(match, fullText);

    const isTenseError = fullText ? isTenseConsistencyError(match, fullText) : false;
    const effectiveHighThreshold = isTenseError ? TENSE_ERROR_THRESHOLD : HIGH_CONFIDENCE_THRESHOLD;

    const highConfidence = confidenceScore >= effectiveHighThreshold;
    const mediumConfidence =
      confidenceScore >= MEDIUM_CONFIDENCE_THRESHOLD && confidenceScore < effectiveHighThreshold;

    const errorText =
      match.context?.text?.substring(
        match.context.offset || 0,
        (match.context.offset || 0) + (match.context.length || 0),
      ) || "";

    const structuredFeedback = generateStructuredFeedback(match, errorText);

    const error: LanguageToolError = {
      start: validatedStart,
      end: validatedEnd,
      length: validatedEnd - validatedStart,
      sentenceIndex: undefined,
      category: getCategory(match),
      rule_id: match.rule?.id || "UNKNOWN",
      message: match.message || match.shortMessage || match.rule?.description || "Error detected",
      suggestions:
        match.replacements?.slice(0, 5).map((r: LanguageToolReplacement) => r.value) || [],
      source: "LT" as const,
      severity: (match.issueType === "error" ? "error" : "warning") as "error" | "warning",
      confidenceScore,
      highConfidence,
      mediumConfidence,
      errorType: structuredFeedback.errorType,
      explanation: structuredFeedback.explanation,
      example: structuredFeedback.example,
    };
    errors.push(error);
  }
  return errors;
}
