import type { LanguageToolError } from "@writeo/shared";
import { MAX_ESSAY_LENGTH, MAX_QUESTION_LENGTH } from "./constants";

/**
 * Truncates essay text to the maximum allowed length for API processing
 */
export function truncateEssayText(text: string): string {
  return text.length > MAX_ESSAY_LENGTH
    ? text.slice(0, MAX_ESSAY_LENGTH) +
        "\n\n[... essay continues but truncated for feedback generation ...]"
    : text;
}

/**
 * Truncates question text to the maximum allowed length
 */
export function truncateQuestionText(text: string): string {
  return text.length > MAX_QUESTION_LENGTH ? text.slice(0, MAX_QUESTION_LENGTH) + "..." : text;
}

export function generateStructuredFeedback(
  match: any,
  errorText: string
): {
  errorType: string;
  explanation: string;
  example: string;
} {
  const category = (
    match.rule?.category?.id ||
    match.rule?.category?.name ||
    "UNKNOWN"
  ).toUpperCase();

  const ruleId = match.rule?.id || "";
  const message = match.message || match.shortMessage || "";
  const suggestions = match.replacements?.slice(0, 3).map((r: any) => r.value) || [];
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

export function isTenseConsistencyError(match: any, fullText: string): boolean {
  const ruleId = (match.rule?.id || "").toUpperCase();
  const message = (match.message || "").toLowerCase();
  const category = (
    match.rule?.category?.id ||
    match.rule?.category?.name ||
    "UNKNOWN"
  ).toUpperCase();

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

  const presentTenseVerbs = [
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
  ];
  const pastTenseVerbs = [
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
  ];

  const errorTextLower = errorText.trim();
  if (
    presentTenseVerbs.some(
      (verb) =>
        errorTextLower === verb ||
        errorTextLower.startsWith(verb + " ") ||
        errorTextLower.endsWith(" " + verb)
    )
  ) {
    const contextStart = Math.max(0, (match.offset || 0) - 100);
    const contextEnd = Math.min(fullText.length, (match.offset || 0) + (match.length || 0) + 100);
    const context = fullText.substring(contextStart, contextEnd).toLowerCase();

    const pastTenseIndicators = [
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
    ];

    const hasPastTenseNearby = pastTenseVerbs.some((verb) => context.includes(verb));
    const hasPastIndicator = pastTenseIndicators.some((indicator) => context.includes(indicator));

    if (hasPastIndicator || hasPastTenseNearby) {
      return true;
    }
  }

  return false;
}

export function calculateErrorConfidence(match: any, fullText?: string): number {
  let confidence = 0.5;

  const category = (
    match.rule?.category?.id ||
    match.rule?.category?.name ||
    "UNKNOWN"
  ).toUpperCase();

  if (category === "GRAMMAR" || category === "SPELLING") {
    confidence += 0.3;
  } else if (category === "TYPOGRAPHY" || category === "TYPO") {
    confidence += 0.2;
  } else if (category === "STYLE" || category === "PUNCTUATION") {
    confidence += 0.1;
  }

  if (fullText && isTenseConsistencyError(match, fullText)) {
    confidence += 0.25;

    const ruleId = (match.rule?.id || "").toUpperCase();
    const message = (match.message || "").toLowerCase();
    if (ruleId.includes("TENSE") || ruleId.includes("PAST") || message.includes("tense")) {
      confidence += 0.15;
    }

    const contextStart = Math.max(0, (match.offset || 0) - 100);
    const contextEnd = Math.min(fullText.length, (match.offset || 0) + (match.length || 0) + 100);
    const context = fullText.substring(contextStart, contextEnd).toLowerCase();
    const strongPastIndicators = ["yesterday", "last week", "last month", "ago", "was", "were"];
    if (strongPastIndicators.some((indicator) => context.includes(indicator))) {
      confidence += 0.1;
    }
  }

  const ruleId = (match.rule?.id || "").toUpperCase();
  const message = (match.message || "").toLowerCase();

  if (
    ruleId.includes("SUBJECT_VERB") ||
    ruleId.includes("AGREEMENT") ||
    (message.includes("subject") && message.includes("verb"))
  ) {
    confidence += 0.1;
  }

  if (match.issueType === "error") {
    confidence += 0.15;
  } else {
    confidence -= 0.1;
  }

  const suggestionCount = match.replacements?.length || 0;
  if (suggestionCount === 0) {
    confidence -= 0.2;
  } else if (suggestionCount === 1) {
    confidence += 0.1;
  } else if (suggestionCount > 3) {
    confidence -= 0.1;
  }

  const ruleType = match.rule?.type || "";
  if (ruleType === "spelling" || ruleType === "grammar") {
    confidence += 0.05;
  }

  return Math.max(0, Math.min(1, confidence));
}

/**
 * Finds the start of a word boundary at or before the given position
 */
function findWordStart(text: string, position: number): number {
  if (position <= 0) return 0;
  if (position >= text.length) return text.length;

  let pos = position;

  // If we're already at a word character, move to the start of that word
  if (/\w/.test(text[pos])) {
    // Move backwards to find the start of the word
    while (pos > 0 && /\w/.test(text[pos - 1])) {
      pos--;
    }
    return pos;
  }

  // If we're at whitespace or punctuation, look for the nearest word
  // First, try moving backwards to find a word
  let backPos = pos;
  while (backPos > 0 && !/\w/.test(text[backPos - 1])) {
    backPos--;
  }
  if (backPos > 0 && /\w/.test(text[backPos - 1])) {
    // Found a word character, move to its start
    while (backPos > 0 && /\w/.test(text[backPos - 1])) {
      backPos--;
    }
    return backPos;
  }

  // If no word found backwards, try forwards
  let forwardPos = pos;
  while (forwardPos < text.length && !/\w/.test(text[forwardPos])) {
    forwardPos++;
  }
  return forwardPos < text.length ? forwardPos : pos;
}

/**
 * Finds the end of a word boundary at or after the given position
 */
function findWordEnd(text: string, position: number): number {
  if (position >= text.length) return text.length;

  let pos = position;

  // If we're already at a word character, move to the end of that word
  if (/\w/.test(text[pos])) {
    // Move forwards to find the end of the word
    while (pos < text.length && /\w/.test(text[pos])) {
      pos++;
    }
    return pos;
  }

  // If we're at whitespace or punctuation, look for the nearest word
  // First, try moving forwards to find a word
  let forwardPos = pos;
  while (forwardPos < text.length && !/\w/.test(text[forwardPos])) {
    forwardPos++;
  }
  if (forwardPos < text.length && /\w/.test(text[forwardPos])) {
    // Found a word character, move to its end
    while (forwardPos < text.length && /\w/.test(text[forwardPos])) {
      forwardPos++;
    }
    return forwardPos;
  }

  // If no word found forwards, try backwards
  let backPos = pos;
  while (backPos > 0 && !/\w/.test(text[backPos - 1])) {
    backPos--;
  }
  if (backPos > 0 && /\w/.test(text[backPos - 1])) {
    // Found a word character, move to its end
    while (backPos < text.length && /\w/.test(text[backPos])) {
      backPos++;
    }
    return backPos;
  }

  return pos;
}

/**
 * Finds a text snippet in the full text using fuzzy matching
 * Returns the position if found, or null if not found
 */
function findTextSnippet(
  snippet: string,
  fullText: string,
  expectedPosition: number,
  maxDistance: number = 100
): { start: number; end: number } | null {
  if (!snippet || snippet.trim().length === 0) return null;

  const normalizedSnippet = snippet.trim().toLowerCase();
  const searchStart = Math.max(0, expectedPosition - maxDistance);
  const searchEnd = Math.min(fullText.length, expectedPosition + maxDistance + snippet.length);
  const searchArea = fullText.substring(searchStart, searchEnd);

  // Try exact match first
  const exactIndex = searchArea.toLowerCase().indexOf(normalizedSnippet);
  if (exactIndex !== -1) {
    return {
      start: searchStart + exactIndex,
      end: searchStart + exactIndex + normalizedSnippet.length,
    };
  }

  // Try fuzzy match - find words that match
  const snippetWords = normalizedSnippet.split(/\s+/).filter((w) => w.length > 0);
  if (snippetWords.length === 0) return null;

  // Find the first word
  const firstWord = snippetWords[0];
  const firstWordIndex = searchArea.toLowerCase().indexOf(firstWord);
  if (firstWordIndex === -1) return null;

  const foundStart = searchStart + firstWordIndex;
  const foundEnd = findWordEnd(fullText, foundStart + firstWord.length);

  // Check if the found text is reasonable
  const foundText = fullText.substring(foundStart, foundEnd).toLowerCase();
  if (foundText.includes(firstWord)) {
    // Extend to include all words if possible
    let currentEnd = foundEnd;
    for (let i = 1; i < snippetWords.length; i++) {
      const nextWord = snippetWords[i];
      const nextIndex = fullText
        .substring(currentEnd, currentEnd + 50)
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
 * Finds text in fullText using the error text and context words (before/after)
 * This is more reliable than trusting LLM-calculated positions
 */
export function findTextWithContext(
  errorText: string,
  wordBefore: string | null,
  wordAfter: string | null,
  fullText: string
): { start: number; end: number } | null {
  if (!errorText || errorText.trim().length === 0) return null;

  const normalizedErrorText = errorText.trim();
  const normalizedFullText = fullText.toLowerCase();
  const normalizedErrorLower = normalizedErrorText.toLowerCase();

  // First, try to find error text with context (wordBefore and wordAfter)
  if (wordBefore || wordAfter) {
    // Build a search pattern: [wordBefore] errorText [wordAfter]
    const beforeLower = wordBefore ? wordBefore.trim().toLowerCase() : null;
    const afterLower = wordAfter ? wordAfter.trim().toLowerCase() : null;

    // Search for error text, then verify context
    let searchStart = 0;
    while (true) {
      const foundIndex = normalizedFullText.indexOf(normalizedErrorLower, searchStart);
      if (foundIndex === -1) break;

      // Check wordBefore context
      let beforeMatches = true;
      if (beforeLower) {
        const beforeArea = normalizedFullText.substring(Math.max(0, foundIndex - 50), foundIndex);
        beforeMatches = beforeArea.includes(beforeLower);
      }

      // Check wordAfter context
      let afterMatches = true;
      if (afterLower && beforeMatches) {
        const afterStart = foundIndex + normalizedErrorLower.length;
        const afterArea = normalizedFullText.substring(afterStart, afterStart + 50);
        afterMatches = afterArea.includes(afterLower);
      }

      if (beforeMatches && afterMatches) {
        // Found with matching context!
        return {
          start: foundIndex,
          end: foundIndex + normalizedErrorText.length,
        };
      }

      // Continue searching from after this match
      searchStart = foundIndex + 1;
    }
  }

  // Fallback: find error text without context (or if context matching failed)
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
  fullText: string
): { start: number; end: number; valid: boolean } {
  // Basic bounds checking
  if (error.start < 0 || error.end > fullText.length || error.start >= error.end) {
    return { start: 0, end: 0, valid: false };
  }

  let start = error.start;
  let end = error.end;

  // If errorText is provided, try to find it in the text
  if (error.errorText && error.errorText.trim().length > 0) {
    const found = findTextSnippet(error.errorText, fullText, start, 200);
    if (found) {
      start = found.start;
      end = found.end;
    }
  }

  // Get the text at the current position
  const currentText = fullText.substring(start, end);

  // Check if the current position splits a word
  // A word is split if:
  // - The character before start is a word char AND start is a word char (we're in middle of word at start)
  // - The character at end-1 is a word char AND end is a word char (we're in middle of word at end)
  const splitsWordAtStart =
    start > 0 && /\w/.test(fullText[start - 1]) && /\w/.test(fullText[start]);
  const splitsWordAtEnd =
    end < fullText.length && /\w/.test(fullText[end - 1]) && /\w/.test(fullText[end]);
  const splitsWord = splitsWordAtStart || splitsWordAtEnd;

  // If the position splits a word or is very close to word boundaries, align to word boundaries
  // This handles cases where positions are slightly off or split words
  const wordStart = findWordStart(fullText, start);
  const wordEnd = findWordEnd(fullText, end);

  // Align to word boundaries if:
  // 1. The position splits a word, OR
  // 2. We're very close to word boundaries (within 5 characters), OR
  // 3. The current text doesn't contain a complete word
  const distanceToWordStart = Math.abs(start - wordStart);
  const distanceToWordEnd = Math.abs(end - wordEnd);
  const hasCompleteWord = /\w+/.test(currentText.trim());

  if (splitsWord || (distanceToWordStart <= 5 && distanceToWordEnd <= 5) || !hasCompleteWord) {
    // Only align if it makes sense (don't expand too far)
    const maxExpansion = 20; // Don't expand more than 20 characters
    if (wordStart <= start + maxExpansion && wordEnd >= end - maxExpansion) {
      start = wordStart;
      end = wordEnd;
    }
  }

  // Ensure we don't have an empty range
  if (start >= end) {
    // Try to find a reasonable word boundary
    const newStart = findWordStart(fullText, error.start);
    const newEnd = findWordEnd(fullText, error.start + 1);
    if (newStart < newEnd) {
      start = newStart;
      end = newEnd;
    } else {
      return { start: 0, end: 0, valid: false };
    }
  }

  // Final validation: ensure the position makes sense
  // Check that we're not highlighting only whitespace
  const finalText = fullText.substring(start, end).trim();
  if (finalText.length === 0) {
    return { start: 0, end: 0, valid: false };
  }

  // Check that we're highlighting at least part of a word
  const hasWordChar = /\w/.test(finalText);
  if (!hasWordChar && error.errorType !== "Punctuation") {
    return { start: 0, end: 0, valid: false };
  }

  return { start, end, valid: true };
}

export function transformLanguageToolResponse(
  ltResponse: any,
  fullText?: string
): LanguageToolError[] {
  if (!ltResponse?.matches || !Array.isArray(ltResponse.matches)) {
    return [];
  }

  const HIGH_CONFIDENCE_THRESHOLD = 0.8;
  const MEDIUM_CONFIDENCE_THRESHOLD = 0.6;
  const TENSE_ERROR_THRESHOLD = 0.7;

  return ltResponse.matches
    .map((match: any) => {
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
            (match.context.offset || 0) + (match.context.length || 0)
          ) || "";

        const validated = validateAndCorrectErrorPosition(
          {
            start,
            end,
            errorText: errorText || fullText.substring(start, end),
            errorType: match.rule?.category?.id || match.rule?.category?.name,
          },
          fullText
        );

        if (!validated.valid) {
          // Skip invalid positions
          return null;
        }

        validatedStart = validated.start;
        validatedEnd = validated.end;
      }

      const confidenceScore = calculateErrorConfidence(match, fullText);

      const isTenseError = fullText ? isTenseConsistencyError(match, fullText) : false;
      const effectiveHighThreshold = isTenseError
        ? TENSE_ERROR_THRESHOLD
        : HIGH_CONFIDENCE_THRESHOLD;

      const highConfidence = confidenceScore >= effectiveHighThreshold;
      const mediumConfidence =
        confidenceScore >= MEDIUM_CONFIDENCE_THRESHOLD && confidenceScore < effectiveHighThreshold;

      const errorText =
        match.context?.text?.substring(
          match.context.offset || 0,
          (match.context.offset || 0) + (match.context.length || 0)
        ) || "";

      const structuredFeedback = generateStructuredFeedback(match, errorText);

      return {
        start: validatedStart,
        end: validatedEnd,
        length: validatedEnd - validatedStart,
        sentenceIndex: undefined,
        category: (
          match.rule?.category?.id ||
          match.rule?.category?.name ||
          "UNKNOWN"
        ).toUpperCase(),
        rule_id: match.rule?.id || "UNKNOWN",
        message: match.message || match.shortMessage || match.rule?.description || "Error detected",
        suggestions: match.replacements?.slice(0, 5).map((r: any) => r.value) || [],
        source: "LT" as const,
        severity: (match.issueType === "error" ? "error" : "warning") as "error" | "warning",
        confidenceScore,
        highConfidence,
        mediumConfidence,
        errorType: structuredFeedback.errorType,
        explanation: structuredFeedback.explanation,
        example: structuredFeedback.example,
      };
    })
    .filter((err: any): err is LanguageToolError => err !== null);
}
