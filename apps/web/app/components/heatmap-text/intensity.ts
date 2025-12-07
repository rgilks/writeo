import type { LanguageToolError } from "@writeo/shared";
import { validateError } from "./utils";

const CONTEXT_SIZE = 50;
const ERROR_INTENSITY = 1.0;
const WARNING_INTENSITY = 0.5;
const IN_ERROR_MULTIPLIER = 0.4;
const CONTEXT_MULTIPLIER = 0.3;

// Maximum possible intensity (GRAMMAR error with 1.5 multiplier * 0.4 in-error multiplier)
const MAX_INTENSITY = 1.5 * ERROR_INTENSITY * IN_ERROR_MULTIPLIER;

const CATEGORY_MULTIPLIERS: Record<string, number> = {
  GRAMMAR: 1.5,
  SPELLING: 1.5,
  TYPOS: 1.2,
};

function calculateBaseIntensity(error: LanguageToolError): number {
  const baseIntensity = error.severity === "error" ? ERROR_INTENSITY : WARNING_INTENSITY;
  const categoryMultiplier = CATEGORY_MULTIPLIERS[error.category] ?? 1.0;
  return baseIntensity * categoryMultiplier;
}

function calculateLocalIntensity(
  position: number,
  errorStart: number,
  errorEnd: number,
  intensity: number,
): number {
  if (position >= errorStart && position < errorEnd) {
    return intensity * IN_ERROR_MULTIPLIER;
  }

  const distance = position < errorStart ? errorStart - position : position - errorEnd;
  const fadeRatio = Math.max(0, (CONTEXT_SIZE - distance) / CONTEXT_SIZE);
  return intensity * fadeRatio * CONTEXT_MULTIPLIER;
}

export function calculateIntensityMap(text: string, errors: LanguageToolError[]): number[] {
  const textLength = text.length;
  if (textLength === 0 || errors.length === 0) {
    return new Array(textLength).fill(0);
  }

  const intensityMap = new Array(textLength).fill(0);

  // Pre-sort errors by start position for better cache locality
  const sortedErrors = [...errors].sort((a, b) => a.start - b.start);

  for (const error of sortedErrors) {
    if (!validateError(error, textLength)) {
      continue;
    }

    const intensity = calculateBaseIntensity(error);
    const contextStart = Math.max(0, error.start - CONTEXT_SIZE);
    const contextEnd = Math.min(textLength, error.end + CONTEXT_SIZE);

    for (let i = contextStart; i < contextEnd; i++) {
      // Early exit: if already at max intensity, skip calculation
      if (intensityMap[i] >= MAX_INTENSITY) {
        continue;
      }

      const localIntensity = calculateLocalIntensity(i, error.start, error.end, intensity);
      if (localIntensity > intensityMap[i]) {
        intensityMap[i] = localIntensity;
      }
    }
  }

  return intensityMap;
}

export function normalizeIntensity(intensityMap: number[]): number[] {
  const length = intensityMap.length;
  if (length === 0) {
    return [];
  }

  // Find max using loop for better performance on large arrays
  let maxIntensity = 0;
  for (let i = 0; i < length; i++) {
    if (intensityMap[i] > maxIntensity) {
      maxIntensity = intensityMap[i];
    }
  }

  if (maxIntensity === 0) {
    return intensityMap;
  }

  // Normalize in place for better memory efficiency
  const result = new Array(length);
  for (let i = 0; i < length; i++) {
    result[i] = intensityMap[i] / maxIntensity;
  }

  return result;
}
