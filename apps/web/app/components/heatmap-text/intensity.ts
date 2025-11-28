import type { LanguageToolError } from "@writeo/shared";
import { validateError } from "./utils";

const CONTEXT_SIZE = 50;
const ERROR_INTENSITY = 1.0;
const WARNING_INTENSITY = 0.5;
const IN_ERROR_MULTIPLIER = 0.4;
const CONTEXT_MULTIPLIER = 0.3;

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
  const intensityMap = new Array(text.length).fill(0);

  for (const error of errors) {
    if (!validateError(error, text.length)) {
      continue;
    }

    const intensity = calculateBaseIntensity(error);
    const contextStart = Math.max(0, error.start - CONTEXT_SIZE);
    const contextEnd = Math.min(text.length, error.end + CONTEXT_SIZE);

    for (let i = contextStart; i < contextEnd; i++) {
      const localIntensity = calculateLocalIntensity(i, error.start, error.end, intensity);
      intensityMap[i] = Math.max(intensityMap[i], localIntensity);
    }
  }

  return intensityMap;
}

export function normalizeIntensity(intensityMap: number[]): number[] {
  if (intensityMap.length === 0) {
    return [];
  }

  const maxIntensity = Math.max(...intensityMap);
  if (maxIntensity === 0) {
    return intensityMap;
  }

  return intensityMap.map((i) => i / maxIntensity);
}
