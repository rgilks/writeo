/**
 * Intensity calculation utilities for heat map
 */

import type { LanguageToolError } from "@writeo/shared";

const CONTEXT_SIZE = 50;

function calculateBaseIntensity(error: LanguageToolError): number {
  const baseIntensity = error.severity === "error" ? 1.0 : 0.5;
  const categoryMultiplier =
    error.category === "GRAMMAR" || error.category === "SPELLING"
      ? 1.5
      : error.category === "TYPOS"
        ? 1.2
        : 1.0;
  return baseIntensity * categoryMultiplier;
}

function calculateLocalIntensity(
  position: number,
  errorStart: number,
  errorEnd: number,
  intensity: number
): number {
  const distFromStart = errorStart - position;
  const distFromEnd = position - errorEnd;

  if (position >= errorStart && position < errorEnd) {
    return intensity * 0.4;
  } else if (position < errorStart) {
    const fadeRatio = Math.max(0, (CONTEXT_SIZE - distFromStart) / CONTEXT_SIZE);
    return intensity * fadeRatio * 0.3;
  } else if (position >= errorEnd) {
    const fadeRatio = Math.max(0, (CONTEXT_SIZE - distFromEnd) / CONTEXT_SIZE);
    return intensity * fadeRatio * 0.3;
  }

  return 0;
}

export function calculateIntensityMap(text: string, errors: LanguageToolError[]): number[] {
  const intensityMap = new Array(text.length).fill(0);

  errors.forEach((error) => {
    if (!error || error.start < 0 || error.end > text.length || error.start >= error.end) {
      return;
    }

    const intensity = calculateBaseIntensity(error);
    const contextStart = Math.max(0, error.start - CONTEXT_SIZE);
    const contextEnd = Math.min(text.length, error.end + CONTEXT_SIZE);

    for (let i = contextStart; i < contextEnd && i < text.length; i++) {
      const localIntensity = calculateLocalIntensity(i, error.start, error.end, intensity);
      intensityMap[i] = Math.max(intensityMap[i], localIntensity);
    }
  });

  return intensityMap;
}

export function normalizeIntensity(intensityMap: number[]): number[] {
  const maxIntensity = Math.max(...intensityMap, 1);
  return intensityMap.map((i) => i / maxIntensity);
}
