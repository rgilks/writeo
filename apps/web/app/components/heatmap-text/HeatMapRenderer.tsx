import { useMemo } from "react";
import type { HeatMapRendererProps } from "./types";

const CONTAINER_STYLES = {
  padding: "var(--spacing-lg)",
  backgroundColor: "transparent",
  borderRadius: "var(--border-radius)",
  lineHeight: "1.5",
  fontSize: "16px",
  whiteSpace: "pre-wrap" as const,
} as const;

const BASE_SPAN_STYLES = {
  transition: "all 0.4s cubic-bezier(0.4, 0, 0.2, 1)",
  borderRadius: "2px",
} as const;

// Threshold for merging spans with similar intensities
// Spans with intensity difference <= this value will be merged
const INTENSITY_MERGE_THRESHOLD = 0.05;

// Quantize intensity to reduce number of unique values
function quantizeIntensity(intensity: number): number {
  // Round to nearest 0.05 for consistent merging
  return Math.round(intensity * 20) / 20;
}

function getSpanStyle(intensity: number, revealed: boolean) {
  if (revealed || intensity <= 0) {
    return {
      ...BASE_SPAN_STYLES,
      backgroundColor: "transparent",
      boxShadow: "none",
      padding: "0",
    };
  }

  const opacity = Math.max(0.1, intensity);
  const redIntensity = Math.floor(intensity * 255);
  const backgroundColor = `rgba(${redIntensity}, 0, 0, ${opacity * 0.3})`;
  const boxShadow =
    intensity > 0.3
      ? `0 0 ${intensity * 8}px rgba(${redIntensity}, 0, 0, ${opacity * 0.5})`
      : "none";
  const padding = intensity > 0.5 ? "2px 1px" : "0";

  return {
    ...BASE_SPAN_STYLES,
    backgroundColor,
    boxShadow,
    padding,
  };
}

function buildSpans(text: string, normalizedIntensity: number[], revealed: boolean) {
  const textLength = text.length;
  if (textLength === 0) {
    return [];
  }

  const spans: React.ReactNode[] = [];
  let currentSpan = text[0];
  let currentIntensity = quantizeIntensity(normalizedIntensity[0] ?? 0);
  let currentStart = 0;

  for (let i = 1; i <= textLength; i++) {
    const rawIntensity = i < textLength ? (normalizedIntensity[i] ?? 0) : -1;
    const intensity = i < textLength ? quantizeIntensity(rawIntensity) : -1;

    // Merge spans with similar intensities or continue building current span
    const shouldMerge =
      i < textLength && Math.abs(intensity - currentIntensity) <= INTENSITY_MERGE_THRESHOLD;

    if (shouldMerge) {
      currentSpan += text[i];
    } else {
      // Push current span
      if (currentSpan) {
        spans.push(
          <span
            key={`span-${currentStart}`}
            translate="no"
            style={getSpanStyle(currentIntensity, revealed)}
          >
            {currentSpan}
          </span>,
        );
      }

      // Start new span
      if (i < textLength) {
        currentSpan = text[i];
        currentIntensity = intensity;
        currentStart = i;
      }
    }
  }

  return spans;
}

export function HeatMapRenderer({ text, normalizedIntensity, revealed }: HeatMapRendererProps) {
  const elements = useMemo(
    () => buildSpans(text, normalizedIntensity, revealed),
    [text, normalizedIntensity, revealed],
  );

  return (
    <div className="prose max-w-none notranslate" translate="no" style={CONTAINER_STYLES}>
      {elements}
    </div>
  );
}
