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
  const spans: React.ReactNode[] = [];
  let currentSpan = "";
  let currentIntensity = -1;
  let currentStart = 0;

  for (let i = 0; i <= text.length; i++) {
    const intensity = i < text.length ? normalizedIntensity[i] : -1;

    if (intensity !== currentIntensity || i === text.length) {
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

      if (i < text.length) {
        currentSpan = text[i];
        currentIntensity = intensity;
        currentStart = i;
      }
    } else {
      currentSpan += text[i];
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
