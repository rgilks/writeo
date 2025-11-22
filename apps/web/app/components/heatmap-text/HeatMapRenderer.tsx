/**
 * Heat map renderer component - renders text with intensity-based highlighting
 */

interface HeatMapRendererProps {
  text: string;
  normalizedIntensity: number[];
  revealed: boolean;
}

export function HeatMapRenderer({ text, normalizedIntensity, revealed }: HeatMapRendererProps) {
  const elements: React.ReactNode[] = [];
  let currentSpan = "";
  let currentIntensity = -1;
  let currentStart = 0;

  for (let i = 0; i <= text.length; i++) {
    const intensity = i < text.length ? normalizedIntensity[i] : -1;

    if (intensity !== currentIntensity || i === text.length) {
      if (currentSpan) {
        const opacity = Math.max(0.1, currentIntensity);
        const redIntensity = Math.floor(currentIntensity * 255);

        elements.push(
          <span
            key={`span-${currentStart}`}
            translate="no"
            lang="en"
            style={{
              backgroundColor: revealed
                ? "transparent"
                : `rgba(${redIntensity}, 0, 0, ${opacity * 0.3})`,
              boxShadow: revealed
                ? "none"
                : currentIntensity > 0.3
                  ? `0 0 ${currentIntensity * 8}px rgba(${redIntensity}, 0, 0, ${opacity * 0.5})`
                  : "none",
              transition: "all 0.4s cubic-bezier(0.4, 0, 0.2, 1)",
              padding: currentIntensity > 0.5 ? "2px 1px" : "0",
              borderRadius: "2px",
            }}
          >
            {currentSpan}
          </span>
        );
      }

      currentSpan = i < text.length ? text[i] : "";
      currentIntensity = intensity;
      currentStart = i;
    } else {
      currentSpan += text[i];
    }
  }

  return (
    <div
      className="prose max-w-none notranslate"
      translate="no"
      lang="en"
      style={{
        padding: "var(--spacing-lg)",
        backgroundColor: "transparent",
        borderRadius: "var(--border-radius)",
        lineHeight: "1.5",
        fontSize: "16px",
        whiteSpace: "pre-wrap",
      }}
    >
      {elements}
    </div>
  );
}
