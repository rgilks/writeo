import { useState, useRef, useEffect, useMemo } from "react";
import type { AnnotatedTextRevealedProps } from "./types";
import { ErrorDetail } from "./ErrorDetail";
import { ErrorSpan } from "./ErrorSpan";
import { validateError, deduplicateErrors, getErrorColor } from "./utils";

const EXPLANATION_STYLES = {
  position: "sticky" as const,
  top: 0,
  zIndex: 1000,
  backgroundColor: "var(--bg-primary)",
  borderBottom: "2px solid var(--border-color)",
  padding: "var(--spacing-md)",
  marginBottom: "var(--spacing-md)",
  borderRadius: "var(--border-radius)",
  transition: "all 0.2s ease",
  backdropFilter: "blur(10px)",
  WebkitBackdropFilter: "blur(10px)",
};

const PLACEHOLDER_STYLES = {
  display: "flex",
  alignItems: "center",
  justifyContent: "center",
  color: "var(--text-secondary)",
  fontSize: "14px",
  padding: "var(--spacing-sm)",
};

const TEXT_CONTAINER_STYLES = {
  padding: "var(--spacing-lg)",
  backgroundColor: "var(--bg-secondary)",
  borderRadius: "var(--border-radius)",
  lineHeight: "1.5",
  fontSize: "16px",
  whiteSpace: "pre-wrap" as const,
};

export function AnnotatedTextRevealed({
  text,
  errors,
  showMediumConfidence = false,
  showExperimental = false,
  mediumConfidenceErrors = [],
  experimentalErrors = [],
}: AnnotatedTextRevealedProps) {
  const [activeErrorIndex, setActiveErrorIndex] = useState<number | null>(null);
  const explanationRef = useRef<HTMLDivElement>(null);

  const sortedErrors = useMemo(() => {
    const allErrors = [
      ...errors,
      ...(showMediumConfidence ? mediumConfidenceErrors : []),
      ...(showExperimental ? experimentalErrors : []),
    ];
    const validated = allErrors.filter((error) => validateError(error, text.length));
    return deduplicateErrors(validated);
  }, [
    errors,
    mediumConfidenceErrors,
    experimentalErrors,
    showMediumConfidence,
    showExperimental,
    text.length,
  ]);

  const elements = useMemo(() => {
    if (sortedErrors.length === 0) return null;

    const result: React.ReactNode[] = [];
    let lastIndex = 0;

    for (let i = 0; i < sortedErrors.length; i++) {
      const error = sortedErrors[i];

      if (lastIndex < error.start) {
        result.push(
          <span key={`text-${lastIndex}`} translate="no">
            {text.slice(lastIndex, error.start)}
          </span>,
        );
      }

      const errorText = text.slice(error.start, error.end);
      const isMediumConfidence = error.mediumConfidence === true;
      const isExperimental = error.highConfidence === false && error.mediumConfidence !== true;
      const severity = error.severity || "error";
      const errorColor = getErrorColor(isExperimental, isMediumConfidence, severity);
      const errorKey = `error-${error.start}-${error.end}-${i}`;
      const isActive = activeErrorIndex === i;

      result.push(
        <ErrorSpan
          key={errorKey}
          errorText={errorText}
          errorColor={errorColor}
          isActive={isActive}
          onActivate={() => setActiveErrorIndex(i)}
          onDeactivate={() => setActiveErrorIndex(null)}
        />,
      );

      lastIndex = error.end;
    }

    if (lastIndex < text.length) {
      result.push(
        <span key={`text-${lastIndex}`} translate="no">
          {text.slice(lastIndex)}
        </span>,
      );
    }

    return result;
  }, [sortedErrors, text, activeErrorIndex]);

  const activeError = activeErrorIndex !== null ? sortedErrors[activeErrorIndex] : null;

  useEffect(() => {
    if (activeError && explanationRef.current) {
      explanationRef.current.scrollIntoView({
        behavior: "smooth",
        block: "start",
        inline: "nearest",
      });
    }
  }, [activeError]);

  if (sortedErrors.length === 0) {
    return <div translate="no">{text}</div>;
  }

  return (
    <div translate="no" style={{ position: "relative" }}>
      <div
        ref={explanationRef}
        style={{
          ...EXPLANATION_STYLES,
          boxShadow: activeError ? "0 4px 12px rgba(0,0,0,0.15)" : "0 2px 8px rgba(0,0,0,0.05)",
          minHeight: activeError ? "auto" : "60px",
        }}
      >
        {activeError ? (
          <ErrorDetail
            error={activeError}
            errorText={text.slice(activeError.start, activeError.end)}
            onClose={() => setActiveErrorIndex(null)}
          />
        ) : (
          <div style={PLACEHOLDER_STYLES}>
            Click on highlighted text to see suggestions for improvement.
          </div>
        )}
      </div>

      <div className="prose max-w-none notranslate" translate="no" style={TEXT_CONTAINER_STYLES}>
        {elements}
      </div>
    </div>
  );
}
