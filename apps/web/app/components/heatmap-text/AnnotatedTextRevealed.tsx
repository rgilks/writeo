/**
 * Annotated text revealed component - shows text with error annotations
 */

import { useState, useRef, useEffect } from "react";
import type { AnnotatedTextRevealedProps } from "./types";
import { ErrorDetail } from "./ErrorDetail";
import { ErrorSpan } from "./ErrorSpan";
import { validateError, deduplicateErrors, getErrorColor } from "./utils";

export function AnnotatedTextRevealed({
  text,
  errors,
  showMediumConfidence = false,
  showExperimental = false,
  mediumConfidenceErrors = [],
  experimentalErrors = [],
}: AnnotatedTextRevealedProps) {
  const [activeErrorKey, setActiveErrorKey] = useState<string | null>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const explanationRef = useRef<HTMLDivElement>(null);

  let allErrors = [...errors];
  if (showMediumConfidence) {
    allErrors = [...allErrors, ...mediumConfidenceErrors];
  }
  if (showExperimental) {
    allErrors = [...allErrors, ...experimentalErrors];
  }

  const validatedErrors = allErrors.filter((error) => validateError(error, text.length));
  const sorted = deduplicateErrors(validatedErrors);

  if (sorted.length === 0) {
    return (
      <div translate="no" lang="en">
        {text}
      </div>
    );
  }

  const elements: React.ReactNode[] = [];
  let lastIndex = 0;

  for (let i = 0; i < sorted.length; i++) {
    const error = sorted[i];
    if (!error || error.start < 0 || error.end > text.length || error.start >= error.end) {
      continue;
    }

    if (lastIndex < error.start) {
      elements.push(
        <span key={`text-${lastIndex}`} translate="no" lang="en">
          {text.slice(lastIndex, error.start)}
        </span>
      );
    }

    const errorText = text.slice(error.start, error.end);
    const isMediumConfidence = error.mediumConfidence === true;
    const isExperimental = error.highConfidence === false && error.mediumConfidence !== true;
    const severity = error.severity || "error";
    const errorColor = getErrorColor(isExperimental, isMediumConfidence, severity);
    const errorKey = `error-${error.start}-${error.end}-${i}`;
    const isActive = activeErrorKey === errorKey;

    elements.push(
      <ErrorSpan
        key={errorKey}
        errorKey={errorKey}
        errorText={errorText}
        errorColor={errorColor}
        isActive={isActive}
        onActivate={() => setActiveErrorKey(errorKey)}
        onDeactivate={() => setActiveErrorKey(null)}
        error={error}
      />
    );

    lastIndex = error.end;
  }

  if (lastIndex < text.length) {
    elements.push(
      <span key={`text-${lastIndex}`} translate="no" lang="en">
        {text.slice(lastIndex)}
      </span>
    );
  }

  const activeError = activeErrorKey
    ? sorted.find((_, i) => `error-${sorted[i].start}-${sorted[i].end}-${i}` === activeErrorKey)
    : null;

  // Scroll to show explanation at the top when an error is activated
  useEffect(() => {
    if (activeError && explanationRef.current) {
      // Scroll the explanation div to the top of the viewport
      explanationRef.current.scrollIntoView({
        behavior: "smooth",
        block: "start",
        inline: "nearest",
      });
    }
  }, [activeError]);

  return (
    <div ref={containerRef} translate="no" lang="en" style={{ position: "relative" }}>
      <div
        ref={explanationRef}
        style={{
          position: "sticky",
          top: 0,
          zIndex: 1000,
          backgroundColor: "var(--bg-primary)",
          borderBottom: "2px solid var(--border-color)",
          padding: "var(--spacing-md)",
          marginBottom: "var(--spacing-md)",
          borderRadius: "var(--border-radius)",
          boxShadow: activeError ? "0 4px 12px rgba(0,0,0,0.15)" : "0 2px 8px rgba(0,0,0,0.05)",
          minHeight: activeError ? "auto" : "60px",
          transition: "all 0.2s ease",
          backdropFilter: "blur(10px)",
          WebkitBackdropFilter: "blur(10px)",
        }}
        lang="en"
      >
        {activeError ? (
          <ErrorDetail
            error={activeError}
            errorText={text.slice(activeError.start, activeError.end)}
            onClose={() => setActiveErrorKey(null)}
          />
        ) : (
          <div
            style={{
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              color: "var(--text-secondary)",
              fontSize: "14px",
              padding: "var(--spacing-sm)",
            }}
            lang="en"
          >
            Click on highlighted text to see suggestions for improvement.
          </div>
        )}
      </div>

      <div
        className="prose max-w-none notranslate"
        translate="no"
        lang="en"
        style={{
          padding: "var(--spacing-lg)",
          backgroundColor: "var(--bg-secondary)",
          borderRadius: "var(--border-radius)",
          lineHeight: "1.5",
          fontSize: "16px",
          whiteSpace: "pre-wrap",
        }}
      >
        {elements}
      </div>
    </div>
  );
}
