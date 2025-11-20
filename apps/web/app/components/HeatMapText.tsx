"use client";

import { useState, useRef, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import type { LanguageToolError } from "@writeo/shared";

// Error detail component - simplified to show explanations directly
function ErrorDetail({
  error,
  errorText,
  onClose,
}: {
  error: LanguageToolError;
  errorText: string;
  onClose?: () => void;
}) {
  const hasStructuredFeedback = !!(error.errorType || error.explanation || error.example);

  if (!hasStructuredFeedback) {
    // Fallback to simple display if no structured feedback
    return (
      <div
        style={{
          marginTop: "var(--spacing-xs)",
          fontSize: "14px",
          color: "var(--text-primary)",
          lineHeight: "1.5",
          position: "relative",
        }}
        lang="en"
      >
        {onClose && (
          <button
            onClick={onClose}
            style={{
              position: "absolute",
              top: "-8px",
              right: "-8px",
              background: "var(--bg-primary)",
              border: "1px solid var(--border-color)",
              borderRadius: "50%",
              width: "24px",
              height: "24px",
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              cursor: "pointer",
              fontSize: "18px",
              lineHeight: "1",
              padding: 0,
              color: "var(--text-secondary)",
            }}
            aria-label="Close"
            lang="en"
          >
            ×
          </button>
        )}
        {error.message}
        {error.suggestions?.[0] && ` → "${error.suggestions[0]}"`}
      </div>
    );
  }

  const isMediumConfidence = error.mediumConfidence === true;
  const isExperimental = error.highConfidence === false && error.mediumConfidence !== true;
  const confidenceLabel = isExperimental
    ? "Experimental"
    : isMediumConfidence
      ? "Medium Confidence"
      : "High Confidence";
  const confidenceColor = isExperimental ? "#f59e0b" : isMediumConfidence ? "#f97316" : "#10b981";

  return (
    <div style={{ marginTop: "var(--spacing-sm)", position: "relative" }} lang="en">
      {onClose && (
        <button
          onClick={onClose}
          style={{
            position: "absolute",
            top: "-8px",
            right: "-8px",
            background: "var(--bg-primary)",
            border: "1px solid var(--border-color)",
            borderRadius: "50%",
            width: "24px",
            height: "24px",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            cursor: "pointer",
            fontSize: "18px",
            lineHeight: "1",
            padding: 0,
            color: "var(--text-secondary)",
            zIndex: 1,
            boxShadow: "0 2px 4px rgba(0,0,0,0.1)",
          }}
          aria-label="Close"
          lang="en"
        >
          ×
        </button>
      )}
      {/* Error type badge with subtle confidence indicator */}
      <div
        style={{
          display: "flex",
          alignItems: "center",
          gap: "var(--spacing-sm)",
          marginBottom: "var(--spacing-sm)",
          flexWrap: "wrap",
        }}
        lang="en"
      >
        <span
          style={{
            padding: "var(--spacing-xs) var(--spacing-sm)",
            backgroundColor: "var(--primary-color)",
            color: "white",
            borderRadius: "var(--spacing-xs)",
            fontSize: "14px",
            fontWeight: 600,
          }}
          lang="en"
        >
          {error.errorType || "Error"}
        </span>
        {/* Small confidence indicator - only show for non-high confidence */}
        {(isMediumConfidence || isExperimental) && (
          <span
            style={{
              fontSize: "10px",
              color: "var(--text-secondary)",
              fontWeight: 400,
            }}
            lang="en"
            title={
              isExperimental
                ? "This is an experimental suggestion - you may want to check with a teacher"
                : "This is a medium-confidence suggestion - likely correct but less certain"
            }
          >
            ({confidenceLabel})
          </span>
        )}
      </div>

      {/* Explanation + Example */}
      <div
        style={{ marginBottom: "var(--spacing-sm)", fontSize: "14px", lineHeight: "1.5" }}
        lang="en"
      >
        {isExperimental && (
          <p
            style={{
              margin: "0 0 var(--spacing-xs) 0",
              padding: "var(--spacing-xs)",
              backgroundColor: "rgba(245, 158, 11, 0.1)",
              borderRadius: "var(--spacing-xs)",
              fontSize: "13px",
              color: "#92400e",
              fontStyle: "italic",
            }}
            lang="en"
          >
            💡 This is an experimental suggestion - you may want to check with a teacher.
          </p>
        )}
        <p style={{ margin: "0 0 var(--spacing-xs) 0", color: "var(--text-primary)" }} lang="en">
          {error.explanation || error.message}
        </p>
        {error.example && (
          <p
            style={{
              margin: "0",
              color: "var(--text-primary)",
              fontFamily: "monospace",
              fontSize: "14px",
            }}
            lang="en"
          >
            Example: {error.example}
          </p>
        )}
      </div>
    </div>
  );
}

interface HeatMapTextProps {
  text: string;
  errors: LanguageToolError[];
  onReveal?: () => void;
  showMediumConfidence?: boolean; // Show medium-confidence errors (60-80%)
  showExperimental?: boolean; // Show low-confidence errors (<60%, experimental suggestions)
}

/**
 * HeatMapText component - shows error intensity as a heat map without revealing details
 * Red glow indicates areas with worse errors
 * Only shows high-confidence errors by default (precision over coverage)
 * Supports three confidence tiers: high (>80%), medium (60-80%), low (<60%)
 */
export function HeatMapText({
  text,
  errors,
  onReveal,
  showMediumConfidence = false,
  showExperimental = false,
}: HeatMapTextProps) {
  const [revealed, setRevealed] = useState(false);
  const [showMediumConfidenceErrors, setShowMediumConfidenceErrors] =
    useState(showMediumConfidence);
  const [showExperimentalSuggestions, setShowExperimentalSuggestions] = useState(showExperimental);

  // Filter errors by confidence tier
  // High confidence: highConfidence === true (>80%)
  // Medium confidence: mediumConfidence === true (60-80%)
  // Low confidence: highConfidence === false and mediumConfidence !== true (<60%)
  const highConfidenceErrors = errors.filter((e) => e.highConfidence === true);
  const mediumConfidenceErrors = errors.filter((e) => e.mediumConfidence === true);
  const lowConfidenceErrors = errors.filter(
    (e) => e.highConfidence === false && e.mediumConfidence !== true
  );

  // Build filtered errors list based on toggles
  let filteredErrors = [...highConfidenceErrors];
  if (showMediumConfidenceErrors) {
    filteredErrors = [...filteredErrors, ...mediumConfidenceErrors];
  }
  if (showExperimentalSuggestions) {
    filteredErrors = [...filteredErrors, ...lowConfidenceErrors];
  }

  if (!errors || errors.length === 0 || !text) {
    return (
      <div className="prose max-w-none" translate="no" lang="en">
        {text || ""}
      </div>
    );
  }

  if (filteredErrors.length === 0 && !showExperimentalSuggestions) {
    return (
      <div>
        <div
          lang="en"
          style={{
            marginBottom: "var(--spacing-md)",
            padding: "var(--spacing-md)",
            backgroundColor: "rgba(16, 185, 129, 0.1)",
            border: "1px solid rgba(16, 185, 129, 0.3)",
            borderRadius: "var(--border-radius)",
            fontSize: "14px",
            lineHeight: "1.5",
          }}
        >
          <p style={{ marginBottom: "var(--spacing-sm)", fontWeight: 600 }} lang="en">
            ✅ Great work! No issues found
          </p>
          {(mediumConfidenceErrors.length > 0 || lowConfidenceErrors.length > 0) && (
            <div
              style={{
                display: "flex",
                gap: "var(--spacing-sm)",
                flexWrap: "wrap",
                alignItems: "center",
                marginTop: "var(--spacing-sm)",
              }}
            >
              {mediumConfidenceErrors.length > 0 && !showMediumConfidenceErrors && (
                <button
                  onClick={() => setShowMediumConfidenceErrors(true)}
                  className="btn btn-secondary"
                  style={{ fontSize: "13px", padding: "var(--spacing-xs) var(--spacing-sm)" }}
                  lang="en"
                >
                  Show {mediumConfidenceErrors.length} more suggestion
                  {mediumConfidenceErrors.length !== 1 ? "s" : ""}
                </button>
              )}
              {lowConfidenceErrors.length > 0 && !showExperimentalSuggestions && (
                <button
                  onClick={() => setShowExperimentalSuggestions(true)}
                  className="btn btn-secondary"
                  style={{ fontSize: "13px", padding: "var(--spacing-xs) var(--spacing-sm)" }}
                  lang="en"
                >
                  Show {lowConfidenceErrors.length} experimental suggestion
                  {lowConfidenceErrors.length !== 1 ? "s" : ""}
                </button>
              )}
            </div>
          )}
        </div>
        <div className="prose max-w-none" translate="no" lang="en">
          {text || ""}
        </div>
      </div>
    );
  }

  // Calculate error intensity for each character position
  // More severe errors and more errors in an area = higher intensity
  const intensityMap = new Array(text.length).fill(0);

  filteredErrors.forEach((error) => {
    if (!error || error.start < 0 || error.end > text.length || error.start >= error.end) {
      return;
    }

    // Calculate intensity: errors are more intense, warnings less
    const baseIntensity = error.severity === "error" ? 1.0 : 0.5;
    // Grammar and spelling errors are more critical
    const categoryMultiplier =
      error.category === "GRAMMAR" || error.category === "SPELLING"
        ? 1.5
        : error.category === "TYPOS"
          ? 1.2
          : 1.0;

    const intensity = baseIntensity * categoryMultiplier;

    // Apply intensity to the error range, with falloff at edges
    // Make heatmap less obvious - highlight general areas, not exact errors
    // Use much larger context area and reduce exact error intensity
    const contextSize = 50; // Characters before and after error - larger area
    const contextStart = Math.max(0, error.start - contextSize);
    const contextEnd = Math.min(text.length, error.end + contextSize);

    for (let i = contextStart; i < contextEnd && i < text.length; i++) {
      let localIntensity = intensity;

      // Calculate distance from error boundaries
      const distFromStart = error.start - i;
      const distFromEnd = i - error.end;

      // Reduce exact error intensity to 40% - make it less obvious
      if (i >= error.start && i < error.end) {
        localIntensity = intensity * 0.4; // Only 40% intensity for exact error
      }
      // Fade out before error - context area gets 30% max intensity
      else if (i < error.start) {
        const fadeRatio = Math.max(0, (contextSize - distFromStart) / contextSize);
        localIntensity = intensity * fadeRatio * 0.3; // 30% max intensity in context
      }
      // Fade out after error - context area gets 30% max intensity
      else if (i >= error.end) {
        const fadeRatio = Math.max(0, (contextSize - distFromEnd) / contextSize);
        localIntensity = intensity * fadeRatio * 0.3; // 30% max intensity in context
      }

      intensityMap[i] = Math.max(intensityMap[i], localIntensity);
    }
  });

  // Normalize intensity to 0-1 range
  const maxIntensity = Math.max(...intensityMap, 1);
  const normalizedIntensity = intensityMap.map((i) => i / maxIntensity);

  // Render text with heat map
  const elements: React.ReactNode[] = [];
  let currentSpan = "";
  let currentIntensity = -1;
  let currentStart = 0;

  for (let i = 0; i <= text.length; i++) {
    const intensity = i < text.length ? normalizedIntensity[i] : -1;

    if (intensity !== currentIntensity || i === text.length) {
      // Finish current span
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

      // Start new span
      currentSpan = i < text.length ? text[i] : "";
      currentIntensity = intensity;
      currentStart = i;
    } else {
      currentSpan += text[i];
    }
  }

  return (
    <div>
      {!revealed && (
        <div
          lang="en"
          style={{
            marginBottom: "var(--spacing-md)",
            padding: "var(--spacing-md)",
            backgroundColor: "rgba(59, 130, 246, 0.1)",
            border: "1px solid rgba(59, 130, 246, 0.2)",
            borderRadius: "var(--border-radius)",
            fontSize: "14px",
            lineHeight: "1.5",
          }}
        >
          <p
            style={{
              marginBottom: "var(--spacing-sm)",
              fontWeight: 600,
              color: "var(--text-primary)",
            }}
          >
            💡 Find the Mistakes
          </p>
          <p
            style={{
              marginBottom: "var(--spacing-md)",
              color: "var(--text-secondary)",
              fontSize: "14px",
            }}
          >
            Look for the red highlights in your text. Can you spot what needs fixing? Click "Show
            Feedback" when you're ready to see suggestions.
          </p>
          <div
            style={{
              display: "flex",
              gap: "var(--spacing-sm)",
              flexWrap: "wrap",
              alignItems: "center",
            }}
          >
            <button
              onClick={() => {
                setRevealed(true);
                onReveal?.();
              }}
              className="btn btn-primary"
              style={{ fontSize: "14px", padding: "var(--spacing-sm) var(--spacing-md)" }}
              lang="en"
            >
              Show Feedback
            </button>
            {(mediumConfidenceErrors.length > 0 || lowConfidenceErrors.length > 0) && (
              <span style={{ fontSize: "13px", color: "var(--text-secondary)" }} lang="en">
                {mediumConfidenceErrors.length > 0 && !showMediumConfidenceErrors && (
                  <button
                    onClick={() => setShowMediumConfidenceErrors(true)}
                    className="btn btn-secondary"
                    style={{
                      fontSize: "13px",
                      padding: "var(--spacing-xs) var(--spacing-sm)",
                      marginLeft: "var(--spacing-xs)",
                    }}
                    lang="en"
                  >
                    +{mediumConfidenceErrors.length} more
                  </button>
                )}
                {lowConfidenceErrors.length > 0 && !showExperimentalSuggestions && (
                  <button
                    onClick={() => setShowExperimentalSuggestions(true)}
                    className="btn btn-secondary"
                    style={{
                      fontSize: "13px",
                      padding: "var(--spacing-xs) var(--spacing-sm)",
                      marginLeft: "var(--spacing-xs)",
                    }}
                    lang="en"
                  >
                    +{lowConfidenceErrors.length} experimental
                  </button>
                )}
              </span>
            )}
          </div>
        </div>
      )}

      {/* Controls when revealed */}
      {revealed && (
        <div
          lang="en"
          style={{
            marginBottom: "var(--spacing-md)",
            padding: "var(--spacing-sm) var(--spacing-md)",
            backgroundColor: "var(--bg-secondary)",
            border: "1px solid rgba(0, 0, 0, 0.1)",
            borderRadius: "var(--border-radius)",
            fontSize: "13px",
            display: "flex",
            justifyContent: "space-between",
            alignItems: "center",
            flexWrap: "wrap",
            gap: "var(--spacing-sm)",
          }}
        >
          <div
            style={{
              display: "flex",
              gap: "var(--spacing-sm)",
              flexWrap: "wrap",
              alignItems: "center",
            }}
          >
            {mediumConfidenceErrors.length > 0 && (
              <button
                onClick={() => setShowMediumConfidenceErrors(!showMediumConfidenceErrors)}
                className="btn btn-secondary"
                style={{ fontSize: "13px", padding: "var(--spacing-xs) var(--spacing-sm)" }}
                lang="en"
              >
                {showMediumConfidenceErrors ? "Hide" : "Show"} {mediumConfidenceErrors.length} more
                suggestion{mediumConfidenceErrors.length !== 1 ? "s" : ""}
              </button>
            )}
            {lowConfidenceErrors.length > 0 && (
              <button
                onClick={() => setShowExperimentalSuggestions(!showExperimentalSuggestions)}
                className="btn btn-secondary"
                style={{ fontSize: "13px", padding: "var(--spacing-xs) var(--spacing-sm)" }}
                lang="en"
              >
                {showExperimentalSuggestions ? "Hide" : "Show"} {lowConfidenceErrors.length}{" "}
                experimental suggestion{lowConfidenceErrors.length !== 1 ? "s" : ""}
              </button>
            )}
          </div>
          <button
            onClick={() => {
              setRevealed(false);
              setShowMediumConfidenceErrors(false);
              setShowExperimentalSuggestions(false);
            }}
            className="btn btn-secondary"
            style={{ fontSize: "13px", padding: "var(--spacing-xs) var(--spacing-sm)" }}
            lang="en"
          >
            Hide Feedback
          </button>
        </div>
      )}
      <div
        className="prose max-w-none notranslate"
        translate="no"
        lang="en"
        style={{
          padding: "var(--spacing-lg)",
          backgroundColor: revealed ? "var(--bg-secondary)" : "transparent",
          borderRadius: "var(--border-radius)",
          lineHeight: "1.5",
          fontSize: "16px",
          whiteSpace: "pre-wrap",
        }}
      >
        {revealed ? (
          // Show annotated text with errors when revealed
          <AnnotatedTextRevealed
            text={text}
            errors={highConfidenceErrors}
            showMediumConfidence={showMediumConfidenceErrors}
            showExperimental={showExperimentalSuggestions}
            mediumConfidenceErrors={mediumConfidenceErrors}
            experimentalErrors={lowConfidenceErrors}
          />
        ) : (
          // Show heat map
          elements
        )}
      </div>
    </div>
  );
}

// Individual error span component with popup
function ErrorSpan({
  errorKey,
  errorText,
  errorColor,
  isActive,
  onActivate,
  onDeactivate,
  error,
}: {
  errorKey: string;
  errorText: string;
  errorColor: string;
  isActive: boolean;
  onActivate: () => void;
  onDeactivate: () => void;
  error: LanguageToolError;
}) {
  const errorRef = useRef<HTMLSpanElement>(null);
  const popupRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!isActive || !popupRef.current || !errorRef.current) return;

    const positionPopup = () => {
      const popup = popupRef.current;
      const trigger = errorRef.current;
      if (!popup || !trigger) return;

      const rect = trigger.getBoundingClientRect();
      const popupRect = popup.getBoundingClientRect();

      // Get viewport dimensions
      const viewportWidth = window.innerWidth;
      const viewportHeight = window.innerHeight;
      const padding = 8; // Padding from viewport edges
      const isMobile = viewportWidth < 640; // Mobile breakpoint

      let top: number;
      let left: number;

      if (isMobile) {
        // On mobile, center horizontally and position below text
        left = Math.max(padding, (viewportWidth - popupRect.width) / 2);
        top = rect.bottom + 8; // 8px gap below the text

        // If popup would go off bottom, try above
        if (top + popupRect.height > viewportHeight - padding) {
          const topAbove = rect.top - popupRect.height - 8;
          if (topAbove >= padding) {
            top = topAbove;
          } else {
            // If can't fit above, center vertically
            top = Math.max(padding, (viewportHeight - popupRect.height) / 2);
          }
        }
      } else {
        // On desktop, try to position to the right of the text first
        top = rect.top;
        left = rect.right + 8; // 8px gap from the text

        // If popup would go off right edge, try left side
        if (left + popupRect.width > viewportWidth - padding) {
          const leftSide = rect.left - popupRect.width - 8;
          if (leftSide >= padding) {
            left = leftSide;
          } else {
            // If can't fit on left either, align to right edge
            left = viewportWidth - popupRect.width - padding;
          }
        }

        // Adjust horizontal position if popup would go off left edge
        if (left < padding) {
          left = padding;
        }

        // Adjust vertical position to keep popup aligned with text
        if (top + popupRect.height > viewportHeight - padding) {
          // Try above the error instead
          const topAbove = rect.top - popupRect.height - 8;
          if (topAbove >= padding) {
            top = topAbove;
          } else {
            // If can't fit above either, position at bottom of viewport
            top = viewportHeight - popupRect.height - padding;
          }
        }
      }

      // Ensure popup doesn't go above viewport
      if (top < padding) {
        top = padding;
      }

      // Apply calculated position
      popup.style.top = `${top}px`;
      popup.style.left = `${left}px`;
      popup.style.position = "fixed";
    };

    // Use requestAnimationFrame to ensure popup is rendered before positioning
    requestAnimationFrame(() => {
      requestAnimationFrame(positionPopup);
    });

    // Handle window resize
    window.addEventListener("resize", positionPopup);
    return () => window.removeEventListener("resize", positionPopup);
  }, [isActive]);

  // Close popup when clicking outside
  useEffect(() => {
    if (!isActive) return;

    const handleClickOutside = (event: MouseEvent) => {
      if (
        popupRef.current &&
        errorRef.current &&
        !popupRef.current.contains(event.target as Node) &&
        !errorRef.current.contains(event.target as Node)
      ) {
        onDeactivate();
      }
    };

    // Use a small delay to avoid closing immediately when opening
    const timeoutId = setTimeout(() => {
      document.addEventListener("mousedown", handleClickOutside);
    }, 100);

    return () => {
      clearTimeout(timeoutId);
      document.removeEventListener("mousedown", handleClickOutside);
    };
  }, [isActive, onDeactivate]);

  return (
    <>
      <span
        ref={errorRef}
        className="relative"
        translate="no"
        lang="en"
        style={{
          position: "relative",
          display: "inline",
          marginBottom: 0,
        }}
        onClick={(e) => {
          e.stopPropagation();
          if (isActive) {
            onDeactivate();
          } else {
            onActivate();
          }
        }}
      >
        <span
          className="underline decoration-wavy cursor-pointer"
          translate="no"
          lang="en"
          style={{
            textDecorationColor: errorColor,
            textDecorationThickness: "3px",
            textUnderlineOffset: "3px",
            backgroundColor: `${errorColor}15`,
            padding: "2px 2px",
            borderRadius: "3px",
            fontWeight: 500,
            cursor: "pointer",
          }}
        >
          {errorText}
        </span>
      </span>
      {/* Error detail popup */}
      <AnimatePresence>
        {isActive && (
          <motion.div
            ref={popupRef}
            style={{
              position: "fixed",
              padding: "var(--spacing-sm)",
              backgroundColor: "var(--bg-primary)",
              border: `1px solid ${errorColor}`,
              borderRadius: "var(--border-radius)",
              boxShadow: "0 2px 8px rgba(0,0,0,0.1)",
              minWidth: "250px",
              maxWidth: "min(400px, calc(100vw - 16px))",
              width: "max-content",
              zIndex: 1000,
            }}
            lang="en"
            onClick={(e) => e.stopPropagation()}
            initial={{ opacity: 0, y: -5, scale: 0.95 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: -5, scale: 0.95 }}
            transition={{ duration: 0.2, ease: [0.4, 0, 0.2, 1] }}
          >
            <ErrorDetail error={error} errorText={errorText} onClose={onDeactivate} />
          </motion.div>
        )}
      </AnimatePresence>
    </>
  );
}

function AnnotatedTextRevealed({
  text,
  errors,
  showMediumConfidence = false,
  showExperimental = false,
  mediumConfidenceErrors = [],
  experimentalErrors = [],
}: {
  text: string;
  errors: LanguageToolError[];
  showMediumConfidence?: boolean;
  showExperimental?: boolean;
  mediumConfidenceErrors?: LanguageToolError[];
  experimentalErrors?: LanguageToolError[];
}) {
  const [activeErrorKey, setActiveErrorKey] = useState<string | null>(null);

  // Combine errors based on toggles
  let allErrors = [...errors];
  if (showMediumConfidence) {
    allErrors = [...allErrors, ...mediumConfidenceErrors];
  }
  if (showExperimental) {
    allErrors = [...allErrors, ...experimentalErrors];
  }

  // Validate and filter errors - ensure positions are valid
  const validatedErrors: LanguageToolError[] = [];
  for (const error of allErrors) {
    if (
      !error ||
      typeof error.start !== "number" ||
      typeof error.end !== "number" ||
      error.start < 0 ||
      error.end > text.length ||
      error.start >= error.end
    ) {
      continue;
    }

    // Basic validation - ensure the position range is reasonable
    // Don't filter based on text content as positions should be correct from backend
    validatedErrors.push(error);
  }

  // Deduplicate errors with the same start/end positions
  // Keep the most helpful error for learners (best explanations + highest confidence)
  const errorMap = new Map<string, LanguageToolError>();
  for (const error of validatedErrors) {
    const key = `${error.start}-${error.end}`;
    const existing = errorMap.get(key);

    if (!existing) {
      errorMap.set(key, error);
    } else {
      // Score errors by how helpful they are to learners
      const scoreError = (e: LanguageToolError): number => {
        let score = 0;
        // Confidence is most important
        if (e.highConfidence === true) score += 100;
        else if (e.mediumConfidence === true) score += 50;
        else score += 10;

        // Prefer errors with better explanations (more helpful to learners)
        if (e.explanation) score += 20;
        if (e.example) score += 15;
        if (e.errorType) score += 10;
        if (e.suggestions && e.suggestions.length > 0) score += 5;

        return score;
      };

      const existingScore = scoreError(existing);
      const currentScore = scoreError(error);

      // Keep the error with the higher helpfulness score
      if (currentScore > existingScore) {
        errorMap.set(key, error);
      }
    }
  }

  const sorted = Array.from(errorMap.values()).sort((a, b) => a.start - b.start);

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
    const suggestion = error.suggestions?.[0];
    const category = error.category || "UNKNOWN";
    const message = error.message || "Error detected";
    const severity = error.severity || "error";
    const isMediumConfidence = error.mediumConfidence === true;
    const isExperimental = error.highConfidence === false && error.mediumConfidence !== true;

    // Use different colors for confidence tiers - more prominent colors
    const errorColor = isExperimental
      ? "#d97706" // Darker amber for experimental/low-confidence
      : isMediumConfidence
        ? "#ea580c" // Darker orange for medium-confidence
        : severity === "error"
          ? "#dc2626" // Red for high-confidence errors
          : "#d97706"; // Amber for warnings

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

  return (
    <div translate="no" lang="en">
      {elements}
    </div>
  );
}
