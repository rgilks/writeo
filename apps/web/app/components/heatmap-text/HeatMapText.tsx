/**
 * HeatMapText component - shows error intensity as a heat map without revealing details
 */

"use client";

import { useState } from "react";
import type { HeatMapTextProps } from "./types";
import { AnnotatedTextRevealed } from "./AnnotatedTextRevealed";
import { NoErrorsMessage } from "./NoErrorsMessage";
import { RevealPrompt } from "./RevealPrompt";
import { FeedbackControls } from "./FeedbackControls";
import { HeatMapRenderer } from "./HeatMapRenderer";
import { filterErrorsByConfidence, buildFilteredErrors } from "./utils";
import { calculateIntensityMap, normalizeIntensity } from "./intensity";

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

  const { highConfidenceErrors, mediumConfidenceErrors, lowConfidenceErrors } =
    filterErrorsByConfidence(errors);

  const filteredErrors = buildFilteredErrors(
    highConfidenceErrors,
    mediumConfidenceErrors,
    lowConfidenceErrors,
    showMediumConfidenceErrors,
    showExperimentalSuggestions
  );

  if (!errors || errors.length === 0 || !text) {
    return (
      <div className="prose max-w-none" translate="no" lang="en">
        {text || ""}
      </div>
    );
  }

  if (filteredErrors.length === 0 && !showExperimentalSuggestions) {
    return (
      <NoErrorsMessage
        mediumConfidenceErrors={mediumConfidenceErrors}
        lowConfidenceErrors={lowConfidenceErrors}
        showMediumConfidenceErrors={showMediumConfidenceErrors}
        showExperimentalSuggestions={showExperimentalSuggestions}
        onShowMediumConfidence={() => setShowMediumConfidenceErrors(true)}
        onShowExperimental={() => setShowExperimentalSuggestions(true)}
      />
    );
  }

  const intensityMap = calculateIntensityMap(text, filteredErrors);
  const normalizedIntensity = normalizeIntensity(intensityMap);

  return (
    <div>
      {!revealed && (
        <RevealPrompt
          onReveal={() => {
            setRevealed(true);
            onReveal?.();
          }}
          mediumConfidenceErrors={mediumConfidenceErrors}
          lowConfidenceErrors={lowConfidenceErrors}
          showMediumConfidenceErrors={showMediumConfidenceErrors}
          showExperimentalSuggestions={showExperimentalSuggestions}
          onShowMediumConfidence={() => setShowMediumConfidenceErrors(true)}
          onShowExperimental={() => setShowExperimentalSuggestions(true)}
        />
      )}

      {revealed && (
        <FeedbackControls
          mediumConfidenceErrors={mediumConfidenceErrors}
          lowConfidenceErrors={lowConfidenceErrors}
          showMediumConfidenceErrors={showMediumConfidenceErrors}
          showExperimentalSuggestions={showExperimentalSuggestions}
          onToggleMediumConfidence={() =>
            setShowMediumConfidenceErrors(!showMediumConfidenceErrors)
          }
          onToggleExperimental={() => setShowExperimentalSuggestions(!showExperimentalSuggestions)}
          onHideFeedback={() => {
            setRevealed(false);
            setShowMediumConfidenceErrors(false);
            setShowExperimentalSuggestions(false);
          }}
        />
      )}

      {revealed && (
        <AnnotatedTextRevealed
          text={text}
          errors={highConfidenceErrors}
          showMediumConfidence={showMediumConfidenceErrors}
          showExperimental={showExperimentalSuggestions}
          mediumConfidenceErrors={mediumConfidenceErrors}
          experimentalErrors={lowConfidenceErrors}
        />
      )}

      {!revealed && (
        <HeatMapRenderer
          text={text}
          normalizedIntensity={normalizedIntensity}
          revealed={revealed}
        />
      )}
    </div>
  );
}
