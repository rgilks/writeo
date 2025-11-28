"use client";

import { useState, useMemo, useCallback } from "react";
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

  const { highConfidenceErrors, mediumConfidenceErrors, lowConfidenceErrors } = useMemo(
    () => filterErrorsByConfidence(errors),
    [errors],
  );

  const filteredErrors = useMemo(
    () =>
      buildFilteredErrors(
        highConfidenceErrors,
        mediumConfidenceErrors,
        lowConfidenceErrors,
        showMediumConfidenceErrors,
        showExperimentalSuggestions,
      ),
    [
      highConfidenceErrors,
      mediumConfidenceErrors,
      lowConfidenceErrors,
      showMediumConfidenceErrors,
      showExperimentalSuggestions,
    ],
  );

  const { intensityMap, normalizedIntensity } = useMemo(() => {
    const map = calculateIntensityMap(text, filteredErrors);
    return {
      intensityMap: map,
      normalizedIntensity: normalizeIntensity(map),
    };
  }, [text, filteredErrors]);

  const handleReveal = useCallback(() => {
    setRevealed(true);
    onReveal?.();
  }, [onReveal]);

  const handleShowMediumConfidence = useCallback(() => {
    setShowMediumConfidenceErrors(true);
  }, []);

  const handleShowExperimental = useCallback(() => {
    setShowExperimentalSuggestions(true);
  }, []);

  const handleToggleMediumConfidence = useCallback(() => {
    setShowMediumConfidenceErrors((prev) => !prev);
  }, []);

  const handleToggleExperimental = useCallback(() => {
    setShowExperimentalSuggestions((prev) => !prev);
  }, []);

  const handleHideFeedback = useCallback(() => {
    setRevealed(false);
    setShowMediumConfidenceErrors(false);
    setShowExperimentalSuggestions(false);
  }, []);

  if (!text) {
    return null;
  }

  if (!errors || errors.length === 0) {
    return (
      <div className="prose max-w-none" translate="no">
        {text}
      </div>
    );
  }

  if (filteredErrors.length === 0) {
    return (
      <NoErrorsMessage
        mediumConfidenceErrors={mediumConfidenceErrors}
        lowConfidenceErrors={lowConfidenceErrors}
        showMediumConfidenceErrors={showMediumConfidenceErrors}
        showExperimentalSuggestions={showExperimentalSuggestions}
        onShowMediumConfidence={handleShowMediumConfidence}
        onShowExperimental={handleShowExperimental}
      />
    );
  }

  return (
    <div>
      {!revealed && (
        <RevealPrompt
          onReveal={handleReveal}
          mediumConfidenceErrors={mediumConfidenceErrors}
          lowConfidenceErrors={lowConfidenceErrors}
          showMediumConfidenceErrors={showMediumConfidenceErrors}
          showExperimentalSuggestions={showExperimentalSuggestions}
          onShowMediumConfidence={handleShowMediumConfidence}
          onShowExperimental={handleShowExperimental}
        />
      )}

      {revealed && (
        <>
          <FeedbackControls
            mediumConfidenceErrors={mediumConfidenceErrors}
            lowConfidenceErrors={lowConfidenceErrors}
            showMediumConfidenceErrors={showMediumConfidenceErrors}
            showExperimentalSuggestions={showExperimentalSuggestions}
            onToggleMediumConfidence={handleToggleMediumConfidence}
            onToggleExperimental={handleToggleExperimental}
            onHideFeedback={handleHideFeedback}
          />
          <AnnotatedTextRevealed
            text={text}
            errors={highConfidenceErrors}
            showMediumConfidence={showMediumConfidenceErrors}
            showExperimental={showExperimentalSuggestions}
            mediumConfidenceErrors={mediumConfidenceErrors}
            experimentalErrors={lowConfidenceErrors}
          />
        </>
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
