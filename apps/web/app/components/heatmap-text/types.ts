/**
 * Type definitions for HeatMapText component
 */

import type { LanguageToolError } from "@writeo/shared";

export interface HeatMapTextProps {
  text: string;
  errors: LanguageToolError[];
  onReveal?: () => void;
  showMediumConfidence?: boolean;
  showExperimental?: boolean;
}

export interface ErrorDetailProps {
  error: LanguageToolError;
  errorText: string;
  onClose?: () => void;
}

export interface ErrorSpanProps {
  errorText: string;
  errorColor: string;
  isActive: boolean;
  onActivate: () => void;
  onDeactivate: () => void;
}

export interface AnnotatedTextRevealedProps {
  text: string;
  errors: LanguageToolError[];
  showMediumConfidence?: boolean;
  showExperimental?: boolean;
  mediumConfidenceErrors?: LanguageToolError[];
  experimentalErrors?: LanguageToolError[];
}

export interface FeedbackControlsProps {
  mediumConfidenceErrors: LanguageToolError[];
  lowConfidenceErrors: LanguageToolError[];
  showMediumConfidenceErrors: boolean;
  showExperimentalSuggestions: boolean;
  onToggleMediumConfidence: () => void;
  onToggleExperimental: () => void;
  onHideFeedback: () => void;
}

export interface HeatMapRendererProps {
  text: string;
  normalizedIntensity: number[];
  revealed: boolean;
}

interface BaseErrorControlProps {
  mediumConfidenceErrors: LanguageToolError[];
  lowConfidenceErrors: LanguageToolError[];
  showMediumConfidenceErrors: boolean;
  showExperimentalSuggestions: boolean;
  onShowMediumConfidence: () => void;
  onShowExperimental: () => void;
}

export interface NoErrorsMessageProps extends BaseErrorControlProps {}

export interface RevealPromptProps extends BaseErrorControlProps {
  onReveal: () => void;
}
