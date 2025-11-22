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
  errorKey: string;
  errorText: string;
  errorColor: string;
  isActive: boolean;
  onActivate: () => void;
  onDeactivate: () => void;
  error: LanguageToolError;
}

export interface AnnotatedTextRevealedProps {
  text: string;
  errors: LanguageToolError[];
  showMediumConfidence?: boolean;
  showExperimental?: boolean;
  mediumConfidenceErrors?: LanguageToolError[];
  experimentalErrors?: LanguageToolError[];
}
