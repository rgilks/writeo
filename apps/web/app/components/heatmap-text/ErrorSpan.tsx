/**
 * Error span component - highlights text and triggers suggestion display
 */

import { useRef } from "react";
import type { ErrorSpanProps } from "./types";

export function ErrorSpan({
  errorKey,
  errorText,
  errorColor,
  isActive,
  onActivate,
  onDeactivate,
  error,
}: ErrorSpanProps) {
  const errorRef = useRef<HTMLSpanElement>(null);

  // Removed scroll behavior - scrolling is now handled by parent component
  // to show the explanation at the top instead of centering the error span

  return (
    <span
      ref={errorRef}
      data-error-span
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
          backgroundColor: isActive ? `${errorColor}30` : `${errorColor}15`,
          padding: "2px 2px",
          borderRadius: "3px",
          fontWeight: isActive ? 600 : 500,
          cursor: "pointer",
          boxShadow: isActive ? `0 0 0 2px ${errorColor}40` : "none",
          transition: "all 0.2s ease",
          outline: isActive ? `2px solid ${errorColor}` : "none",
          outlineOffset: "2px",
        }}
      >
        {errorText}
      </span>
    </span>
  );
}
