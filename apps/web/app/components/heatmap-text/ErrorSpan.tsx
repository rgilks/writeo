import { useMemo } from "react";
import type { ErrorSpanProps } from "./types";

const BASE_STYLES = {
  textDecorationThickness: "3px",
  textUnderlineOffset: "3px",
  padding: "2px 2px",
  borderRadius: "3px",
  transition: "all 0.2s ease",
  outlineOffset: "2px",
} as const;

export function ErrorSpan({
  errorText,
  errorColor,
  isActive,
  onActivate,
  onDeactivate,
}: ErrorSpanProps) {
  const styles = useMemo(
    () => ({
      ...BASE_STYLES,
      textDecorationColor: errorColor,
      backgroundColor: isActive ? `${errorColor}30` : `${errorColor}15`,
      fontWeight: isActive ? 600 : 500,
      boxShadow: isActive ? `0 0 0 2px ${errorColor}40` : "none",
      outline: isActive ? `2px solid ${errorColor}` : "none",
    }),
    [errorColor, isActive],
  );

  return (
    <span
      data-error-span
      className="underline decoration-wavy cursor-pointer"
      translate="no"
      style={styles}
      onClick={(e) => {
        e.stopPropagation();
        isActive ? onDeactivate() : onActivate();
      }}
    >
      {errorText}
    </span>
  );
}
