import { useMemo, useState } from "react";
import type { ErrorSpanProps } from "./types";

const BASE_STYLES = {
  textDecorationThickness: "3px",
  textUnderlineOffset: "3px",
  padding: "2px 2px",
  borderRadius: "3px",
  transition: "all 0.2s ease",
  outlineOffset: "2px",
  cursor: "pointer",
  userSelect: "none" as const,
  WebkitUserSelect: "none" as const,
} as const;

export function ErrorSpan({
  errorText,
  errorColor,
  isActive,
  onActivate,
  onDeactivate,
}: ErrorSpanProps) {
  const [isHovered, setIsHovered] = useState(false);

  const styles = useMemo(
    () => ({
      ...BASE_STYLES,
      textDecorationColor: errorColor,
      backgroundColor: isActive
        ? `${errorColor}30`
        : isHovered
          ? `${errorColor}25`
          : `${errorColor}15`,
      fontWeight: isActive ? 600 : isHovered ? 550 : 500,
      boxShadow: isActive
        ? `0 0 0 2px ${errorColor}40`
        : isHovered
          ? `0 0 0 1px ${errorColor}30`
          : "none",
      outline: isActive ? `2px solid ${errorColor}` : "none",
      transform: isHovered && !isActive ? "scale(1.02)" : "scale(1)",
    }),
    [errorColor, isActive, isHovered],
  );

  return (
    <span
      data-error-span
      className="underline decoration-wavy"
      translate="no"
      style={styles}
      title="Click to see feedback"
      onMouseEnter={() => setIsHovered(true)}
      onMouseLeave={() => setIsHovered(false)}
      onClick={(e) => {
        e.stopPropagation();
        if (isActive) {
          onDeactivate();
        } else {
          onActivate();
        }
      }}
      onKeyDown={(e) => {
        if (e.key === "Enter" || e.key === " ") {
          e.preventDefault();
          e.stopPropagation();
          if (isActive) {
            onDeactivate();
          } else {
            onActivate();
          }
        }
      }}
      tabIndex={0}
      role="button"
      aria-label={`Error: ${errorText}. Click to see feedback`}
    >
      {errorText}
    </span>
  );
}
