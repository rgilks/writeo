/**
 * Component for displaying error types with explanations
 */

import { useId, useState } from "react";
import type { LanguageToolError } from "@writeo/shared";
import { getErrorExplanation } from "./utils";

export function ErrorTypeItem({
  type,
  count,
  examples,
}: {
  type: string;
  count: number;
  examples: LanguageToolError[];
}) {
  const [isExpanded, setIsExpanded] = useState(false);
  const contentId = useId();
  const toggleExpansion = () => setIsExpanded((prev) => !prev);

  const listItemStyle = { marginBottom: "var(--spacing-sm)", fontSize: "16px" } as const;
  const toggleButtonStyle = {
    display: "flex",
    alignItems: "center",
    gap: "var(--spacing-sm)",
    cursor: "pointer",
    width: "100%",
    background: "transparent",
    border: "none",
    padding: 0,
    color: "var(--text-primary)",
    textAlign: "left" as const,
  };
  const expandedIconStyle = { fontSize: "12px", color: "var(--text-secondary)" } as const;
  const countStyle = { color: "var(--text-secondary)" } as const;
  const detailsStyle = {
    marginTop: "var(--spacing-xs)",
    marginLeft: "var(--spacing-md)",
    padding: "var(--spacing-sm)",
    backgroundColor: "var(--primary-bg-light)",
    borderRadius: "var(--border-radius)",
    fontSize: "14px",
    lineHeight: "1.5",
  } as const;

  return (
    <li style={listItemStyle} lang="en">
      <button
        type="button"
        onClick={toggleExpansion}
        aria-expanded={isExpanded}
        aria-controls={contentId}
        style={toggleButtonStyle}
        lang="en"
      >
        <span style={expandedIconStyle}>{isExpanded ? "▼" : "▶"}</span>
        <strong>{type}</strong>
        <span style={countStyle}>
          ({count} {count === 1 ? "time" : "times"})
        </span>
      </button>
      {isExpanded && (
        <div style={detailsStyle} id={contentId} lang="en">
          <p style={{ marginBottom: "var(--spacing-xs)", fontWeight: 600 }} lang="en">
            {getErrorExplanation(type, count)}
          </p>
          {examples.length > 0 && (
            <div lang="en">
              <p style={{ marginBottom: "4px", fontSize: "12px", fontWeight: 600 }} lang="en">
                Examples from your essay:
              </p>
              <ul
                style={{ margin: 0, paddingLeft: "var(--spacing-md)", fontSize: "13px" }}
                lang="en"
              >
                {examples.map((err, idx) => (
                  <li key={idx} lang="en">
                    {err.message}
                    {err.suggestions && err.suggestions.length > 0 && (
                      <span style={{ color: "var(--text-secondary)" }}>
                        {" "}
                        → Try: {err.suggestions[0]}
                      </span>
                    )}
                  </li>
                ))}
              </ul>
            </div>
          )}
        </div>
      )}
    </li>
  );
}
