/**
 * Component for displaying error types with explanations
 */

import { useState } from "react";
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

  return (
    <li style={{ marginBottom: "var(--spacing-sm)", fontSize: "16px" }} lang="en">
      <div
        style={{
          display: "flex",
          alignItems: "center",
          gap: "var(--spacing-sm)",
          cursor: "pointer",
        }}
        onClick={() => setIsExpanded(!isExpanded)}
        lang="en"
      >
        <span style={{ fontSize: "12px", color: "var(--text-secondary)" }}>
          {isExpanded ? "▼" : "▶"}
        </span>
        <strong>{type}</strong>
        <span style={{ color: "var(--text-secondary)" }}>
          ({count} {count === 1 ? "time" : "times"})
        </span>
      </div>
      {isExpanded && (
        <div
          style={{
            marginTop: "var(--spacing-xs)",
            marginLeft: "var(--spacing-md)",
            padding: "var(--spacing-sm)",
            backgroundColor: "rgba(102, 126, 234, 0.1)",
            borderRadius: "var(--border-radius)",
            fontSize: "14px",
            lineHeight: "1.5",
          }}
          lang="en"
        >
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
