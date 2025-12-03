import { useId, useState } from "react";
import type { LanguageToolError } from "@writeo/shared";
import { AnimatePresence, motion } from "framer-motion";
import { getErrorIcon } from "./ErrorIcons";
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

  return (
    <li
      style={{
        marginBottom: "var(--spacing-md)",
        listStyle: "none",
      }}
    >
      <button
        type="button"
        onClick={toggleExpansion}
        aria-expanded={isExpanded}
        aria-controls={contentId}
        style={{
          display: "flex",
          alignItems: "center",
          width: "100%",
          background: "var(--bg-surface, #fff)", // Fallback or variable
          border: "1px solid var(--border-color, #e2e8f0)",
          borderRadius: "var(--border-radius, 8px)",
          padding: "var(--spacing-md)",
          cursor: "pointer",
          textAlign: "left",
          transition: "background-color 0.2s",
          position: "relative",
          boxShadow: "0 1px 3px rgba(0,0,0,0.05)",
        }}
        onMouseEnter={(e) => {
          e.currentTarget.style.backgroundColor = "var(--bg-surface-hover, #f8fafc)";
        }}
        onMouseLeave={(e) => {
          e.currentTarget.style.backgroundColor = "var(--bg-surface, #fff)";
        }}
      >
        <div
          style={{
            marginRight: "var(--spacing-md)",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            width: "40px",
            height: "40px",
            background: "var(--primary-bg-light, #eff6ff)",
            borderRadius: "50%",
            flexShrink: 0,
          }}
        >
          {getErrorIcon(type)}
        </div>

        <div style={{ flex: 1 }}>
          <div
            style={{
              fontSize: "16px",
              fontWeight: 600,
              color: "var(--text-primary, #1e293b)",
              marginBottom: "4px",
            }}
          >
            {type}
          </div>
          <div style={{ fontSize: "14px", color: "var(--text-secondary, #64748b)" }}>
            {count} {count === 1 ? "issue" : "issues"} found
          </div>
        </div>

        <motion.div
          animate={{ rotate: isExpanded ? 180 : 0 }}
          transition={{ duration: 0.2 }}
          style={{
            color: "var(--text-secondary, #94a3b8)",
            marginLeft: "var(--spacing-sm)",
          }}
        >
          <svg
            xmlns="http://www.w3.org/2000/svg"
            width="20"
            height="20"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="2"
            strokeLinecap="round"
            strokeLinejoin="round"
          >
            <polyline points="6 9 12 15 18 9"></polyline>
          </svg>
        </motion.div>
      </button>

      <AnimatePresence>
        {isExpanded && (
          <motion.div
            id={contentId}
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: "auto", opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.2 }}
            style={{ overflow: "hidden" }}
          >
            <div
              style={{
                padding: "var(--spacing-md)",
                paddingTop: 0,
                borderLeft: "1px solid var(--border-color, #e2e8f0)",
                borderRight: "1px solid var(--border-color, #e2e8f0)",
                borderBottom: "1px solid var(--border-color, #e2e8f0)",
                borderBottomLeftRadius: "var(--border-radius, 8px)",
                borderBottomRightRadius: "var(--border-radius, 8px)",
                background: "var(--bg-surface-subtle, #f8fafc)",
                marginTop: "-1px", // Merge borders
                position: "relative",
                zIndex: 0,
              }}
            >
              <p
                style={{
                  marginBottom: "var(--spacing-md)",
                  color: "var(--text-primary, #334155)",
                  lineHeight: 1.5,
                }}
              >
                {getErrorExplanation(type, count)}
              </p>

              {examples.length > 0 && (
                <div>
                  <p
                    style={{
                      fontSize: "12px",
                      fontWeight: 600,
                      textTransform: "uppercase",
                      letterSpacing: "0.05em",
                      color: "var(--text-tertiary, #94a3b8)",
                      marginBottom: "var(--spacing-sm)",
                    }}
                  >
                    Examples
                  </p>
                  <ul style={{ margin: 0, padding: 0, listStyle: "none" }}>
                    {examples.map((err, idx) => (
                      <li
                        key={idx}
                        style={{
                          background: "var(--bg-surface, #fff)",
                          border: "1px solid var(--border-color, #e2e8f0)",
                          borderRadius: "var(--border-radius-sm, 4px)",
                          padding: "var(--spacing-sm) var(--spacing-md)",
                          marginBottom: "var(--spacing-sm)",
                          fontSize: "14px",
                        }}
                      >
                        <div style={{ marginBottom: "4px", color: "var(--text-primary, #334155)" }}>
                          {err.message}
                        </div>
                        {err.suggestions && err.suggestions.length > 0 && (
                          <div
                            style={{
                              color: "var(--success-text, #059669)",
                              display: "flex",
                              alignItems: "center",
                              gap: "4px",
                              fontSize: "13px",
                              fontWeight: 500,
                            }}
                          >
                            <span>→ Try:</span>
                            <span
                              style={{
                                background: "var(--success-bg, #d1fae5)",
                                padding: "2px 6px",
                                borderRadius: "4px",
                              }}
                            >
                              {err.suggestions[0]}
                            </span>
                          </div>
                        )}
                      </li>
                    ))}
                  </ul>
                </div>
              )}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </li>
  );
}
