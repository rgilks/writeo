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
          background: "var(--bg-surface, #fff)",
          border: isExpanded
            ? "1px solid var(--border-color, #e2e8f0)"
            : "1px solid var(--border-color, #e2e8f0)",
          borderRadius: isExpanded
            ? "var(--border-radius, 8px) var(--border-radius, 8px) 0 0"
            : "var(--border-radius, 8px)",
          padding: "var(--spacing-md, 16px)",
          cursor: "pointer",
          textAlign: "left",
          transition: "all 0.2s ease",
          position: "relative",
          boxShadow: isExpanded ? "0 2px 4px rgba(0,0,0,0.06)" : "0 1px 2px rgba(0,0,0,0.04)",
        }}
        onMouseEnter={(e) => {
          e.currentTarget.style.backgroundColor = "var(--bg-surface-hover, #f8fafc)";
          e.currentTarget.style.boxShadow = "0 2px 4px rgba(0,0,0,0.08)";
        }}
        onMouseLeave={(e) => {
          e.currentTarget.style.backgroundColor = "var(--bg-surface, #fff)";
          e.currentTarget.style.boxShadow = isExpanded
            ? "0 2px 4px rgba(0,0,0,0.06)"
            : "0 1px 2px rgba(0,0,0,0.04)";
        }}
      >
        <div
          style={{
            marginRight: "var(--spacing-md, 16px)",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            width: "44px",
            height: "44px",
            background: "var(--primary-bg-light, #eff6ff)",
            borderRadius: "50%",
            flexShrink: 0,
            transition: "transform 0.2s ease",
          }}
        >
          <div style={{ color: "var(--primary-color, #3b82f6)", display: "flex" }}>
            {getErrorIcon(type)}
          </div>
        </div>

        <div style={{ flex: 1, minWidth: 0 }}>
          <div
            style={{
              fontSize: "16px",
              fontWeight: 600,
              color: "var(--text-primary, #1e293b)",
              marginBottom: "2px",
              lineHeight: 1.4,
            }}
          >
            {type}
          </div>
          <div
            style={{
              fontSize: "13px",
              color: "var(--text-secondary, #64748b)",
              lineHeight: 1.4,
            }}
          >
            {count} {count === 1 ? "issue" : "issues"} found
          </div>
        </div>

        <motion.div
          animate={{ rotate: isExpanded ? 180 : 0 }}
          transition={{ duration: 0.2, ease: "easeInOut" }}
          style={{
            color: "var(--text-secondary, #94a3b8)",
            marginLeft: "var(--spacing-sm, 8px)",
            flexShrink: 0,
          }}
        >
          <svg
            xmlns="http://www.w3.org/2000/svg"
            width="18"
            height="18"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="2.5"
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
                padding: "var(--spacing-md, 16px)",
                paddingTop: "var(--spacing-md, 16px)",
                borderLeft: "1px solid var(--border-color, #e2e8f0)",
                borderRight: "1px solid var(--border-color, #e2e8f0)",
                borderBottom: "1px solid var(--border-color, #e2e8f0)",
                borderBottomLeftRadius: "var(--border-radius, 8px)",
                borderBottomRightRadius: "var(--border-radius, 8px)",
                background: "var(--bg-surface-subtle, #f8fafc)",
                marginTop: "-1px",
                position: "relative",
                zIndex: 0,
              }}
            >
              <p
                style={{
                  marginBottom: "var(--spacing-md, 16px)",
                  color: "var(--text-primary, #334155)",
                  lineHeight: 1.6,
                  fontSize: "14px",
                }}
              >
                {getErrorExplanation(type, count)}
              </p>

              {examples.length > 0 && (
                <div>
                  <p
                    style={{
                      fontSize: "11px",
                      fontWeight: 600,
                      textTransform: "uppercase",
                      letterSpacing: "0.08em",
                      color: "var(--text-tertiary, #94a3b8)",
                      marginBottom: "var(--spacing-sm, 8px)",
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
                          borderRadius: "var(--border-radius-sm, 6px)",
                          padding: "var(--spacing-sm, 8px) var(--spacing-md, 16px)",
                          marginBottom: "var(--spacing-sm, 8px)",
                          fontSize: "13px",
                          transition: "box-shadow 0.2s ease",
                        }}
                        onMouseEnter={(e) => {
                          e.currentTarget.style.boxShadow = "0 1px 3px rgba(0,0,0,0.08)";
                        }}
                        onMouseLeave={(e) => {
                          e.currentTarget.style.boxShadow = "none";
                        }}
                      >
                        <div
                          style={{
                            marginBottom: "6px",
                            color: "var(--text-primary, #334155)",
                            lineHeight: 1.5,
                          }}
                        >
                          {err.message}
                        </div>
                        {err.suggestions && err.suggestions.length > 0 && (
                          <div
                            style={{
                              color: "var(--success-text, #059669)",
                              display: "flex",
                              alignItems: "center",
                              gap: "6px",
                              fontSize: "13px",
                              fontWeight: 500,
                            }}
                          >
                            <span style={{ flexShrink: 0 }}>→ Try:</span>
                            <span
                              style={{
                                background: "var(--success-bg, #d1fae5)",
                                padding: "3px 8px",
                                borderRadius: "4px",
                                fontFamily: "monospace",
                                fontSize: "12px",
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
