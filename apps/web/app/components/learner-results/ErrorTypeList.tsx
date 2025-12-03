/**
 * Error type list component
 */

import { getTopErrorTypes } from "@/app/lib/utils/progress";
import type { LanguageToolError } from "@writeo/shared";
import { ErrorTypeItem } from "./ErrorTypeItem";

export function ErrorTypeList({ grammarErrors }: { grammarErrors: LanguageToolError[] }) {
  const topErrorTypes = getTopErrorTypes(grammarErrors, 3);

  if (!topErrorTypes?.length) {
    return null;
  }

  return (
    <div
      className="card"
      lang="en"
      style={{
        padding: "var(--spacing-lg, 24px)",
        background: "var(--bg-surface, #fff)",
        borderRadius: "var(--border-radius-lg, 12px)",
        boxShadow: "0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06)",
      }}
      data-testid="grammar-errors-section"
    >
      <h2
        style={{
          fontSize: "22px",
          marginBottom: "var(--spacing-xs, 4px)",
          fontWeight: 700,
          color: "var(--text-primary, #1e293b)",
          letterSpacing: "-0.01em",
        }}
        lang="en"
      >
        Common Areas to Improve
      </h2>
      <p
        style={{
          marginBottom: "var(--spacing-lg, 24px)",
          fontSize: "15px",
          color: "var(--text-secondary, #64748b)",
          lineHeight: 1.5,
        }}
        lang="en"
      >
        You made these types of errors most often:
      </p>
      <ul
        style={{
          margin: 0,
          padding: 0,
          listStyle: "none",
          display: "flex",
          flexDirection: "column",
          gap: "var(--spacing-md, 16px)",
        }}
        lang="en"
      >
        {topErrorTypes.map(({ type, count }) => {
          const exampleErrors = grammarErrors
            .filter((err) => (err.errorType || err.category) === type)
            .slice(0, 2);

          return <ErrorTypeItem key={type} type={type} count={count} examples={exampleErrors} />;
        })}
      </ul>
    </div>
  );
}
