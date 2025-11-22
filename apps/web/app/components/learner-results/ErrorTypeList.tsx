/**
 * Error type list component
 */

import { getTopErrorTypes } from "@/app/lib/utils/progress";
import type { LanguageToolError } from "@writeo/shared";
import { ErrorTypeItem } from "./ErrorTypeItem";

export function ErrorTypeList({ grammarErrors }: { grammarErrors: LanguageToolError[] }) {
  const topErrorTypes = getTopErrorTypes(grammarErrors, 3);

  if (!topErrorTypes || topErrorTypes.length === 0) {
    return null;
  }

  return (
    <div className="card" lang="en">
      <h2
        style={{
          fontSize: "20px",
          marginBottom: "var(--spacing-md)",
          fontWeight: 600,
        }}
        lang="en"
      >
        Common Areas to Improve
      </h2>
      <p
        style={{
          marginBottom: "var(--spacing-md)",
          fontSize: "16px",
          color: "var(--text-secondary)",
        }}
        lang="en"
      >
        You made these types of errors most often:
      </p>
      <ul style={{ margin: 0, paddingLeft: "var(--spacing-md)" }} lang="en">
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
