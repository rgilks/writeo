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

  const cardStyle = {
    padding: "var(--spacing-md)",
  } as const;

  const headingStyle = {
    fontSize: "20px",
    marginBottom: "var(--spacing-md)",
    fontWeight: 600,
  } as const;

  const descriptionStyle = {
    marginBottom: "var(--spacing-md)",
    fontSize: "16px",
    color: "var(--text-secondary)",
  } as const;

  const listStyle = { margin: 0, paddingLeft: "var(--spacing-md)" } as const;

  return (
    <div className="card" lang="en" style={cardStyle} data-testid="grammar-errors-section">
      <h2 style={headingStyle} lang="en">
        Common Areas to Improve
      </h2>
      <p style={descriptionStyle} lang="en">
        You made these types of errors most often:
      </p>
      <ul style={listStyle} lang="en">
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
