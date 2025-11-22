/**
 * Change cell component for draft comparison table
 */

export function ChangeCell({
  scoreChange,
  wordChange,
  errorChange,
}: {
  scoreChange: number | null;
  wordChange: number | null;
  errorChange: number | null;
}) {
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "2px" }}>
      {scoreChange !== null && scoreChange !== 0 && (
        <span
          style={{
            color: scoreChange > 0 ? "var(--secondary-accent)" : "var(--error-color)",
          }}
        >
          {scoreChange > 0 ? "+" : ""}
          {scoreChange.toFixed(1)} score
        </span>
      )}
      {wordChange !== null && wordChange !== 0 && (
        <span>
          {wordChange > 0 ? "+" : ""}
          {wordChange} words
        </span>
      )}
      {errorChange !== null && errorChange !== 0 && (
        <span
          style={{
            color:
              errorChange < 0
                ? "var(--secondary-accent)"
                : errorChange > 0
                  ? "var(--error-color)"
                  : "inherit",
          }}
        >
          {errorChange > 0 ? "+" : ""}
          {errorChange} errors
        </span>
      )}
      {scoreChange === 0 && wordChange === 0 && errorChange === 0 && <span>No change</span>}
    </div>
  );
}
