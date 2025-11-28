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
  const entries = [
    {
      value: scoreChange,
      label: "score",
      formatter: (value: number) => value.toFixed(1),
      getColor: (value: number) => (value > 0 ? "var(--secondary-accent)" : "var(--error-color)"),
    },
    {
      value: wordChange,
      label: "words",
      formatter: (value: number) => value.toString(),
    },
    {
      value: errorChange,
      label: "errors",
      formatter: (value: number) => value.toString(),
      getColor: (value: number) =>
        value < 0 ? "var(--secondary-accent)" : value > 0 ? "var(--error-color)" : "inherit",
    },
  ];

  const renderedChanges = entries
    .filter(({ value }) => value !== null && value !== 0)
    .map(({ value, label, formatter, getColor }) => {
      if (value === null) {
        return null;
      }

      const prefix = value > 0 ? "+" : "";
      const color = getColor?.(value);

      return (
        <span key={label} style={color ? { color } : undefined}>
          {prefix}
          {formatter(value)} {label}
        </span>
      );
    })
    .filter(Boolean);

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "2px" }}>
      {renderedChanges.length > 0 ? renderedChanges : <span>No change</span>}
    </div>
  );
}
