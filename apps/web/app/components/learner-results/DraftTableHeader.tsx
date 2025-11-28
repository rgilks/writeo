/**
 * Draft table header component
 */

const headerBaseStyle = {
  padding: "var(--spacing-sm)",
  fontWeight: 600,
  color: "var(--text-primary)",
} as const;

const columns = [
  { label: "Draft", align: "left" },
  { label: "Score", align: "right" },
  { label: "Words", align: "right" },
  { label: "Errors", align: "right" },
  { label: "Change", align: "left" },
] as const;

export function DraftTableHeader() {
  return (
    <thead>
      <tr style={{ borderBottom: "2px solid var(--border-color)" }}>
        {columns.map(({ label, align }) => (
          <th key={label} style={{ ...headerBaseStyle, textAlign: align }} lang="en">
            {label}
          </th>
        ))}
      </tr>
    </thead>
  );
}
