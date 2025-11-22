/**
 * Draft table header component
 */

export function DraftTableHeader() {
  return (
    <thead>
      <tr style={{ borderBottom: "2px solid var(--border-color)" }}>
        <th
          style={{
            textAlign: "left",
            padding: "var(--spacing-sm)",
            fontWeight: 600,
            color: "var(--text-primary)",
          }}
          lang="en"
        >
          Draft
        </th>
        <th
          style={{
            textAlign: "right",
            padding: "var(--spacing-sm)",
            fontWeight: 600,
            color: "var(--text-primary)",
          }}
          lang="en"
        >
          Score
        </th>
        <th
          style={{
            textAlign: "right",
            padding: "var(--spacing-sm)",
            fontWeight: 600,
            color: "var(--text-primary)",
          }}
          lang="en"
        >
          Words
        </th>
        <th
          style={{
            textAlign: "right",
            padding: "var(--spacing-sm)",
            fontWeight: 600,
            color: "var(--text-primary)",
          }}
          lang="en"
        >
          Errors
        </th>
        <th
          style={{
            textAlign: "left",
            padding: "var(--spacing-sm)",
            fontWeight: 600,
            color: "var(--text-primary)",
          }}
          lang="en"
        >
          Change
        </th>
      </tr>
    </thead>
  );
}
