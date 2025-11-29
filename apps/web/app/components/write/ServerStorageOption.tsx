"use client";

interface ServerStorageOptionProps {
  storeResults: boolean;
  onStoreResultsChange: (checked: boolean) => void;
}

export function ServerStorageOption({
  storeResults,
  onStoreResultsChange,
}: ServerStorageOptionProps) {
  return (
    <div
      style={{
        marginTop: "var(--spacing-md)",
        padding: "var(--spacing-md)",
        backgroundColor: "var(--primary-bg-light)",
        borderRadius: "var(--border-radius)",
        border: "1px solid var(--primary-border-light)",
      }}
    >
      <label
        style={{
          display: "flex",
          alignItems: "flex-start",
          gap: "var(--spacing-sm)",
          fontSize: "14px",
          cursor: "pointer",
          lineHeight: "1.5",
        }}
      >
        <input
          type="checkbox"
          checked={storeResults}
          onChange={(e) => onStoreResultsChange(e.target.checked)}
          style={{ cursor: "pointer", marginTop: "2px" }}
        />
        <span>
          <strong>Save results on server (optional)</strong>
          <br />
          <span style={{ fontSize: "13px", color: "var(--text-secondary)" }}>
            By default, your results are only saved in your browser. Check this box to enable server
            storage so you can access your results from any device. Your data will be stored for 90
            days.
          </span>
        </span>
      </label>
    </div>
  );
}
