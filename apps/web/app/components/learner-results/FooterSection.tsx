/**
 * Footer section component
 */

import Link from "next/link";

export function FooterSection() {
  return (
    <div
      style={{
        marginTop: "var(--spacing-lg)",
        display: "flex",
        gap: "var(--spacing-md)",
        flexWrap: "wrap",
      }}
    >
      <Link href="/" className="btn btn-primary" lang="en">
        ← Back to Tasks
      </Link>
      <p
        style={{
          fontSize: "14px",
          color: "var(--text-secondary)",
          fontStyle: "italic",
          marginTop: "var(--spacing-sm)",
          lineHeight: "1.5",
        }}
        lang="en"
      >
        Your text is processed by an AI model; no one else reads it.{" "}
        <Link
          href="/privacy"
          style={{
            color: "var(--primary-color)",
            textDecoration: "underline",
          }}
        >
          See our privacy policy
        </Link>
        .
      </p>
    </div>
  );
}
