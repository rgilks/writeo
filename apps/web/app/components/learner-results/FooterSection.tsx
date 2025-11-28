/**
 * Footer section component
 */

import Link from "next/link";

export function FooterSection() {
  const containerStyle = {
    marginTop: "var(--spacing-lg)",
    display: "flex",
    gap: "var(--spacing-md)",
    flexWrap: "wrap",
    alignItems: "baseline",
  } as const;

  const noticeStyle = {
    fontSize: "14px",
    color: "var(--text-secondary)",
    fontStyle: "italic",
    marginTop: "var(--spacing-sm)",
    lineHeight: "1.5",
  } as const;

  const privacyLinkStyle = {
    color: "var(--primary-color)",
    textDecoration: "underline",
  } as const;

  return (
    <div style={containerStyle}>
      <Link href="/" className="btn btn-primary" lang="en">
        ← Back to Tasks
      </Link>
      <p style={noticeStyle} lang="en">
        Your text is processed by an AI model; no one else reads it.{" "}
        <Link href="/privacy" style={privacyLinkStyle}>
          See our privacy policy
        </Link>
        .
      </p>
    </div>
  );
}
