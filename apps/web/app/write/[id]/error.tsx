"use client";

import { useEffect } from "react";
import Link from "next/link";
import { getErrorMessage } from "@/app/lib/utils/error-messages";

const isDevelopment = typeof process !== "undefined" && process.env?.NODE_ENV === "development";

export default function Error({
  error,
  reset,
}: {
  error: Error & { digest?: string };
  reset: () => void;
}) {
  useEffect(() => {
    console.error("Error:", error);
  }, [error]);

  const errorMessage = getErrorMessage(error, "write");

  return (
    <>
      <header className="header">
        <div className="header-content">
          <div className="logo-group">
            <Link href="/" className="logo">
              Writeo
            </Link>
          </div>
          <nav className="header-actions" aria-label="Error actions">
            <Link href="/" className="nav-back-link">
              <span aria-hidden="true">←</span> Back to Home
            </Link>
          </nav>
        </div>
      </header>

      <div className="container">
        <div
          className="card"
          style={{ maxWidth: "600px", margin: "var(--spacing-3xl) auto", textAlign: "center" }}
        >
          <div style={{ fontSize: "48px", marginBottom: "var(--spacing-md)" }}>⚠️</div>
          <h1 style={{ fontSize: "24px", fontWeight: 600, marginBottom: "var(--spacing-md)" }}>
            Unable to Submit Essay
          </h1>

          <p
            style={{
              color: "var(--text-secondary)",
              marginBottom: "var(--spacing-lg)",
              lineHeight: "1.6",
            }}
          >
            {errorMessage}
          </p>

          {isDevelopment && error.digest && (
            <details
              style={{
                marginBottom: "var(--spacing-lg)",
                padding: "var(--spacing-md)",
                backgroundColor: "var(--bg-tertiary)",
                borderRadius: "var(--border-radius)",
                fontSize: "14px",
                textAlign: "left",
              }}
            >
              <summary style={{ cursor: "pointer", fontWeight: 500 }}>
                Technical Details (Development Only)
              </summary>
              <pre
                style={{
                  marginTop: "var(--spacing-sm)",
                  whiteSpace: "pre-wrap",
                  wordBreak: "break-word",
                  fontSize: "12px",
                  color: "var(--text-secondary)",
                }}
              >
                {error.message || "No error message available"}
                {error.digest && `\n\nDigest: ${error.digest}`}
                {error.stack && `\n\nStack:\n${error.stack.substring(0, 500)}`}
              </pre>
            </details>
          )}

          <div
            style={{
              display: "flex",
              gap: "var(--spacing-md)",
              justifyContent: "center",
              flexWrap: "wrap",
            }}
          >
            <button onClick={reset} className="btn btn-primary">
              Try Again
            </button>
            <Link href="/" className="btn btn-secondary">
              Back to Home
            </Link>
          </div>
        </div>
      </div>
    </>
  );
}
