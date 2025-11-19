"use client";

import { useEffect } from "react";
import Link from "next/link";

function getErrorMessage(error: Error): string {
  const message = error?.message;
  if (typeof message === "string" && message.length > 0) {
    // Check for common error patterns and provide friendly messages
    if (
      message.includes("not found") ||
      message.includes("404") ||
      message.includes("does not exist")
    ) {
      return "We couldn't find the results you're looking for. This might happen if the submission ID is incorrect or the results have expired.";
    }
    if (message.includes("API_KEY") || message.includes("API_BASE_URL")) {
      return "There's a configuration issue. Please try again later.";
    }
    if (
      message.includes("fetch") ||
      message.includes("network") ||
      message.includes("Failed to fetch")
    ) {
      return "We're having trouble connecting to our servers. Please check your internet connection and try again.";
    }
    if (message.includes("timeout") || message.includes("timed out")) {
      return "The request took too long. Please try again.";
    }
    // For Server Component errors, provide a friendly message
    if (message.includes("Server Component") || message.includes("Server Components")) {
      return "We encountered an issue while loading your results. This might happen if the submission ID is invalid or the results are no longer available.";
    }
    return message;
  }
  return "We couldn't load your results. This might happen if the submission ID is incorrect or the results have expired.";
}

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

  const errorMessage = getErrorMessage(error);

  return (
    <>
      <header className="header">
        <div className="header-content">
          <Link href="/" className="logo">
            Writeo
          </Link>
          <Link href="/" className="btn btn-secondary">
            ← Back to Tasks
          </Link>
        </div>
      </header>
      <div className="container">
        <div
          className="card"
          style={{
            maxWidth: "600px",
            margin: "48px auto",
            padding: "var(--spacing-xl)",
            textAlign: "center",
            backgroundColor: "var(--bg-primary)",
            border: "1px solid var(--border-color)",
          }}
        >
          <div
            style={{
              fontSize: "48px",
              marginBottom: "var(--spacing-md)",
            }}
          >
            📝
          </div>
          <h1
            style={{
              fontSize: "24px",
              fontWeight: 600,
              marginBottom: "var(--spacing-md)",
              color: "var(--text-primary)",
            }}
          >
            Results Not Available
          </h1>

          <p
            style={{
              color: "var(--text-secondary)",
              marginBottom: "var(--spacing-lg)",
              lineHeight: "1.6",
              fontSize: "16px",
            }}
          >
            {errorMessage.includes("Server Component") ||
            errorMessage.includes("omitted in production")
              ? "We couldn't load your results. This might happen if the submission ID is incorrect or the results are no longer available."
              : errorMessage}
          </p>

          {typeof process !== "undefined" &&
            process.env?.NODE_ENV === "development" &&
            error.digest && (
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
                  {typeof error?.message === "string"
                    ? error.message
                    : "No error message available"}
                  {typeof error?.digest === "string" && `\n\nDigest: ${error.digest}`}
                  {typeof error?.stack === "string" &&
                    `\n\nStack:\n${error.stack.substring(0, 500)}`}
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
            <Link href="/" className="btn btn-primary">
              Try Another Essay →
            </Link>
            <button onClick={reset} className="btn btn-secondary">
              Try Again
            </button>
          </div>
        </div>
      </div>
    </>
  );
}
