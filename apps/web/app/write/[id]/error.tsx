"use client";

import { useEffect } from "react";
import Link from "next/link";

function getErrorMessage(error: Error): string {
  const message = error?.message;
  if (typeof message === "string" && message.length > 0) {
    // Handle Next.js production Server Component errors
    if (
      message.includes("Server Components render") ||
      message.includes("omitted in production builds") ||
      message.includes("digest property")
    ) {
      return "We encountered an issue while processing your submission. Please try again, or if the problem persists, try refreshing the page.";
    }
    if (message.includes("API_KEY") || message.includes("API_BASE_URL")) {
      return "There's a configuration issue. Please try again later or contact support if the problem persists.";
    }
    if (
      message.includes("fetch") ||
      message.includes("network") ||
      message.includes("Failed to fetch")
    ) {
      return "Unable to connect to our servers. Please check your internet connection and try again.";
    }
    if (message.includes("timeout") || message.includes("timed out")) {
      return "The request took too long. Please try again.";
    }
    if (message.includes("Server configuration error")) {
      return "There's a server configuration issue. Please try again later.";
    }
    // For other errors, return the message if it's user-friendly, otherwise provide a generic one
    if (message.length > 100 || message.includes("Error:") || message.includes("at ")) {
      return "Something went wrong while submitting your essay. Please try again.";
    }
    return message;
  }
  return "Something unexpected happened. Please try again, or refresh the page if the problem persists.";
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
    <div
      style={{
        minHeight: "100vh",
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        justifyContent: "center",
        padding: "20px",
        backgroundColor: "#f9fafb",
      }}
    >
      <div
        style={{
          maxWidth: "600px",
          width: "100%",
          backgroundColor: "white",
          borderRadius: "8px",
          padding: "32px",
          boxShadow: "0 1px 3px rgba(0,0,0,0.1)",
        }}
      >
        <div style={{ fontSize: "48px", marginBottom: "16px", textAlign: "center" }}>⚠️</div>
        <h1
          style={{
            fontSize: "24px",
            fontWeight: 600,
            marginBottom: "16px",
            color: "#1f2937",
            textAlign: "center",
          }}
        >
          Unable to Submit Essay
        </h1>

        <p
          style={{
            color: "#6b7280",
            marginBottom: "24px",
            lineHeight: "1.6",
            textAlign: "center",
          }}
        >
          {errorMessage}
        </p>

        {typeof process !== "undefined" &&
          process.env?.NODE_ENV === "development" &&
          error.digest && (
            <details
              style={{
                marginBottom: "24px",
                padding: "12px",
                backgroundColor: "#f3f4f6",
                borderRadius: "4px",
                fontSize: "14px",
              }}
            >
              <summary style={{ cursor: "pointer", fontWeight: 500 }}>
                Technical Details (Development Only)
              </summary>
              <pre
                style={{
                  marginTop: "8px",
                  whiteSpace: "pre-wrap",
                  wordBreak: "break-word",
                  fontSize: "12px",
                  color: "#374151",
                }}
              >
                {typeof error?.message === "string" ? error.message : "No error message available"}
                {typeof error?.digest === "string" && `\n\nDigest: ${error.digest}`}
                {typeof error?.stack === "string" && `\n\nStack:\n${error.stack.substring(0, 500)}`}
              </pre>
            </details>
          )}

        <div style={{ display: "flex", gap: "12px", flexWrap: "wrap" }}>
          <button
            onClick={reset}
            style={{
              padding: "10px 20px",
              backgroundColor: "#6366f1",
              color: "white",
              border: "none",
              borderRadius: "6px",
              cursor: "pointer",
              fontWeight: 500,
              fontSize: "14px",
            }}
          >
            Try again
          </button>

          <Link
            href="/"
            style={{
              padding: "10px 20px",
              backgroundColor: "#f3f4f6",
              color: "#374151",
              border: "none",
              borderRadius: "6px",
              cursor: "pointer",
              fontWeight: 500,
              fontSize: "14px",
              textDecoration: "none",
              display: "inline-block",
            }}
          >
            Back to Tasks
          </Link>
        </div>
      </div>
    </div>
  );
}
