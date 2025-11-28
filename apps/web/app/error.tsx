"use client";

import { useEffect } from "react";
import Link from "next/link";
import { getErrorMessage } from "@/app/lib/utils/error-messages";
import { errorLogger } from "@/app/lib/utils/error-logger";

const isDevelopment = typeof process !== "undefined" && process.env?.NODE_ENV === "development";

export default function GlobalError({
  error,
  reset,
}: {
  error: Error & { digest?: string };
  reset: () => void;
}) {
  useEffect(() => {
    errorLogger.logError(error, {
      page: typeof window !== "undefined" ? window.location.pathname : "unknown",
      action: "global_error",
      digest: error.digest,
    });
  }, [error]);

  const errorMessage = getErrorMessage(error, "global");

  return (
    <html>
      <head>
        <style>{`
          :root {
            --bg-primary: #ffffff;
            --bg-secondary: #f9fafb;
            --bg-tertiary: #f3f4f6;
            --text-primary: #111827;
            --text-secondary: #6b7280;
            --primary-color: #3b82f6;
            --border-radius: 8px;
            --spacing-sm: 8px;
            --spacing-md: 16px;
            --spacing-lg: 24px;
            --spacing-xl: 32px;
            --spacing-3xl: 48px;
            --shadow-md: 0 4px 6px -1px rgb(0 0 0 / 0.1), 0 2px 4px -1px rgb(0 0 0 / 0.06);
          }
          body {
            margin: 0;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", sans-serif;
          }
          .btn {
            display: inline-block;
            padding: var(--spacing-sm) var(--spacing-lg);
            min-height: 44px;
            border-radius: var(--border-radius);
            font-size: 16px;
            font-weight: 600;
            text-align: center;
            text-decoration: none;
            cursor: pointer;
            border: none;
            transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
            line-height: 1.5;
          }
          .btn-primary {
            background: var(--primary-color);
            color: white;
          }
          .btn-primary:hover {
            background: #2563eb;
          }
          .btn-secondary {
            background-color: var(--bg-tertiary);
            color: var(--text-primary);
            border: 1px solid #e5e7eb;
          }
          .btn-secondary:hover {
            background-color: var(--bg-secondary);
          }
        `}</style>
      </head>
      <body>
        <div
          style={{
            minHeight: "100vh",
            display: "flex",
            flexDirection: "column",
            alignItems: "center",
            justifyContent: "center",
            padding: "var(--spacing-md)",
            backgroundColor: "var(--bg-secondary)",
          }}
        >
          <div
            style={{
              maxWidth: "600px",
              width: "100%",
              backgroundColor: "var(--bg-primary)",
              borderRadius: "var(--border-radius)",
              padding: "var(--spacing-xl)",
              boxShadow: "var(--shadow-md)",
              textAlign: "center",
            }}
          >
            <div style={{ fontSize: "48px", marginBottom: "var(--spacing-md)" }}>✨</div>
            <h1
              style={{
                fontSize: "24px",
                fontWeight: 600,
                marginBottom: "var(--spacing-md)",
                color: "var(--text-primary)",
              }}
            >
              Oops! Something went wrong
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
      </body>
    </html>
  );
}
