"use client";

import React from "react";

interface ErrorBoundaryProps {
  children: React.ReactNode;
  fallback?: React.ReactNode;
}

interface ErrorBoundaryState {
  hasError: boolean;
  error: Error | null;
}

export class ErrorBoundary extends React.Component<ErrorBoundaryProps, ErrorBoundaryState> {
  constructor(props: ErrorBoundaryProps) {
    super(props);
    this.state = { hasError: false, error: null };
  }

  static getDerivedStateFromError(error: Error): ErrorBoundaryState {
    return { hasError: true, error };
  }

  componentDidCatch(error: Error, errorInfo: React.ErrorInfo) {
    console.error("ErrorBoundary caught an error:", error, errorInfo);
  }

  render() {
    if (this.state.hasError) {
      return (
        this.props.fallback || (
          <div
            className="card"
            style={{
              padding: "var(--spacing-lg)",
              textAlign: "center",
            }}
            lang="en"
          >
            <h2
              style={{ marginBottom: "var(--spacing-md)", color: "var(--error-color)" }}
              lang="en"
            >
              Something went wrong
            </h2>
            <p
              style={{ marginBottom: "var(--spacing-md)", color: "var(--text-secondary)" }}
              lang="en"
            >
              {this.state.error?.message || "An unexpected error occurred"}
            </p>
            <button
              onClick={() => {
                this.setState({ hasError: false, error: null });
                window.location.reload();
              }}
              className="btn btn-primary"
              lang="en"
            >
              Try again
            </button>
          </div>
        )
      );
    }

    return this.props.children;
  }
}
