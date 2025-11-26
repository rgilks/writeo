"use client";

import React from "react";
import { useRouter } from "next/navigation";

interface ErrorBoundaryProps {
  children: React.ReactNode;
  fallback?: React.ReactNode;
}

interface ErrorBoundaryState {
  hasError: boolean;
  error: Error | null;
}

// Wrapper component to provide router to class component
function ErrorBoundaryWithRouter(props: ErrorBoundaryProps) {
  const router = useRouter();
  return <ErrorBoundaryInner {...props} router={router} />;
}

interface ErrorBoundaryInnerProps extends ErrorBoundaryProps {
  router: ReturnType<typeof useRouter>;
}

class ErrorBoundaryInner extends React.Component<ErrorBoundaryInnerProps, ErrorBoundaryState> {
  constructor(props: ErrorBoundaryInnerProps) {
    super(props);
    this.state = { hasError: false, error: null };
  }

  static getDerivedStateFromError(error: Error): ErrorBoundaryState {
    return { hasError: true, error };
  }

  componentDidCatch(error: Error, errorInfo: React.ErrorInfo) {
    console.error("ErrorBoundary caught an error:", error, errorInfo);
  }

  handleReset = () => {
    this.setState({ hasError: false, error: null });
    // Use Next.js router to refresh instead of window.location.reload()
    this.props.router.refresh();
  };

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
            <button onClick={this.handleReset} className="btn btn-primary" lang="en">
              Try again
            </button>
          </div>
        )
      );
    }

    return this.props.children;
  }
}

export const ErrorBoundary = ErrorBoundaryWithRouter;
