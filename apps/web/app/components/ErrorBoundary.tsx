"use client";

import React from "react";
import { useRouter } from "next/navigation";

interface ErrorBoundaryProps {
  children: React.ReactNode;
  fallback?: React.ReactNode;
}

interface ErrorBoundaryState {
  error: Error | null;
}

interface ErrorBoundaryInnerProps extends ErrorBoundaryProps {
  router: ReturnType<typeof useRouter>;
}

interface DefaultFallbackProps {
  error: Error | null;
  onReset: () => void;
}

function DefaultFallback({ error, onReset }: DefaultFallbackProps) {
  return (
    <div
      className="card"
      style={{
        padding: "var(--spacing-lg)",
        textAlign: "center",
      }}
      lang="en"
    >
      <h2 style={{ marginBottom: "var(--spacing-md)", color: "var(--error-color)" }}>
        Something went wrong
      </h2>
      <p style={{ marginBottom: "var(--spacing-md)", color: "var(--text-secondary)" }}>
        {error?.message || "An unexpected error occurred"}
      </p>
      <button onClick={onReset} className="btn btn-primary">
        Try again
      </button>
    </div>
  );
}

class ErrorBoundaryInner extends React.Component<ErrorBoundaryInnerProps, ErrorBoundaryState> {
  constructor(props: ErrorBoundaryInnerProps) {
    super(props);
    this.state = { error: null };
  }

  static getDerivedStateFromError(error: Error): ErrorBoundaryState {
    return { error };
  }

  componentDidCatch(error: Error, errorInfo: React.ErrorInfo) {
    console.error("ErrorBoundary caught an error:", error, errorInfo);
  }

  handleReset = () => {
    this.setState({ error: null });
    this.props.router.refresh();
  };

  render() {
    if (this.state.error) {
      return (
        this.props.fallback || (
          <DefaultFallback error={this.state.error} onReset={this.handleReset} />
        )
      );
    }

    return this.props.children;
  }
}

/**
 * ErrorBoundary - Catches React errors and displays a fallback UI
 * Wrapper component to inject Next.js router into the class component
 */
function ErrorBoundaryWithRouter(props: ErrorBoundaryProps) {
  const router = useRouter();
  return <ErrorBoundaryInner {...props} router={router} />;
}

export const ErrorBoundary = ErrorBoundaryWithRouter;
