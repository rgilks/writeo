/**
 * Error logging utility
 * Can be extended to integrate with error tracking services like Sentry, LogRocket, etc.
 */

interface ErrorContext {
  userId?: string;
  submissionId?: string;
  page?: string;
  action?: string;
  [key: string]: unknown;
}

class ErrorLogger {
  private enabled: boolean;
  private environment: string;

  constructor() {
    this.enabled = typeof window !== "undefined";
    this.environment =
      typeof process !== "undefined" ? process.env.NODE_ENV || "development" : "development";
  }

  /**
   * Log an error with context
   */
  logError(error: Error | unknown, context?: ErrorContext): void {
    if (!this.enabled) return;

    const errorObj = error instanceof Error ? error : new Error(String(error));
    const errorInfo = {
      message: errorObj.message,
      stack: errorObj.stack,
      name: errorObj.name,
      context: {
        ...context,
        environment: this.environment,
        timestamp: new Date().toISOString(),
        userAgent: typeof navigator !== "undefined" ? navigator.userAgent : undefined,
        url: typeof window !== "undefined" ? window.location.href : undefined,
      },
    };

    // Log to console in development
    if (this.environment === "development") {
      console.error("Error logged:", errorInfo);
    }

    // TODO: Integrate with error tracking service
    // Example: Sentry.captureException(errorObj, { extra: context });
    // Example: LogRocket.captureException(errorObj, { extra: context });

    // Store errors in localStorage for debugging (limit to last 10)
    try {
      const storedErrors = this.getStoredErrors();
      storedErrors.unshift(errorInfo);
      if (storedErrors.length > 10) {
        storedErrors.pop();
      }
      localStorage.setItem("writeo_errors", JSON.stringify(storedErrors));
    } catch {
      // Ignore localStorage errors
    }
  }

  /**
   * Log a warning
   */
  logWarning(message: string, context?: ErrorContext): void {
    if (!this.enabled) return;

    const warningInfo = {
      message,
      context: {
        ...context,
        environment: this.environment,
        timestamp: new Date().toISOString(),
      },
    };

    if (this.environment === "development") {
      console.warn("Warning logged:", warningInfo);
    }

    // TODO: Integrate with error tracking service
    // Example: Sentry.captureMessage(message, { level: "warning", extra: context });
  }

  /**
   * Get stored errors (for debugging)
   */
  getStoredErrors(): Array<{ message: string; context: ErrorContext }> {
    try {
      const stored = localStorage.getItem("writeo_errors");
      return stored ? JSON.parse(stored) : [];
    } catch {
      return [];
    }
  }

  /**
   * Clear stored errors
   */
  clearStoredErrors(): void {
    try {
      localStorage.removeItem("writeo_errors");
    } catch {
      // Ignore
    }
  }

  /**
   * Initialize error tracking (call this in app initialization)
   */
  init(): void {
    if (!this.enabled) return;

    // Global error handler
    window.addEventListener("error", (event) => {
      this.logError(event.error || new Error(event.message), {
        page: window.location.pathname,
        filename: event.filename,
        lineno: event.lineno,
        colno: event.colno,
      });
    });

    // Unhandled promise rejection handler
    window.addEventListener("unhandledrejection", (event) => {
      this.logError(event.reason, {
        page: window.location.pathname,
        type: "unhandled_promise_rejection",
      });
    });
  }
}

// Singleton instance
export const errorLogger = new ErrorLogger();

// Initialize on module load (client-side only)
if (typeof window !== "undefined") {
  errorLogger.init();
}
