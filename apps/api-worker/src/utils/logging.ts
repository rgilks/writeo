const SENSITIVE_PATTERNS = [
  /api[_-]?key/gi,
  /token/gi,
  /password/gi,
  /secret/gi,
  /credential/gi,
  /authorization/gi,
  /bearer\s+\w+/gi,
  /gsk_\w+/gi,
  /eyJ[\w-]+\.eyJ[\w-]+\.[\w-]+/gi,
];

const MAX_STRING_LENGTH = 500;
const TRUNCATE_TO_LENGTH = 100;
const REDACT_PREVIEW_LENGTH = 20;

const REDACTED_PLACEHOLDER = "[REDACTED]";
const TRUNCATED_SUFFIX = "...[TRUNCATED]";
const REDACTED_SUFFIX = "...[REDACTED]";

/**
 * Redacts a sensitive string value, showing only a short preview.
 */
function redactString(value: string): string {
  const previewLength = Math.min(value.length, REDACT_PREVIEW_LENGTH);
  return "*".repeat(previewLength) + REDACTED_SUFFIX;
}

/**
 * Truncates a long string value.
 */
function truncateString(value: string): string {
  return value.substring(0, TRUNCATE_TO_LENGTH) + TRUNCATED_SUFFIX;
}

/**
 * Recursively sanitizes values to remove sensitive data and truncate long strings.
 */
function sanitizeValue(value: unknown): unknown {
  if (value === null || value === undefined) {
    return value;
  }

  if (typeof value === "string") {
    for (const pattern of SENSITIVE_PATTERNS) {
      if (pattern.test(value)) {
        return redactString(value);
      }
    }
    if (value.length > MAX_STRING_LENGTH) {
      return truncateString(value);
    }
    return value;
  }

  if (typeof value === "object") {
    if (Array.isArray(value)) {
      return value.map(sanitizeValue);
    }

    const sanitized: Record<string, unknown> = {};
    for (const [key, val] of Object.entries(value)) {
      const isSensitiveKey = SENSITIVE_PATTERNS.some((pattern) => pattern.test(key));
      sanitized[key] = isSensitiveKey ? REDACTED_PLACEHOLDER : sanitizeValue(val);
    }
    return sanitized;
  }

  return value;
}

/**
 * Internal logging helper that sanitizes data before logging.
 */
function safeLog(level: "error" | "warn" | "info", message: string, data?: unknown): void {
  const sanitizedData = data ? sanitizeValue(data) : undefined;
  const logFn = level === "error" ? console.error : level === "warn" ? console.warn : console.log;
  logFn(message, sanitizedData);
}

/**
 * Logs an error with sanitized data.
 */
export function safeLogError(message: string, data?: unknown): void {
  safeLog("error", message, data);
}

/**
 * Logs a warning with sanitized data.
 */
export function safeLogWarn(message: string, data?: unknown): void {
  safeLog("warn", message, data);
}

/**
 * Logs an info message with sanitized data.
 */
export function safeLogInfo(message: string, data?: unknown): void {
  safeLog("info", message, data);
}

/**
 * Sanitizes an error object, extracting message and name while removing sensitive data.
 */
export function sanitizeError(error: unknown): {
  message: string;
  name?: string;
} {
  if (error instanceof Error) {
    const sanitizedMessage = sanitizeValue(error.message) as string;
    return {
      message: sanitizedMessage,
      name: error.name,
    };
  }
  return {
    message: sanitizeValue(String(error)) as string,
  };
}
