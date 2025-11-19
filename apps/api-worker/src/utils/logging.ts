/**
 * Safe logging utilities that sanitize sensitive data before logging
 */

// Patterns that indicate sensitive data
const SENSITIVE_PATTERNS = [
  /api[_-]?key/gi,
  /token/gi,
  /password/gi,
  /secret/gi,
  /credential/gi,
  /authorization/gi,
  /bearer\s+\w+/gi,
  /gsk_\w+/gi, // Groq API key pattern
  /eyJ[\w-]+\.eyJ[\w-]+\.[\w-]+/gi, // JWT token pattern
];

/**
 * Sanitizes a value by redacting sensitive information
 */
function sanitizeValue(value: any): any {
  if (value === null || value === undefined) {
    return value;
  }

  if (typeof value === "string") {
    // Check if the string contains sensitive patterns
    for (const pattern of SENSITIVE_PATTERNS) {
      if (pattern.test(value)) {
        // Redact the sensitive part
        return value.replace(/./g, "*").substring(0, Math.min(value.length, 20)) + "...[REDACTED]";
      }
    }
    // If it's a long string that might contain sensitive data, truncate it
    if (value.length > 500) {
      return value.substring(0, 100) + "...[TRUNCATED]";
    }
    return value;
  }

  if (typeof value === "object") {
    if (Array.isArray(value)) {
      return value.map(sanitizeValue);
    }

    const sanitized: Record<string, any> = {};
    for (const [key, val] of Object.entries(value)) {
      // Check if the key itself indicates sensitive data
      const isSensitiveKey = SENSITIVE_PATTERNS.some((pattern) => pattern.test(key));
      if (isSensitiveKey) {
        sanitized[key] = "[REDACTED]";
      } else {
        sanitized[key] = sanitizeValue(val);
      }
    }
    return sanitized;
  }

  return value;
}

/**
 * Safely logs an error message with sanitized data
 */
export function safeLogError(message: string, data?: any): void {
  const sanitizedData = data ? sanitizeValue(data) : undefined;
  console.error(message, sanitizedData);
}

/**
 * Safely logs a warning message with sanitized data
 */
export function safeLogWarn(message: string, data?: any): void {
  const sanitizedData = data ? sanitizeValue(data) : undefined;
  console.warn(message, sanitizedData);
}

/**
 * Safely logs an info message with sanitized data
 */
export function safeLogInfo(message: string, data?: any): void {
  const sanitizedData = data ? sanitizeValue(data) : undefined;
  console.log(message, sanitizedData);
}

/**
 * Sanitizes an error object for logging
 * Never includes stack traces to prevent information disclosure
 * Stack traces can leak file paths, internal structure, and other sensitive details
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
      // Never include stack traces - they can leak sensitive information:
      // - File paths and directory structure
      // - Internal function names
      // - Line numbers
      // - Module structure
      // Stack traces should only be viewed in local development via debugger
    };
  }
  return {
    message: sanitizeValue(String(error)) as string,
  };
}
