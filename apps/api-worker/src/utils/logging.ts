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

function sanitizeValue(value: any): any {
  if (value === null || value === undefined) {
    return value;
  }

  if (typeof value === "string") {
    for (const pattern of SENSITIVE_PATTERNS) {
      if (pattern.test(value)) {
        return value.replace(/./g, "*").substring(0, Math.min(value.length, 20)) + "...[REDACTED]";
      }
    }
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
      const isSensitiveKey = SENSITIVE_PATTERNS.some((pattern) => pattern.test(key));
      sanitized[key] = isSensitiveKey ? "[REDACTED]" : sanitizeValue(val);
    }
    return sanitized;
  }

  return value;
}

export function safeLogError(message: string, data?: any): void {
  const sanitizedData = data ? sanitizeValue(data) : undefined;
  console.error(message, sanitizedData);
}

export function safeLogWarn(message: string, data?: any): void {
  const sanitizedData = data ? sanitizeValue(data) : undefined;
  console.warn(message, sanitizedData);
}

export function safeLogInfo(message: string, data?: any): void {
  const sanitizedData = data ? sanitizeValue(data) : undefined;
  console.log(message, sanitizedData);
}

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
