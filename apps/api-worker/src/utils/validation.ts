const DEFAULT_MAX_TEXT_LENGTH = 50000;
const MAX_REPEATED_CHAR_COUNT = 100;
const MAX_BRACKET_DEPTH = 100;
const MIN_SUSPICIOUS_REPEAT = 3;
const MISMATCHED_BRACKET_DEPTH = 1000;

const REPEATED_CHAR_PATTERN = new RegExp(`(.)\\1{${MAX_REPEATED_CHAR_COUNT},}`);
const SUSPICIOUS_REPEAT_PATTERN = new RegExp(`([<>"'&])\\1{${MIN_SUSPICIOUS_REPEAT},}`);
const CONTROL_CHARS_PATTERN = /[\x00-\x08\x0B-\x0C\x0E-\x1F\x7F-\x9F]/g;
const CONTROL_CHARS_SANITIZE_PATTERN = /[\x00-\x08\x0B-\x0C\x0E-\x1F\x7F]/g;

const DANGEROUS_PATTERNS = [
  /<script[\s\S]*?>[\s\S]*?<\/script>/gi,
  /<script[\s\S]*?>/gi,
  /<\/script>/gi,
  /on\w+\s*=/gi,
  /javascript:/gi,
  /javascript\s*:/gi,
  /data:\s*text\/html/gi,
  /data:\s*text\/javascript/gi,
  /<iframe[\s\S]*?>/gi,
  /<\/iframe>/gi,
  /<object[\s\S]*?>/gi,
  /<embed[\s\S]*?>/gi,
  /<style[\s\S]*?expression[\s\S]*?>/gi,
  /<svg[\s\S]*?<script/gi,
  /vbscript:/gi,
  /<img[\s\S]*?src[\s\S]*?=[\s\S]*?javascript:/gi,
  /<body[\s\S]*?onload/gi,
  /<input[\s\S]*?onfocus/gi,
  /\u003cscript/gi, // <script in Unicode
  /\u003c\/script/gi, // </script in Unicode
  CONTROL_CHARS_PATTERN,
] as const;

const SANITIZE_PATTERNS = [
  [/<script[\s\S]*?>[\s\S]*?<\/script>/gi, ""],
  [/\s*on\w+\s*=\s*["'][^"']*["']/gi, ""],
  [/\s*on\w+\s*=\s*[^\s>]*/gi, ""],
  [/javascript\s*:/gi, ""],
] as const;

/**
 * Counts opening brackets of a given type in text.
 */
function countBrackets(text: string, bracket: string): number {
  return (text.match(new RegExp(`\\${bracket}`, "g")) || []).length;
}

/**
 * Validates text for security issues, length, and suspicious patterns.
 *
 * @param text - The text to validate
 * @param maxLength - Maximum allowed text length (default: 50000)
 * @returns Validation result with error message if invalid
 */
export function validateText(
  text: string,
  maxLength: number = DEFAULT_MAX_TEXT_LENGTH,
): { valid: boolean; error?: string } {
  if (!text || typeof text !== "string") {
    return { valid: false, error: "Text must be a non-empty string" };
  }

  let normalizedText: string;
  try {
    normalizedText = text.normalize("NFKC");
  } catch {
    return { valid: false, error: "Text contains invalid Unicode characters" };
  }

  if (normalizedText.trim().length === 0) {
    return { valid: false, error: "Text cannot be empty or whitespace only" };
  }

  if (normalizedText.length > maxLength) {
    return {
      valid: false,
      error: `Text exceeds maximum length of ${maxLength} characters`,
    };
  }

  if (REPEATED_CHAR_PATTERN.test(normalizedText)) {
    return {
      valid: false,
      error: "Text contains suspicious character patterns",
    };
  }

  for (const pattern of DANGEROUS_PATTERNS) {
    if (pattern.test(normalizedText)) {
      return {
        valid: false,
        error: "Text contains potentially unsafe content",
      };
    }
  }

  if (SUSPICIOUS_REPEAT_PATTERN.test(normalizedText)) {
    return {
      valid: false,
      error: "Text contains suspicious character patterns",
    };
  }

  const bracketDepth = Math.max(
    countBrackets(normalizedText, "("),
    countBrackets(normalizedText, "["),
    countBrackets(normalizedText, "{"),
  );
  if (bracketDepth > MAX_BRACKET_DEPTH) {
    return {
      valid: false,
      error: "Text contains excessive nesting",
    };
  }

  return { valid: true };
}

/**
 * Validates request body size and JSON depth to prevent DoS attacks.
 *
 * @param request - The request to validate
 * @param maxSizeBytes - Maximum allowed body size in bytes (default: 1MB)
 * @param maxJsonDepth - Maximum allowed JSON nesting depth (default: 10)
 * @returns Validation result with error message and size if invalid
 */
export async function validateRequestBodySize(
  request: Request,
  maxSizeBytes: number = 1024 * 1024,
  maxJsonDepth: number = 10,
): Promise<{ valid: boolean; error?: string; size?: number }> {
  const contentLength = request.headers.get("content-length");

  if (contentLength) {
    const size = parseInt(contentLength, 10);
    if (isNaN(size) || size < 0) {
      return { valid: false, error: "Invalid content-length header" };
    }
    if (size > maxSizeBytes) {
      return {
        valid: false,
        error: `Request body too large (max ${maxSizeBytes / 1024 / 1024}MB)`,
        size,
      };
    }
  }

  const contentType = request.headers.get("content-type");
  if (contentType?.includes("application/json")) {
    try {
      // Clone request to read body without consuming it
      const clonedRequest = request.clone();
      const text = await clonedRequest.text();
      if (text) {
        const depth = calculateJsonDepth(text);
        if (depth > maxJsonDepth) {
          return {
            valid: false,
            error: `JSON structure too deeply nested (max depth: ${maxJsonDepth})`,
          };
        }
      }
    } catch {
      // Pre-check failed; JSON parsing will catch errors later
    }
  }

  return {
    valid: true,
    size: contentLength ? parseInt(contentLength, 10) : undefined,
  };
}

/**
 * Calculates the maximum nesting depth of a JSON string.
 * Returns a high value for mismatched brackets to trigger validation error.
 */
function calculateJsonDepth(jsonString: string): number {
  let maxDepth = 0;
  let currentDepth = 0;

  for (const char of jsonString) {
    if (char === "{" || char === "[" || char === "(") {
      currentDepth++;
      maxDepth = Math.max(maxDepth, currentDepth);
    } else if (char === "}" || char === "]" || char === ")") {
      currentDepth--;
      if (currentDepth < 0) {
        return MISMATCHED_BRACKET_DEPTH;
      }
    }
  }

  return maxDepth;
}

/**
 * Sanitizes text by removing dangerous patterns and control characters.
 * Less strict than validateText - removes rather than rejecting.
 *
 * @param text - The text to sanitize
 * @returns Sanitized text, or empty string if input is invalid
 */
export function sanitizeText(text: string): string {
  if (!text || typeof text !== "string") {
    return "";
  }

  let sanitized = text.replace(CONTROL_CHARS_SANITIZE_PATTERN, "");
  for (const [pattern, replacement] of SANITIZE_PATTERNS) {
    sanitized = sanitized.replace(pattern, replacement);
  }

  return sanitized.trim();
}
