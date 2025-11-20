export function validateText(
  text: string,
  maxLength: number = 50000
): { valid: boolean; error?: string } {
  if (!text || typeof text !== "string") {
    return { valid: false, error: "Text must be a non-empty string" };
  }

  let normalizedText: string;
  try {
    normalizedText = text.normalize("NFKC");
  } catch (error) {
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

  if (/(.)\1{100,}/.test(normalizedText)) {
    return {
      valid: false,
      error: "Text contains suspicious character patterns",
    };
  }

  const dangerousPatterns = [
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
    // Unicode-based XSS attempts
    /\u003cscript/gi, // <script in Unicode
    /\u003c\/script/gi, // </script in Unicode
    // Control characters (except common whitespace)
    /[\x00-\x08\x0B-\x0C\x0E-\x1F\x7F-\x9F]/g,
  ];

  for (const pattern of dangerousPatterns) {
    if (pattern.test(normalizedText)) {
      return {
        valid: false,
        error: "Text contains potentially unsafe content",
      };
    }
  }

  // Check for suspicious repeated patterns (potential attack patterns)
  if (/([<>\"'&])\1{3,}/.test(normalizedText)) {
    return {
      valid: false,
      error: "Text contains suspicious character patterns",
    };
  }

  // Check for excessive nesting of brackets/parentheses (potential DoS)
  const bracketDepth = Math.max(
    (normalizedText.match(/\(/g) || []).length,
    (normalizedText.match(/\[/g) || []).length,
    (normalizedText.match(/\{/g) || []).length
  );
  if (bracketDepth > 100) {
    return {
      valid: false,
      error: "Text contains excessive nesting",
    };
  }

  return { valid: true };
}

/**
 * Validates request body size and JSON depth to prevent DoS attacks
 */
export async function validateRequestBodySize(
  request: Request,
  maxSizeBytes: number = 1024 * 1024,
  maxJsonDepth: number = 10
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

  // Check JSON depth if Content-Type indicates JSON
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
    } catch (error) {
      // If we can't parse JSON depth, that's okay - JSON parsing will catch it later
      // This is just a pre-check to prevent DoS
    }
  }

  return {
    valid: true,
    size: contentLength ? parseInt(contentLength, 10) : undefined,
  };
}

/**
 * Calculates the maximum nesting depth of a JSON string
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
        // Mismatched brackets - return high depth to trigger error
        return 1000;
      }
    }
  }

  return maxDepth;
}

export function sanitizeText(text: string): string {
  if (!text || typeof text !== "string") {
    return "";
  }

  let sanitized = text.replace(/[\x00-\x08\x0B-\x0C\x0E-\x1F\x7F]/g, "");
  sanitized = sanitized.replace(/<script[\s\S]*?>[\s\S]*?<\/script>/gi, "");
  sanitized = sanitized.replace(/\s*on\w+\s*=\s*["'][^"']*["']/gi, "");
  sanitized = sanitized.replace(/\s*on\w+\s*=\s*[^\s>]*/gi, "");
  sanitized = sanitized.replace(/javascript\s*:/gi, "");

  return sanitized.trim();
}
