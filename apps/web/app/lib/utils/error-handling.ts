/**
 * Error handling utilities for Next.js Server Actions
 */

export async function getErrorMessage(response: Response): Promise<string> {
  if (response.status === 429) {
    return "Too many requests. Please wait a moment and try again.";
  }
  if (response.status >= 500) {
    return "Server error. Please try again in a moment.";
  }
  if (response.status === 404) {
    return "Resource not found. Please check and try again.";
  }
  if (response.status === 401 || response.status === 403) {
    return "Authentication error. Please refresh the page and try again.";
  }

  try {
    const text = await response.text();
    if (!text) {
      return `HTTP ${response.status}`;
    }

    let errorText: string | null = null;
    try {
      const errorData = JSON.parse(text);
      errorText = errorData?.error || errorData?.message || null;
    } catch {
      errorText = text;
    }

    if (errorText) {
      if (errorText.includes("network") || errorText.includes("fetch")) {
        return "Unable to connect to the server. Please check your internet connection and try again.";
      }
      if (errorText.includes("timeout")) {
        return "The request took too long. Please try again.";
      }
      return errorText;
    }
  } catch {
    // Fall through to default
  }

  return `HTTP ${response.status}`;
}

export function makeSerializableError(error: unknown): Error {
  if (error instanceof Error) {
    return new Error(error.message || "An unexpected error occurred");
  }
  if (typeof error === "string") {
    return new Error(error);
  }

  try {
    const errorStr = JSON.stringify(error);
    if (errorStr && errorStr !== "{}") {
      return new Error(`Error: ${errorStr.substring(0, 200)}`);
    }
  } catch {
    // Fall through to default
  }

  return new Error("An unexpected error occurred");
}
