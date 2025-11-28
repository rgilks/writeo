export const DEFAULT_ERROR_MESSAGES = {
  global: "Something unexpected happened. Don't worry—this is usually temporary. Please try again.",
  write:
    "Something went wrong while submitting your essay. Please try again, or refresh the page if the problem persists.",
  results:
    "We couldn't load your results. This might happen if the submission ID is incorrect or the results have expired.",
} as const;

export function getErrorMessage(
  error: Error,
  context: keyof typeof DEFAULT_ERROR_MESSAGES = "global",
): string {
  const message = error?.message;
  if (typeof message !== "string" || message.length === 0) {
    return DEFAULT_ERROR_MESSAGES[context];
  }

  const lowerMessage = message.toLowerCase();

  // Handle Server Component errors
  if (
    message.includes("Server Components render") ||
    message.includes("omitted in production builds") ||
    message.includes("digest property") ||
    message.includes("Server Component") ||
    message.includes("Server Components")
  ) {
    if (context === "write") {
      return "We encountered an issue while processing your submission. Please try again, or if the problem persists, try refreshing the page.";
    }
    if (context === "global") {
      return "We encountered an issue while loading the page. Please try refreshing or navigating back to the homepage.";
    }
    return DEFAULT_ERROR_MESSAGES[context];
  }

  // Configuration errors
  if (message.includes("API_KEY") || message.includes("API_BASE_URL")) {
    if (context === "write") {
      return "There's a configuration issue. Please try again later or contact support if the problem persists.";
    }
    return "There's a configuration issue. Please try again later.";
  }

  // Network errors
  if (
    lowerMessage.includes("fetch") ||
    lowerMessage.includes("network") ||
    lowerMessage.includes("failed to fetch")
  ) {
    if (context === "write") {
      return "Unable to connect to our servers. Please check your internet connection and try again.";
    }
    return "We're having trouble connecting to our servers. Please check your internet connection and try again.";
  }

  // Timeout errors
  if (lowerMessage.includes("timeout") || lowerMessage.includes("timed out")) {
    return "The request took too long. Please try again.";
  }

  // Server configuration errors
  if (lowerMessage.includes("server configuration error")) {
    return "There's a server configuration issue. Please try again later.";
  }

  // Results-specific: Not found errors
  if (context === "results") {
    if (
      lowerMessage.includes("not found") ||
      lowerMessage.includes("404") ||
      lowerMessage.includes("does not exist")
    ) {
      return "We couldn't find the results you're looking for. This might happen if the submission ID is incorrect or the results have expired.";
    }
  }

  // Only return the message if it's user-friendly (not a stack trace or technical error)
  if (message.length < 200 && !message.includes("Error:") && !message.includes("at ")) {
    return message;
  }

  return DEFAULT_ERROR_MESSAGES[context];
}
