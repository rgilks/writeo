const DEFAULT_TIMEOUT_MS = 30000;

export interface FetchWithTimeoutOptions extends RequestInit {
  /** Timeout in milliseconds (default: DEFAULT_TIMEOUT_MS) */
  timeout?: number;
}

/**
 * Fetches a resource with a configurable timeout to prevent hanging requests.
 *
 * @param url - The URL to fetch
 * @param options - Fetch options including optional timeout
 * @returns Promise that resolves to Response or rejects with timeout error
 *
 * @example
 * ```typescript
 * await fetchWithTimeout("https://api.example.com", {
 *   method: "POST",
 *   body: JSON.stringify(data),
 *   timeout: 10000,
 * });
 * ```
 */
export async function fetchWithTimeout(
  url: string,
  options: FetchWithTimeoutOptions = {},
): Promise<Response> {
  const { timeout = DEFAULT_TIMEOUT_MS, ...fetchOptions } = options;

  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), timeout);

  try {
    const response = await fetch(url, {
      ...fetchOptions,
      signal: controller.signal,
    });
    return response;
  } catch (error) {
    if (error instanceof Error && error.name === "AbortError") {
      throw new Error(`Request timeout after ${timeout}ms: ${url}`);
    }
    throw error;
  } finally {
    clearTimeout(timeoutId);
  }
}
