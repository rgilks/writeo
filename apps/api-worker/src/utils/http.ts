import { fetchWithTimeout } from "./fetch-with-timeout";

const AUTH_TOKEN_PREFIX = "Token";

/**
 * Sends a POST request with JSON body and Token-based authentication.
 *
 * @param url - The URL to send the request to
 * @param apiKey - API key for authentication
 * @param body - Request body (will be JSON stringified)
 * @param timeout - Request timeout in milliseconds
 * @returns Promise that resolves to Response
 *
 * @example
 * ```typescript
 * const response = await postJsonWithAuth(
 *   "https://api.example.com/endpoint",
 *   "your-api-key",
 *   { data: "value" },
 *   30000,
 * );
 * ```
 */
export function postJsonWithAuth(
  url: string,
  apiKey: string,
  body: unknown,
  timeout: number,
): Promise<Response> {
  return fetchWithTimeout(url, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `${AUTH_TOKEN_PREFIX} ${apiKey}`,
    },
    body: JSON.stringify(body),
    timeout,
  });
}
