/**
 * API client utilities
 */

import { getApiBase, getApiKey } from "../api-config";

export interface ApiRequestOptions extends Omit<RequestInit, "body"> {
  endpoint: string;
  body?: unknown;
}

/**
 * Makes an authenticated API request
 * @param options - Request options including endpoint, method, body, and other fetch options
 * @returns Promise resolving to the Response
 */
export async function apiRequest(options: ApiRequestOptions): Promise<Response> {
  const { endpoint, body, method = "GET", ...fetchOptions } = options;
  const apiBase = getApiBase();
  const apiKey = getApiKey();

  if (apiKey === "MISSING_API_KEY" || apiBase === "MISSING_API_BASE_URL") {
    throw new Error("Server configuration error: API credentials not set");
  }

  const headers = new Headers(fetchOptions.headers);
  headers.set("Content-Type", "application/json");
  headers.set("Authorization", `Token ${apiKey}`);

  const response = await fetch(`${apiBase}${endpoint}`, {
    ...fetchOptions,
    method,
    headers,
    body: body !== undefined ? JSON.stringify(body) : undefined,
  });

  return response;
}
