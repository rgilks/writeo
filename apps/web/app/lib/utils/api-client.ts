/**
 * API client utilities
 */

import { getApiBase, getApiKey } from "../api-config";

export async function apiRequest(endpoint: string, method: string, body: any): Promise<Response> {
  const apiBase = getApiBase();
  const apiKey = getApiKey();

  if (apiKey === "MISSING_API_KEY" || apiBase === "MISSING_API_BASE_URL") {
    throw new Error("Server configuration error: API credentials not set");
  }

  // No retries - just make the request directly
  const response = await fetch(`${apiBase}${endpoint}`, {
    method,
    headers: {
      "Content-Type": "application/json",
      Authorization: `Token ${apiKey}`,
    },
    body: JSON.stringify(body),
  });

  return response;
}
