import { randomUUID } from "crypto";

const DEFAULT_LOCAL_API_BASE = process.env.LOCAL_API_BASE || "http://localhost:8787";
const forceRemote = process.env.API_BASE_FORCE_REMOTE === "true";
const shouldPreferLocal = !forceRemote && process.env.API_BASE_PREFER_LOCAL !== "false";

export const API_BASE =
  process.env.API_BASE_OVERRIDE ||
  (shouldPreferLocal ? DEFAULT_LOCAL_API_BASE : undefined) ||
  process.env.API_BASE ||
  process.env.API_BASE_URL ||
  DEFAULT_LOCAL_API_BASE;
// Always prefer TEST_API_KEY for tests (higher rate limits)
export const API_KEY = process.env.TEST_API_KEY || process.env.API_KEY || "";

if (!API_KEY) {
  throw new Error("TEST_API_KEY or API_KEY environment variable required");
}

export async function apiRequest(method: string, path: string, body?: object) {
  const response = await fetch(`${API_BASE}${path}`, {
    method,
    headers: {
      Authorization: `Token ${API_KEY}`,
      "Content-Type": "application/json",
      // Note: Rate limiting is enforced for all requests, including tests
    },
    body: body ? JSON.stringify(body) : undefined,
  });

  let json;
  const text = await response.text();
  try {
    json = text ? JSON.parse(text) : {};
  } catch (e) {
    json = { error: text };
  }

  // Log errors for debugging
  if (response.status >= 500) {
    console.error(
      `[API Error ${response.status}] ${method} ${path}:`,
      JSON.stringify(json, null, 2),
    );
  }

  return { status: response.status, json };
}

// Poll for results (GET requests only - PUT returns results immediately)
export async function pollResults(submissionId: string, maxAttempts = 30) {
  for (let i = 0; i < maxAttempts; i++) {
    const { status, json } = await apiRequest("GET", `/text/submissions/${submissionId}`);
    if (status === 200 && json.status === "success") {
      return json;
    }
    await new Promise((resolve) => setTimeout(resolve, 2000));
  }
  throw new Error("Results not ready");
}

export function generateIds() {
  return { questionId: randomUUID(), answerId: randomUUID(), submissionId: randomUUID() };
}

// Note: Cleanup endpoint has been removed for security reasons.
// Tests should use unique IDs to avoid conflicts and be independent.
