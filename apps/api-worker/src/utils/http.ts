import { fetchWithTimeout } from "./fetch-with-timeout";

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
      Authorization: `Token ${apiKey}`,
    },
    body: JSON.stringify(body),
    timeout,
  });
}
