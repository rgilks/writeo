/**
 * Test helpers for api-worker tests
 */

import { vi } from "vitest";
import type { Context } from "hono";
import type { Env } from "../../apps/api-worker/src/types/env";

interface CreateContextOptions {
  path?: string;
  url?: string;
  method?: string;
  headers?: Record<string, string>;
  env?: Partial<Env>;
  body?: any;
}

export function createContext(options: CreateContextOptions = {}): Context<{ Bindings: Env }> {
  const {
    path = "/",
    url = `http://localhost:8787${path}`,
    method = "GET",
    headers = {},
    env = {},
    body,
  } = options;

  const request = new Request(url, {
    method,
    headers,
    body: body ? JSON.stringify(body) : undefined,
  });

  const contextData: Record<string, any> = {};

  const context = {
    req: {
      url,
      method,
      header: (name: string) => headers[name] || null,
      param: (name: string) => {
        // Simple param extraction for tests
        const match = path.match(new RegExp(`/:${name}/([^/]+)`));
        return match ? match[1] : "";
      },
      json: async () => body || {},
      raw: request,
    },
    env: {
      API_KEY: "test-admin-key",
      MODAL_DEBERTA_URL: "https://modal.example.com/deberta",
      WRITEO_DATA: {} as any,
      WRITEO_RESULTS: {} as any,
      OPENAI_API_KEY: "test-openai-key",
      ...env,
    } as Env,
    set: vi.fn((key: string, value: any) => {
      contextData[key] = value;
    }),
    get: vi.fn((key: string) => {
      return contextData[key];
    }),
    json: vi.fn((data: any, status?: number, headers?: Record<string, string>) => {
      return new Response(JSON.stringify(data), {
        status: status || 200,
        headers: { "Content-Type": "application/json", ...headers },
      });
    }),
    header: vi.fn(),
  } as any;

  return context;
}
