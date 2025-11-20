import type { Context } from "hono";
import type { Env } from "../types/env";

export function errorResponse(
  status: number,
  message: string,
  c?: Context<{ Bindings: Env }> | Context<any>
): Response {
  const isProduction = c
    ? !c.req.url.includes("localhost") && !c.req.url.includes("127.0.0.1")
    : true;
  const safeMessage =
    status >= 500 && isProduction ? "An internal error occurred. Please try again later." : message;

  return new Response(JSON.stringify({ error: safeMessage }), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}
