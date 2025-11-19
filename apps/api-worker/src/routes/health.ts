import { Hono } from "hono";
import { swaggerUI } from "@hono/swagger-ui";
import { openApiSpec } from "../openapi";

export const healthRouter = new Hono();

healthRouter.get("/docs", swaggerUI({ url: "/openapi.json" }));

healthRouter.get("/openapi.json", (c) => {
  // Dynamically set server URL from request
  const url = new URL(c.req.url);
  const baseUrl = `${url.protocol}//${url.host}`;

  const spec = {
    ...openApiSpec,
    servers: [
      {
        url: baseUrl,
        description: "Current server",
      },
      {
        url: "http://localhost:8787",
        description: "Local development server",
      },
    ],
  };

  return c.json(spec);
});

healthRouter.get("/health", (c) => {
  return c.json({ status: "ok" });
});
