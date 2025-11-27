/**
 * Health endpoint paths
 */

import { noAuth } from "../utils";

export const healthPaths = {
  "/health": {
    get: {
      tags: ["Health"],
      summary: "Health check",
      description: "Returns the health status of the API. No authentication required.",
      operationId: "healthCheck",
      security: noAuth,
      responses: {
        "200": {
          description: "Service is healthy",
          content: {
            "application/json": {
              schema: {
                type: "object" as const,
                properties: {
                  status: {
                    type: "string" as const,
                    example: "ok",
                  },
                },
              },
            },
          },
        },
      },
    },
  },
  "/docs": {
    get: {
      tags: ["Health"],
      summary: "API Documentation",
      description: "Swagger UI documentation for the API. No authentication required.",
      operationId: "getDocs",
      security: noAuth,
      responses: {
        "200": {
          description: "Swagger UI HTML page",
        },
      },
    },
  },
} as const;
