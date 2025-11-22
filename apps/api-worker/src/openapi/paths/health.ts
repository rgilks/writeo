/**
 * Health endpoint paths
 */

export const healthPaths = {
  "/health": {
    get: {
      tags: ["Health"],
      summary: "Health check",
      description: "Returns the health status of the API. No authentication required.",
      operationId: "healthCheck",
      security: [],
      responses: {
        "200": {
          description: "Service is healthy",
          content: {
            "application/json": {
              schema: {
                type: "object",
                properties: {
                  status: {
                    type: "string",
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
      security: [],
    },
  },
};
