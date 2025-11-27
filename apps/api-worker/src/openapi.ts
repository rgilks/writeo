import { openApiMetadata } from "./openapi/metadata";
import { questionsPath } from "./openapi/paths/questions";
import { submissionsPath } from "./openapi/paths/submissions";
import { feedbackPaths } from "./openapi/paths/feedback";
import { healthPaths } from "./openapi/paths/health";

/**
 * Complete OpenAPI 3.0 specification combining metadata and all API paths.
 * Used by /openapi.json and /docs endpoints.
 */
export const openApiSpec = {
  ...openApiMetadata,
  paths: {
    ...questionsPath,
    ...submissionsPath,
    ...feedbackPaths,
    ...healthPaths,
  },
};
