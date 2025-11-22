/**
 * OpenAPI specification - combines all path modules
 */

import { openApiMetadata } from "./openapi/metadata";
import { questionsPath } from "./openapi/paths/questions";
import { submissionsPath } from "./openapi/paths/submissions";
import { feedbackPaths } from "./openapi/paths/feedback";
import { healthPaths } from "./openapi/paths/health";

export const openApiSpec = {
  ...openApiMetadata,
  paths: {
    ...questionsPath,
    ...submissionsPath,
    ...feedbackPaths,
    ...healthPaths,
  },
};
