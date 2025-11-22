/**
 * Essay result processing
 */

import type { AssessmentResults } from "@writeo/shared";
import { safeLogError } from "../../utils/logging";

export async function processEssayResult(
  essayResult: PromiseSettledResult<Response>,
  submissionId: string
): Promise<AssessmentResults | null> {
  let essayAssessment: AssessmentResults | null = null;
  if (essayResult.status === "fulfilled") {
    const response = essayResult.value;
    if (response.ok) {
      try {
        essayAssessment = await response.json<AssessmentResults>();
      } catch (parseError) {
        const errorMsg = parseError instanceof Error ? parseError.message : String(parseError);
        safeLogError("Failed to parse essay assessment response", {
          error: errorMsg,
          status: response.status,
          statusText: response.statusText,
        });
      }
    } else {
      const errorText = await response.text().catch(() => response.statusText);
      safeLogError("Essay grading service failed", {
        status: response.status,
        statusText: response.statusText,
        error: errorText.substring(0, 500),
      });
    }
  } else {
    const errorMsg =
      essayResult.reason instanceof Error ? essayResult.reason.message : String(essayResult.reason);
    safeLogError("Essay grading request failed", {
      error: errorMsg,
      submissionId,
    });
  }
  return essayAssessment;
}
