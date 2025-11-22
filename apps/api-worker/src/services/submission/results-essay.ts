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
        console.log(
          `[Essay Assessment] Successfully parsed essay assessment for submission ${submissionId}`,
          {
            hasResults: !!essayAssessment?.results,
            hasParts: !!essayAssessment?.results?.parts,
            partsCount: essayAssessment?.results?.parts?.length ?? 0,
          }
        );
        // Log assessor results for debugging
        if (essayAssessment?.results?.parts) {
          for (const part of essayAssessment.results.parts) {
            if (part.answers && part.answers.length > 0) {
              const firstAnswer = part.answers[0];
              const assessorResults = firstAnswer?.["assessor-results"];
              console.log(
                `[Essay Assessment] Part ${part.part} has ${assessorResults?.length ?? 0} assessor result(s)`,
                {
                  assessorIds: assessorResults?.map((ar: any) => ar.id) ?? [],
                  assessorNames: assessorResults?.map((ar: any) => ar.name) ?? [],
                }
              );
            }
          }
        }
      } catch (parseError) {
        const errorMsg = parseError instanceof Error ? parseError.message : String(parseError);
        safeLogError("Failed to parse essay assessment response", {
          error: errorMsg,
          status: response.status,
          statusText: response.statusText,
          submissionId,
        });
      }
    } else {
      const errorText = await response.text().catch(() => response.statusText);
      safeLogError("Essay grading service failed", {
        status: response.status,
        statusText: response.statusText,
        error: errorText.substring(0, 500),
        submissionId,
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
