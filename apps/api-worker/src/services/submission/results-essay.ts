/**
 * Essay result processing
 */

import type { AssessmentResults, AssessorResult } from "@writeo/shared";
import { safeLogError } from "../../utils/logging";

async function readErrorTextWithTimeout(response: Response, timeoutMs = 1000): Promise<string> {
  const fallback = response.statusText;
  const timeoutPromise = new Promise<string>((_, reject) =>
    setTimeout(() => reject(new Error("response body timeout")), timeoutMs),
  );

  try {
    return await Promise.race([response.text(), timeoutPromise]);
  } catch {
    return fallback;
  }
}

function logAssessorDetails(essayAssessment: AssessmentResults, submissionId: string) {
  if (!essayAssessment?.results?.parts) {
    return;
  }

  for (const part of essayAssessment.results.parts) {
    if (!part.answers?.length) {
      continue;
    }

    const firstAnswer = part.answers[0];
    const assessorResults = firstAnswer?.assessorResults as AssessorResult[] | undefined;
    console.log(
      `[Essay Assessment][debug] Part ${part.part} has ${assessorResults?.length ?? 0} assessor result(s)`,
      {
        submissionId,
        assessorIds: assessorResults?.map((ar) => ar.id) ?? [],
        assessorNames: assessorResults?.map((ar) => ar.name) ?? [],
      },
    );
  }
}

async function parseEssayAssessmentResponse(
  response: Response,
  submissionId: string,
): Promise<AssessmentResults | null> {
  try {
    const essayAssessment = await response.json<AssessmentResults>();
    console.log(
      `[Essay Assessment] Successfully parsed essay assessment for submission ${submissionId}`,
      {
        hasResults: !!essayAssessment?.results,
        hasParts: !!essayAssessment?.results?.parts,
        partsCount: essayAssessment?.results?.parts?.length ?? 0,
        status: essayAssessment?.status,
      },
    );
    logAssessorDetails(essayAssessment, submissionId);

    // Validate that we have essay assessor results
    if (essayAssessment?.results?.parts) {
      for (const part of essayAssessment.results.parts) {
        if (part.answers?.length) {
          for (const answer of part.answers) {
            const assessorResults = answer.assessorResults as AssessorResult[] | undefined;
            const essayAssessor = assessorResults?.find((ar) => ar.id === "T-AES-ESSAY");
            if (!essayAssessor) {
              safeLogError("Essay assessment missing T-AES-ESSAY assessor result", {
                submissionId,
                part: part.part,
                answerId: answer.id,
                assessorIds: assessorResults?.map((ar) => ar.id) ?? [],
                assessorCount: assessorResults?.length ?? 0,
              });
            } else {
              console.log(`[Essay Assessment] Found essay assessor for answer ${answer.id}`, {
                overall: essayAssessor.overall,
                label: essayAssessor.label,
                hasDimensions: !!essayAssessor.dimensions,
              });
            }
          }
        }
      }
    }

    return essayAssessment;
  } catch (parseError) {
    const errorMsg = parseError instanceof Error ? parseError.message : String(parseError);
    safeLogError("Failed to parse essay assessment response", {
      error: errorMsg,
      status: response.status,
      statusText: response.statusText,
      submissionId,
    });
    return null;
  }
}

export async function processEssayResult(
  essayResult: PromiseSettledResult<Response>,
  submissionId: string,
): Promise<AssessmentResults | null> {
  let essayAssessment: AssessmentResults | null = null;
  if (essayResult.status === "rejected") {
    const errorMsg =
      essayResult.reason instanceof Error ? essayResult.reason.message : String(essayResult.reason);
    safeLogError("Essay grading request failed", {
      error: errorMsg,
      submissionId,
    });
    return null;
  }

  const response = essayResult.value;

  if (!response.ok) {
    const errorText = await readErrorTextWithTimeout(response);
    safeLogError("Essay grading service failed", {
      status: response.status,
      statusText: response.statusText,
      error: errorText.substring(0, 500),
      submissionId,
    });
    return null;
  }

  essayAssessment = await parseEssayAssessmentResponse(response, submissionId);

  return essayAssessment;
}
