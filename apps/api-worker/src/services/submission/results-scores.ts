/**
 * Essay scores extraction
 */

import type { AssessmentResults, AssessorResult } from "@writeo/shared";
import type { ModalRequest } from "@writeo/shared";
import { buildPartLookup } from "./utils";

type EssayScoreSummary = {
  overall?: number;
  dimensions?: {
    TA?: number;
    CC?: number;
    Vocab?: number;
    Grammar?: number;
    Overall?: number;
  };
  label?: string;
};

export function extractEssayScores(
  essayAssessment: AssessmentResults | null,
  modalParts: ModalRequest["parts"],
): Map<string, EssayScoreSummary> {
  const essayScoresByAnswerId = new Map<string, EssayScoreSummary>();
  const modalPartsById = buildPartLookup(modalParts);

  if (!essayAssessment?.results?.parts) {
    return essayScoresByAnswerId;
  }

  for (const essayPart of essayAssessment.results.parts) {
    const matchingPart = modalPartsById.get(essayPart.part);
    if (!matchingPart?.answers?.length) {
      continue;
    }

    const partAnswers = essayPart.answers ?? [];
    const essayAssessor = partAnswers
      .flatMap((answer) => (answer?.["assessor-results"] as AssessorResult[] | undefined) ?? [])
      .find((a) => a.id === "T-AES-ESSAY");

    if (!essayAssessor) {
      continue;
    }

    for (const answer of matchingPart.answers) {
      essayScoresByAnswerId.set(answer.id, {
        overall: essayAssessor.overall,
        dimensions: essayAssessor.dimensions,
        label: essayAssessor.label,
      });
    }
  }

  return essayScoresByAnswerId;
}
