/**
 * Essay scores extraction
 */

import type { AssessmentResults, AssessorResult } from "@writeo/shared";
import type { ModalRequest } from "@writeo/shared";

export function extractEssayScores(
  essayAssessment: AssessmentResults | null,
  modalParts: ModalRequest["parts"],
): Map<
  string,
  {
    overall?: number;
    dimensions?: {
      TA?: number;
      CC?: number;
      Vocab?: number;
      Grammar?: number;
      Overall?: number;
    };
    label?: string;
  }
> {
  const essayScoresByAnswerId = new Map<
    string,
    {
      overall?: number;
      dimensions?: {
        TA?: number;
        CC?: number;
        Vocab?: number;
        Grammar?: number;
        Overall?: number;
      };
      label?: string;
    }
  >();

  if (essayAssessment?.results?.parts) {
    for (const essayPart of essayAssessment.results.parts) {
      const matchingPart = modalParts.find((p) => p.part === essayPart.part);
      if (matchingPart) {
        const essayAssessor = essayPart.answers?.[0]?.["assessor-results"]?.find(
          (a: AssessorResult) => a.id === "T-AES-ESSAY",
        );
        if (essayAssessor) {
          for (const answer of matchingPart.answers) {
            essayScoresByAnswerId.set(answer.id, {
              overall: essayAssessor.overall,
              dimensions: essayAssessor.dimensions,
              label: essayAssessor.label,
            });
          }
        }
      }
    }
  }

  return essayScoresByAnswerId;
}
