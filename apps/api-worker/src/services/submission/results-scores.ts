/**
 * Essay scores extraction
 */

import type { AssessmentResults } from "@writeo/shared";
import type { ModalRequest } from "@writeo/shared";

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
  _essayAssessment: AssessmentResults | null,
  modalParts: ModalRequest["parts"],
  genericResults?: Map<string, Map<string, unknown>>,
): Map<string, EssayScoreSummary> {
  const essayScoresByAnswerId = new Map<string, EssayScoreSummary>();

  for (const part of modalParts) {
    for (const answer of part.answers) {
      // 1. Check DeBERTa (new model)
      if (genericResults) {
        const debertaService = genericResults.get("deberta");
        if (debertaService) {
          const result = debertaService.get(answer.id) as any; // Cast as any or import DebertaResult if available
          if (result && typeof result.overall === "number") {
            essayScoresByAnswerId.set(answer.id, {
              overall: result.overall,
              dimensions: result.dimensions,
              label: result.label,
            });
          }
        }
      }
    }
  }

  return essayScoresByAnswerId;
}
