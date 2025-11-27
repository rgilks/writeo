/**
 * Essay scoring service client
 */

import { BaseServiceClient } from "./base-client";
import { buildServiceError } from "./helpers";
import type { ModalRequest } from "@writeo/shared";

export interface EssayResult {
  submission_id: string;
  parts: Array<{
    part: number;
    answers: Array<{
      answer_id: string;
      scores: {
        TA?: number;
        CC?: number;
        Vocab?: number;
        Grammar?: number;
        Overall?: number;
      };
      label?: string;
    }>;
  }>;
}

const GRADE_ENDPOINT = "/grade";
const ESSAY_SERVICE_NAME = "Essay scoring";

const isEssayResult = (payload: unknown): payload is EssayResult => {
  if (!payload || typeof payload !== "object" || Array.isArray(payload)) {
    return false;
  }

  const { submission_id, parts } = payload as Partial<EssayResult>;
  return (
    typeof submission_id === "string" &&
    Array.isArray(parts) &&
    parts.every(
      (part) =>
        typeof part === "object" &&
        part !== null &&
        typeof part.part === "number" &&
        Array.isArray(part.answers),
    )
  );
};

export class EssayScoringClient extends BaseServiceClient {
  async grade(request: ModalRequest): Promise<EssayResult> {
    const response = await this.request(GRADE_ENDPOINT, {
      method: "POST",
      body: JSON.stringify(request),
    });

    if (!response.ok) {
      throw await buildServiceError(ESSAY_SERVICE_NAME, response);
    }

    const result = (await response.json()) as unknown;

    if (!isEssayResult(result)) {
      throw new Error(`${ESSAY_SERVICE_NAME} returned an unexpected payload shape`);
    }

    return result;
  }
}
