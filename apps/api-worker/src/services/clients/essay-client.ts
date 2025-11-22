/**
 * Essay scoring service client
 */

import { BaseServiceClient } from "./base-client";
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

export class EssayScoringClient extends BaseServiceClient {
  async grade(request: ModalRequest): Promise<EssayResult> {
    const response = await this.request("/grade", {
      method: "POST",
      body: JSON.stringify(request),
    });

    if (!response.ok) {
      throw new Error(`Essay scoring failed: HTTP ${response.status}`);
    }

    return response.json();
  }
}
