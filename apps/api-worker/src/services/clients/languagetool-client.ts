/**
 * LanguageTool service client
 */

import { BaseServiceClient, type ServiceClientOptions } from "./base-client";
import { buildServiceError } from "./helpers";
import type { LanguageToolResponse } from "@writeo/shared";

export interface LanguageToolCheckRequest {
  language: string;
  text: string;
  answer_id: string;
}

const CHECK_ENDPOINT = "/check";
const LANGUAGETOOL_SERVICE_NAME = "LanguageTool";

const isLanguageToolResponse = (payload: unknown): payload is LanguageToolResponse => {
  if (!payload || typeof payload !== "object" || Array.isArray(payload)) {
    return false;
  }

  const { matches } = payload as Partial<LanguageToolResponse>;
  return Array.isArray(matches);
};

export class LanguageToolClient extends BaseServiceClient {
  constructor(options: ServiceClientOptions) {
    super({ ...options, timeout: options.timeout ?? 30000 }); // LanguageTool is faster, shorter default timeout
  }

  async check(request: LanguageToolCheckRequest): Promise<LanguageToolResponse> {
    const response = await this.request(CHECK_ENDPOINT, {
      method: "POST",
      body: JSON.stringify(request),
    });

    if (!response.ok) {
      throw await buildServiceError(LANGUAGETOOL_SERVICE_NAME, response);
    }

    const result = (await response.json()) as unknown;

    if (!isLanguageToolResponse(result)) {
      throw new Error(`${LANGUAGETOOL_SERVICE_NAME} returned an unexpected payload shape`);
    }

    return result;
  }
}
