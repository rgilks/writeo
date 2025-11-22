/**
 * LanguageTool service client
 */

import { BaseServiceClient, type ServiceClientOptions } from "./base-client";
import type { LanguageToolResponse } from "@writeo/shared";

export interface LanguageToolCheckRequest {
  language: string;
  text: string;
  answer_id: string;
}

export class LanguageToolClient extends BaseServiceClient {
  constructor(options: ServiceClientOptions) {
    super({ ...options, timeout: options.timeout ?? 30000 }); // LanguageTool is faster, shorter default timeout
  }

  async check(request: LanguageToolCheckRequest): Promise<LanguageToolResponse> {
    const response = await this.request("/check", {
      method: "POST",
      body: JSON.stringify(request),
    });

    if (!response.ok) {
      throw new Error(`LanguageTool check failed: HTTP ${response.status}`);
    }

    return response.json();
  }
}
