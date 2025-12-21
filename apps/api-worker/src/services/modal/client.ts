import type { AppConfig } from "../config";
import { postJsonWithAuth } from "../../utils/http";
import type { ModalService } from "./types";

export class ModalClient implements ModalService {
  constructor(private config: AppConfig) {}

  /**
   * Helper for simple POST JSON requests without auth (public Modal services)
   */
  private postJson(url: string, body: object): Promise<Response> {
    return fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
  }

  async checkGrammar(text: string, language: string, answerId: string): Promise<Response> {
    if (!this.config.modal.ltUrl) {
      throw new Error("LanguageTool URL not configured");
    }
    return postJsonWithAuth(
      `${this.config.modal.ltUrl}/check`,
      this.config.api.key,
      { language, text, answer_id: answerId },
      30000,
    );
  }

  async scoreFeedback(text: string): Promise<Response> {
    return this.postJson(`${this.config.modal.feedbackUrl}/score`, { text });
  }

  async scoreDeberta(text: string): Promise<Response> {
    return this.postJson(`${this.config.modal.debertaUrl}/score`, { text, max_length: 512 });
  }

  async correctGrammar(text: string): Promise<Response> {
    return this.postJson(`${this.config.modal.gecUrl}/gec_endpoint`, { text });
  }

  async correctGrammarGector(text: string): Promise<Response> {
    return this.postJson(`${this.config.modal.gectorUrl}/gector_endpoint`, { text });
  }
}
