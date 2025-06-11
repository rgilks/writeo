import {
  LanguageToolCheckRequest,
  LanguageToolResponse,
  LanguageToolResponseSchema,
} from './types';

class LanguageToolClient {
  private endpoint: string;

  constructor(endpoint: string) {
    this.endpoint = endpoint;
  }

  async checkText(request: LanguageToolCheckRequest): Promise<LanguageToolResponse> {
    const formData = new FormData();
    formData.append('text', request.text);
    formData.append('language', request.language);
    formData.append('enabledOnly', request.enabledOnly.toString());
    formData.append('level', request.level);

    const response = await fetch(`${this.endpoint}/v2/check`, {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      throw new Error(`LanguageTool API error: ${response.status} ${response.statusText}`);
    }

    const data = await response.json();
    return LanguageToolResponseSchema.parse(data);
  }

  async getLanguages(): Promise<Array<{ name: string; code: string; longCode: string }>> {
    const response = await fetch(`${this.endpoint}/v2/languages`);

    if (!response.ok) {
      throw new Error(`LanguageTool API error: ${response.status} ${response.statusText}`);
    }

    return response.json();
  }

  async healthCheck(): Promise<boolean> {
    try {
      const response = await fetch(`${this.endpoint}/v2/languages`, {
        method: 'GET',
      });
      return response.ok;
    } catch {
      return false;
    }
  }
}

export const createLanguageToolClient = (endpoint: string) => new LanguageToolClient(endpoint);
