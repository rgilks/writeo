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
    console.log('LanguageTool API Request:', JSON.stringify(request, null, 2));
    const params = new URLSearchParams();
    params.append('text', request.text);
    params.append('language', request.language);
    if (request.motherTongue) {
      params.append('motherTongue', request.motherTongue);
    }
    params.append('enabledOnly', request.enabledOnly.toString());
    params.append('level', request.level);
    if (request.enabledCategories) {
      params.append('enabledCategories', request.enabledCategories);
    }

    const response = await fetch(`${this.endpoint}/v2/check`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/x-www-form-urlencoded',
      },
      body: params,
    });

    if (!response.ok) {
      throw new Error(`LanguageTool API error: ${response.status} ${response.statusText}`);
    }

    const data = await response.json();
    console.log('LanguageTool API Response:', JSON.stringify(data, null, 2));
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
