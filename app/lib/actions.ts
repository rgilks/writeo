'use server';

import { createLanguageToolClient } from './languagetool-client';
import {
  LanguageToolCheckRequest,
  LanguageToolCheckRequestSchema,
  LanguageToolResponse,
} from './types';
import { getLanguageToolEndpoint } from './config';

export async function checkText(
  request: LanguageToolCheckRequest
): Promise<{ success: true; data: LanguageToolResponse } | { success: false; error: string }> {
  try {
    const validatedRequest = LanguageToolCheckRequestSchema.parse(request);
    const client = createLanguageToolClient(getLanguageToolEndpoint());
    const result = await client.checkText(validatedRequest);

    return { success: true, data: result };
  } catch (error) {
    console.error('Error checking text with LanguageTool:', error);
    return {
      success: false,
      error: error instanceof Error ? error.message : 'Unknown error occurred',
    };
  }
}

export async function getAvailableLanguages(): Promise<
  | { success: true; data: Array<{ name: string; code: string; longCode: string }> }
  | { success: false; error: string }
> {
  try {
    const client = createLanguageToolClient(getLanguageToolEndpoint());
    const languages = await client.getLanguages();

    return { success: true, data: languages };
  } catch (error) {
    console.error('Error fetching languages from LanguageTool:', error);
    return {
      success: false,
      error: error instanceof Error ? error.message : 'Unknown error occurred',
    };
  }
}

export async function checkLanguageToolHealth(): Promise<{ success: boolean; message: string }> {
  try {
    const client = createLanguageToolClient(getLanguageToolEndpoint());
    const isHealthy = await client.healthCheck();

    return {
      success: isHealthy,
      message: isHealthy
        ? 'LanguageTool service is healthy'
        : 'LanguageTool service is not responding',
    };
  } catch (error) {
    console.error('Error checking LanguageTool health:', error);
    return {
      success: false,
      message: error instanceof Error ? error.message : 'Unknown error occurred',
    };
  }
}
