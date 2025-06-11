import { Amplify } from 'aws-amplify';

let amplifyConfig: Record<string, unknown> | null = null;

try {
  // eslint-disable-next-line @typescript-eslint/no-require-imports
  amplifyConfig = require('../../amplify_outputs.json');
  if (amplifyConfig) {
    Amplify.configure(amplifyConfig);
  }
} catch {
  console.warn('Amplify outputs not found, using environment variables');
}

export const getLanguageToolEndpoint = (): string => {
  if (
    amplifyConfig?.custom &&
    typeof amplifyConfig.custom === 'object' &&
    amplifyConfig.custom !== null
  ) {
    const custom = amplifyConfig.custom as Record<string, unknown>;
    if (typeof custom.languageToolEndpoint === 'string') {
      return custom.languageToolEndpoint;
    }
  }

  const endpoint = process.env.LANGUAGETOOL_ENDPOINT;
  if (!endpoint) {
    throw new Error(
      'LanguageTool endpoint not configured. Please deploy the Amplify backend or set LANGUAGETOOL_ENDPOINT environment variable.'
    );
  }

  return endpoint;
};

export const config = {
  languageTool: {
    endpoint: getLanguageToolEndpoint,
  },
};
