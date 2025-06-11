import { defineBackend } from '@aws-amplify/backend';
import { auth } from './auth/resource';
import { data } from './data/resource';
import { createLanguageToolService } from './languagetool/resource';

const backend = defineBackend({
  auth,
  data,
});

const languageToolStack = backend.createStack('LanguageToolStack');
const languageToolService = createLanguageToolService(languageToolStack, 'LanguageTool');

backend.addOutput({
  custom: {
    languageToolEndpoint: languageToolService.endpoint,
    languageToolVpcId: languageToolService.vpcId,
  },
});
