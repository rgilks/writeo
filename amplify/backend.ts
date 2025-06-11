import { defineBackend } from '@aws-amplify/backend';
import { auth } from './auth/resource';
import { data } from './data/resource';
import { LanguageToolService } from './languagetool/resource';

const backend = defineBackend({
  auth,
  data,
});

const languageToolStack = backend.createStack('LanguageToolStack');
const languageToolService = new LanguageToolService(languageToolStack, 'LanguageToolService');

backend.addOutput({
  custom: {
    languageToolEndpoint: languageToolService.publicUrl,
    languageToolVpcId: languageToolService.vpc.vpcId,
  },
});
