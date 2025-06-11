import { defineBackend } from '@aws-amplify/backend';
import { auth } from './auth/resource';
import { data } from './data/resource';
import { createLanguageToolService } from './languagetool/resource';

const backend = defineBackend({
  auth,
  data,
});

const languageToolService = createLanguageToolService(backend);

backend.addOutput({
  custom: {
    languageToolEndpoint: languageToolService.serviceUrl,
    languageToolVpcId: languageToolService.vpc.vpcId,
  },
});
