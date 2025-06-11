import type { NextConfig } from 'next';

const nextConfig: NextConfig = {
  env: {
    LANGUAGETOOL_ENDPOINT: process.env.LANGUAGETOOL_ENDPOINT || '',
  },
  serverExternalPackages: ['aws-cdk-lib'],
};

export default nextConfig;
