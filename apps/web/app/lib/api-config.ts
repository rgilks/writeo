/**
 * API configuration helpers
 * These can be used in both server actions and route handlers
 */

// Helper to get API base URL
// With OpenNext + cloudflare-node + nodejs_compat, process.env should be available
// Cloudflare Workers inject [vars] from wrangler.toml and secrets into process.env
const getEnv = (...keys: Array<keyof NodeJS.ProcessEnv>): string | undefined => {
  for (const key of keys) {
    const value = process.env[key];
    if (value) return value;
  }
  return undefined;
};

export const getApiBase = (): string => {
  // Prefer explicit overrides but fall back to deployment-specific env names
  const base =
    getEnv(
      "API_BASE_URL",
      "NEXT_PUBLIC_API_BASE",
      "PRODUCTION_API_URL",
      "PRODUCTION_API_BASE_URL",
      "API_BASE",
    ) || undefined;
  return base || "MISSING_API_BASE_URL";
};

// Helper to get API key
// With OpenNext + cloudflare-node + nodejs_compat, process.env should be available
// Secrets set via 'wrangler secret put' are injected into process.env
export const getApiKey = (): string => {
  // Accept multiple env names so CI/Cloudflare deployments don't need renames
  const key =
    getEnv("API_KEY", "PRODUCTION_API_KEY", "NEXT_PUBLIC_API_KEY", "API_TOKEN") || undefined;
  return key || "MISSING_API_KEY";
};
