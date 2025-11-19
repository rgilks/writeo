/**
 * API configuration helpers
 * These can be used in both server actions and route handlers
 */

// Helper to get API base URL
// With OpenNext + cloudflare-node + nodejs_compat, process.env should be available
// Cloudflare Workers inject [vars] from wrangler.toml and secrets into process.env
export const getApiBase = (): string => {
  // process.env should be available with nodejs_compat flag
  const base = process.env.API_BASE_URL || process.env.NEXT_PUBLIC_API_BASE;
  if (!base) {
    return "MISSING_API_BASE_URL";
  }
  return base;
};

// Helper to get API key
// With OpenNext + cloudflare-node + nodejs_compat, process.env should be available
// Secrets set via 'wrangler secret put' are injected into process.env
export const getApiKey = (): string => {
  // process.env should be available with nodejs_compat flag
  const key = process.env.API_KEY;
  if (!key) {
    return "MISSING_API_KEY";
  }
  return key;
};
