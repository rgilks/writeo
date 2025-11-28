/**
 * API configuration helpers
 * These can be used in both server actions and route handlers
 *
 * Note: With OpenNext + cloudflare-node + nodejs_compat, process.env should be available.
 * Cloudflare Workers inject [vars] from wrangler.toml and secrets into process.env.
 */

/**
 * Gets the first available environment variable from the provided keys
 */
function getEnv(...keys: string[]): string | undefined {
  for (const key of keys) {
    const value = process.env[key];
    if (value) return value;
  }
  return undefined;
}

/**
 * Gets the API base URL from environment variables
 * Supports multiple env var names for different deployment configurations
 */
export function getApiBase(): string {
  return (
    getEnv(
      "API_BASE_URL",
      "NEXT_PUBLIC_API_BASE",
      "PRODUCTION_API_URL",
      "PRODUCTION_API_BASE_URL",
      "API_BASE",
    ) ?? "MISSING_API_BASE_URL"
  );
}

/**
 * Gets the API key from environment variables
 * Supports multiple env var names so CI/Cloudflare deployments don't need renames
 */
export function getApiKey(): string {
  return (
    getEnv("API_KEY", "PRODUCTION_API_KEY", "NEXT_PUBLIC_API_KEY", "API_TOKEN") ?? "MISSING_API_KEY"
  );
}
