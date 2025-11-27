/**
 * Base service client - minimal abstraction for HTTP service calls
 */

import { fetchWithTimeout } from "../../utils/fetch-with-timeout";
import { retryWithBackoff, type RetryOptions } from "@writeo/shared";

export interface ServiceClientOptions {
  baseUrl: string;
  apiKey: string;
  timeout?: number;
  retry?: RetryOptions | false;
}

export type RequestOptions = RequestInit & { timeout?: number };

export interface RequestContext {
  url: string;
  options: RequestOptions;
}

const DEFAULT_TIMEOUT_MS = 60000;
const DEFAULT_RETRY_OPTIONS: RetryOptions = {
  maxAttempts: 3,
  baseDelayMs: 500,
  maxDelayMs: 10000,
};

const normalizeHeaders = (headers?: HeadersInit): Headers => {
  return headers ? new Headers(headers) : new Headers();
};

const buildRequestHeaders = (apiKey: string, headers?: HeadersInit): Headers => {
  const normalized = normalizeHeaders(headers);

  if (!normalized.has("Content-Type")) {
    normalized.set("Content-Type", "application/json");
  }

  normalized.set("Authorization", `Token ${apiKey}`);

  return normalized;
};

/**
 * Base class for service clients
 * Provides common functionality: timeout, retry, error handling
 */
export abstract class BaseServiceClient {
  protected baseUrl: string;
  protected apiKey: string;
  protected timeout: number;
  protected retryOptions?: RetryOptions;

  constructor(options: ServiceClientOptions) {
    this.baseUrl = options.baseUrl;
    this.apiKey = options.apiKey;
    this.timeout = options.timeout ?? DEFAULT_TIMEOUT_MS;
    this.retryOptions =
      options.retry === false ? undefined : (options.retry ?? DEFAULT_RETRY_OPTIONS);
  }

  /**
   * Makes a request with timeout and optional retry
   */
  protected async request(endpoint: string, options: RequestOptions = {}): Promise<Response> {
    const url = `${this.baseUrl}${endpoint}`;
    const { timeout, headers, ...fetchOptions } = options;
    const requestOptions: RequestOptions = {
      ...fetchOptions,
      timeout: timeout ?? this.timeout,
      headers: buildRequestHeaders(this.apiKey, headers),
    };

    const requestFn = async () => {
      try {
        return await fetchWithTimeout(url, requestOptions);
      } catch (error) {
        return this.handleRequestError(error, { url, options: requestOptions });
      }
    };

    if (this.retryOptions) {
      return retryWithBackoff(requestFn, this.retryOptions);
    }

    return requestFn();
  }

  /**
   * Hook for subclasses to log or transform request errors
   */
  protected handleRequestError(error: unknown, _context: RequestContext): never {
    if (error instanceof Error) {
      throw error;
    }
    throw new Error(String(error));
  }
}
