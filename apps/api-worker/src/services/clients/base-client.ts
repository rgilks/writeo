/**
 * Base service client - minimal abstraction for HTTP service calls
 */

import { fetchWithTimeout } from "../../utils/fetch-with-timeout";
import { retryWithBackoff, type RetryOptions } from "@writeo/shared";

export interface ServiceClientOptions {
  baseUrl: string;
  apiKey: string;
  timeout?: number;
  retry?: RetryOptions;
}

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
    this.timeout = options.timeout ?? 60000;
    this.retryOptions = options.retry;
  }

  /**
   * Makes a request with timeout and optional retry
   */
  protected async request(
    endpoint: string,
    options: RequestInit & { timeout?: number }
  ): Promise<Response> {
    const url = `${this.baseUrl}${endpoint}`;
    const { timeout, ...fetchOptions } = options;
    const requestFn = () =>
      fetchWithTimeout(url, {
        ...fetchOptions,
        timeout: timeout ?? this.timeout,
        headers: {
          "Content-Type": "application/json",
          Authorization: `Token ${this.apiKey}`,
          ...fetchOptions.headers,
        },
      });

    if (this.retryOptions) {
      return retryWithBackoff(requestFn, this.retryOptions);
    }

    return requestFn();
  }
}
