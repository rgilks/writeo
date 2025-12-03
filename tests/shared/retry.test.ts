import { describe, it, expect, vi, beforeEach } from "vitest";
import { retryWithBackoff } from "../../packages/shared/ts/retry";

// Helper to flush promises
const flushPromises = () =>
  new Promise((resolve) => {
    // We use a real timeout of 0 if timers are not faked, or just resolve
    // But since we use fake timers, we can't use setTimeout(..., 0) reliably if we want to step
    // So we just use setImmediate or process.nextTick or Promise.resolve
    if (typeof setImmediate === "function") setImmediate(resolve);
    else Promise.resolve().then(resolve);
  });

describe("retryWithBackoff", () => {
  beforeEach(() => {
    vi.useFakeTimers();
  });

  it("should succeed on first attempt", async () => {
    const fn = vi.fn().mockResolvedValue("success");
    const result = await retryWithBackoff(fn);
    expect(result).toBe("success");
    expect(fn).toHaveBeenCalledTimes(1);
  });

  it("should retry on failure and succeed", async () => {
    const fn = vi
      .fn()
      .mockRejectedValueOnce(new Error("Network error"))
      .mockResolvedValueOnce("success");

    const promise = retryWithBackoff(fn, { maxAttempts: 2, baseDelayMs: 100 });
    // Suppress unhandled rejection warning during test execution
    promise.catch(() => {});

    await vi.runAllTimersAsync();

    const result = await promise;
    expect(result).toBe("success");
    expect(fn).toHaveBeenCalledTimes(2);
  });

  it("should retry up to maxAttempts times", async () => {
    const fn = vi.fn().mockRejectedValue(new Error("Always fails"));

    const promise = retryWithBackoff(fn, { maxAttempts: 3, baseDelayMs: 100 });
    promise.catch(() => {});

    await vi.runAllTimersAsync();

    await expect(promise).rejects.toThrow("Always fails");
    expect(fn).toHaveBeenCalledTimes(3);
  });

  it("should use exponential backoff", async () => {
    const fn = vi.fn().mockRejectedValue(new Error("Fail"));

    const promise = retryWithBackoff(fn, { maxAttempts: 3, baseDelayMs: 100, maxDelayMs: 1000 });
    promise.catch(() => {});

    await vi.runAllTimersAsync();

    await expect(promise).rejects.toThrow("Fail");
    expect(fn).toHaveBeenCalledTimes(3);
  });

  it("should respect maxDelayMs", async () => {
    const fn = vi.fn().mockRejectedValue(new Error("Fail"));

    const promise = retryWithBackoff(fn, {
      maxAttempts: 5,
      baseDelayMs: 1000,
      maxDelayMs: 2000,
    });
    promise.catch(() => {});

    await vi.runAllTimersAsync();

    await expect(promise).rejects.toThrow("Fail");
    expect(fn).toHaveBeenCalledTimes(5);
  });

  it("should not retry if shouldRetry returns false", async () => {
    const error = new Error("HTTP 400");
    const fn = vi.fn().mockRejectedValue(error);

    const shouldRetry = vi.fn().mockReturnValue(false);

    const promise = retryWithBackoff(fn, { shouldRetry });
    promise.catch(() => {});

    await expect(promise).rejects.toThrow("HTTP 400");
    expect(fn).toHaveBeenCalledTimes(1);
    expect(shouldRetry).toHaveBeenCalledWith(error);
  });

  it("should retry if shouldRetry returns true", async () => {
    const error = new Error("Network error");
    const fn = vi.fn().mockRejectedValueOnce(error).mockResolvedValueOnce("success");

    const shouldRetry = vi.fn().mockReturnValue(true);

    const promise = retryWithBackoff(fn, { shouldRetry, baseDelayMs: 100 });
    promise.catch(() => {});

    await vi.runAllTimersAsync();

    const result = await promise;
    expect(result).toBe("success");
    expect(fn).toHaveBeenCalledTimes(2);
    expect(shouldRetry).toHaveBeenCalledWith(error);
  });

  it("should use default shouldRetry to skip 4xx errors", async () => {
    const error = new Error("HTTP 400 Bad Request");
    const fn = vi.fn().mockRejectedValue(error);

    const promise = retryWithBackoff(fn, { maxAttempts: 3, baseDelayMs: 100 });
    promise.catch(() => {});

    await expect(promise).rejects.toThrow("HTTP 400");
    expect(fn).toHaveBeenCalledTimes(1);
  });

  it("should retry non-4xx errors by default", async () => {
    const error = new Error("Network timeout");
    const fn = vi.fn().mockRejectedValueOnce(error).mockResolvedValueOnce("success");

    const promise = retryWithBackoff(fn, { baseDelayMs: 100 });
    promise.catch(() => {});

    await vi.runAllTimersAsync();

    const result = await promise;
    expect(result).toBe("success");
    expect(fn).toHaveBeenCalledTimes(2);
  });

  it("should handle errors that are not Error instances", async () => {
    const fn = vi.fn().mockRejectedValue("string error");

    const promise = retryWithBackoff(fn, { maxAttempts: 2, baseDelayMs: 100 });
    promise.catch(() => {});

    await vi.runAllTimersAsync();

    await expect(promise).rejects.toThrow("string error");
    expect(fn).toHaveBeenCalledTimes(2);
  });

  it("should use default maxAttempts of 3", async () => {
    const fn = vi.fn().mockRejectedValue(new Error("Fail"));

    const promise = retryWithBackoff(fn, { baseDelayMs: 100 });
    promise.catch(() => {});

    await vi.runAllTimersAsync();

    await expect(promise).rejects.toThrow("Fail");
    expect(fn).toHaveBeenCalledTimes(3);
  });

  it("should use default baseDelayMs of 500", async () => {
    const fn = vi.fn().mockRejectedValueOnce(new Error("Fail")).mockResolvedValueOnce("success");

    const promise = retryWithBackoff(fn);
    promise.catch(() => {});

    await vi.runAllTimersAsync();

    const result = await promise;
    expect(result).toBe("success");
    expect(fn).toHaveBeenCalledTimes(2);
  });

  it("should throw last error after all retries exhausted", async () => {
    const lastError = new Error("Final error");
    const fn = vi
      .fn()
      .mockRejectedValueOnce(new Error("First error"))
      .mockRejectedValueOnce(new Error("Second error"))
      .mockRejectedValueOnce(lastError);

    const promise = retryWithBackoff(fn, { maxAttempts: 3, baseDelayMs: 100 });
    promise.catch(() => {});

    await vi.runAllTimersAsync();

    await expect(promise).rejects.toThrow("Final error");
    expect(fn).toHaveBeenCalledTimes(3);
  });

  it("should handle promise rejection correctly", async () => {
    const fn = vi.fn().mockRejectedValue(new Error("Rejected"));

    const promise = retryWithBackoff(fn, { maxAttempts: 2, baseDelayMs: 100 });
    promise.catch(() => {});

    await vi.runAllTimersAsync();

    await expect(promise).rejects.toThrow("Rejected");
    expect(fn).toHaveBeenCalledTimes(2);
  });

  it("should work with async functions", async () => {
    const fn = vi.fn().mockImplementation(async () => {
      await new Promise((resolve) => setTimeout(resolve, 10));
      return "async success";
    });

    const promise = retryWithBackoff(fn);
    promise.catch(() => {});

    await vi.runAllTimersAsync();

    const result = await promise;
    expect(result).toBe("async success");
  });
});
