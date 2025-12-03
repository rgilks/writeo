import { describe, it, expect, vi } from "vitest";
import { retryWithBackoff } from "../../packages/shared/ts/retry";

describe("retryWithBackoff", () => {
  it("should succeed on first attempt", async () => {
    const fn = vi.fn().mockResolvedValue("success");
    const result = await retryWithBackoff(fn);
    expect(result).toBe("success");
    expect(fn).toHaveBeenCalledTimes(1);
  });

  it("should retry on failure and succeed", async () => {
    vi.useFakeTimers();
    const fn = vi
      .fn()
      .mockRejectedValueOnce(new Error("Network error"))
      .mockResolvedValueOnce("success");

    const promise = retryWithBackoff(fn, { maxAttempts: 2, baseDelayMs: 100 });

    expect(fn).toHaveBeenCalledTimes(1);

    await vi.advanceTimersByTimeAsync(150);

    const result = await promise;
    expect(result).toBe("success");
    expect(fn).toHaveBeenCalledTimes(2);
    vi.useRealTimers();
  });

  it("should retry up to maxAttempts times", async () => {
    vi.useFakeTimers();
    const fn = vi
      .fn()
      .mockRejectedValueOnce(new Error("Always fails"))
      .mockRejectedValueOnce(new Error("Always fails"))
      .mockRejectedValueOnce(new Error("Always fails"));

    const promise = retryWithBackoff(fn, { maxAttempts: 3, baseDelayMs: 100 });

    expect(fn).toHaveBeenCalledTimes(1);

    await vi.runAllTimersAsync();

    await expect(promise).rejects.toThrow("Always fails");
    expect(fn).toHaveBeenCalledTimes(3);
    vi.useRealTimers();
  });

  it("should use exponential backoff", async () => {
    vi.useFakeTimers();
    const fn = vi
      .fn()
      .mockRejectedValueOnce(new Error("Fail"))
      .mockRejectedValueOnce(new Error("Fail"))
      .mockRejectedValueOnce(new Error("Fail"));

    const promise = retryWithBackoff(fn, { maxAttempts: 3, baseDelayMs: 100, maxDelayMs: 1000 });

    expect(fn).toHaveBeenCalledTimes(1);

    await vi.runAllTimersAsync();

    await expect(promise).rejects.toThrow("Fail");
    expect(fn).toHaveBeenCalledTimes(3);
    vi.useRealTimers();
  });

  it("should respect maxDelayMs", async () => {
    vi.useFakeTimers();
    const fn = vi
      .fn()
      .mockRejectedValueOnce(new Error("Fail"))
      .mockRejectedValueOnce(new Error("Fail"))
      .mockRejectedValueOnce(new Error("Fail"))
      .mockRejectedValueOnce(new Error("Fail"))
      .mockRejectedValueOnce(new Error("Fail"));

    const promise = retryWithBackoff(fn, {
      maxAttempts: 5,
      baseDelayMs: 1000,
      maxDelayMs: 2000,
    });

    expect(fn).toHaveBeenCalledTimes(1);

    await vi.runAllTimersAsync();

    await expect(promise).rejects.toThrow("Fail");
    expect(fn).toHaveBeenCalledTimes(5);
    vi.useRealTimers();
  });

  it("should not retry if shouldRetry returns false", async () => {
    const error = new Error("HTTP 400");
    const fn = vi.fn().mockRejectedValue(error);

    const shouldRetry = vi.fn().mockReturnValue(false);

    const promise = retryWithBackoff(fn, { shouldRetry });

    await expect(promise).rejects.toThrow("HTTP 400");
    expect(fn).toHaveBeenCalledTimes(1);
    expect(shouldRetry).toHaveBeenCalledWith(error);
  });

  it("should retry if shouldRetry returns true", async () => {
    vi.useFakeTimers();
    const error = new Error("Network error");
    const fn = vi.fn().mockRejectedValueOnce(error).mockResolvedValueOnce("success");

    const shouldRetry = vi.fn().mockReturnValue(true);

    const promise = retryWithBackoff(fn, { shouldRetry, baseDelayMs: 100 });

    expect(fn).toHaveBeenCalledTimes(1);

    await vi.advanceTimersByTimeAsync(150);

    const result = await promise;
    expect(result).toBe("success");
    expect(fn).toHaveBeenCalledTimes(2);
    expect(shouldRetry).toHaveBeenCalledWith(error);
    vi.useRealTimers();
  });

  it("should use default shouldRetry to skip 4xx errors", async () => {
    const error = new Error("HTTP 400 Bad Request");
    const fn = vi.fn().mockRejectedValue(error);

    const promise = retryWithBackoff(fn, { maxAttempts: 3, baseDelayMs: 100 });

    await expect(promise).rejects.toThrow("HTTP 400");
    expect(fn).toHaveBeenCalledTimes(1);
  });

  it("should retry non-4xx errors by default", async () => {
    vi.useFakeTimers();
    const error = new Error("Network timeout");
    const fn = vi.fn().mockRejectedValueOnce(error).mockResolvedValueOnce("success");

    const promise = retryWithBackoff(fn, { baseDelayMs: 100 });

    expect(fn).toHaveBeenCalledTimes(1);

    await vi.advanceTimersByTimeAsync(150);

    const result = await promise;
    expect(result).toBe("success");
    expect(fn).toHaveBeenCalledTimes(2);
    vi.useRealTimers();
  });

  it("should handle errors that are not Error instances", async () => {
    vi.useFakeTimers();
    const fn = vi.fn().mockRejectedValue("string error");

    const promise = retryWithBackoff(fn, { maxAttempts: 2, baseDelayMs: 100 });

    expect(fn).toHaveBeenCalledTimes(1);

    await vi.advanceTimersByTimeAsync(150);

    await expect(promise).rejects.toThrow("string error");
    expect(fn).toHaveBeenCalledTimes(2);
    vi.useRealTimers();
  });

  it("should use default maxAttempts of 3", async () => {
    vi.useFakeTimers();
    const fn = vi
      .fn()
      .mockRejectedValueOnce(new Error("Fail"))
      .mockRejectedValueOnce(new Error("Fail"))
      .mockRejectedValueOnce(new Error("Fail"));

    const promise = retryWithBackoff(fn, { baseDelayMs: 100 });

    expect(fn).toHaveBeenCalledTimes(1);

    await vi.runAllTimersAsync();

    await expect(promise).rejects.toThrow("Fail");
    expect(fn).toHaveBeenCalledTimes(3);
    vi.useRealTimers();
  });

  it("should use default baseDelayMs of 500", async () => {
    vi.useFakeTimers();
    const fn = vi.fn().mockRejectedValueOnce(new Error("Fail")).mockResolvedValueOnce("success");

    const promise = retryWithBackoff(fn);

    expect(fn).toHaveBeenCalledTimes(1);

    await vi.advanceTimersByTimeAsync(550);

    const result = await promise;
    expect(result).toBe("success");
    expect(fn).toHaveBeenCalledTimes(2);
    vi.useRealTimers();
  });

  it("should throw last error after all retries exhausted", async () => {
    vi.useFakeTimers();
    const lastError = new Error("Final error");
    const fn = vi
      .fn()
      .mockRejectedValueOnce(new Error("First error"))
      .mockRejectedValueOnce(new Error("Second error"))
      .mockRejectedValueOnce(lastError);

    const promise = retryWithBackoff(fn, { maxAttempts: 3, baseDelayMs: 100 });

    expect(fn).toHaveBeenCalledTimes(1);

    await vi.advanceTimersByTimeAsync(150);
    expect(fn).toHaveBeenCalledTimes(2);

    await vi.advanceTimersByTimeAsync(250);
    expect(fn).toHaveBeenCalledTimes(3);

    await expect(promise).rejects.toThrow("Final error");
    vi.useRealTimers();
  });

  it("should handle promise rejection correctly", async () => {
    vi.useFakeTimers();
    const fn = vi.fn().mockImplementation(() => Promise.reject(new Error("Rejected")));

    const promise = retryWithBackoff(fn, { maxAttempts: 2, baseDelayMs: 100 });

    expect(fn).toHaveBeenCalledTimes(1);

    await vi.advanceTimersByTimeAsync(150);

    await expect(promise).rejects.toThrow("Rejected");
    expect(fn).toHaveBeenCalledTimes(2);
    vi.useRealTimers();
  });

  it("should work with async functions", async () => {
    vi.useFakeTimers();
    const fn = vi.fn().mockImplementation(async () => {
      await new Promise((resolve) => setTimeout(resolve, 10));
      return "async success";
    });

    const promise = retryWithBackoff(fn);

    await vi.advanceTimersByTimeAsync(20);

    const result = await promise;
    expect(result).toBe("async success");
    vi.useRealTimers();
  });
});
