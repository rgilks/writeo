/**
 * Unit tests for storage utilities
 */

import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import {
  createSafeStorage,
  cleanupExpiredStorage,
  getAvailableStorageSpace,
  StorageError,
} from "../../apps/web/app/lib/utils/storage";

describe("getAvailableStorageSpace", () => {
  it("should return null when window is undefined", () => {
    const originalWindow = global.window;
    // @ts-ignore
    delete global.window;

    const result = getAvailableStorageSpace();
    expect(result).toBeNull();

    global.window = originalWindow;
  });

  it("should return estimated storage space when window is available", () => {
    if (typeof window !== "undefined") {
      const result = getAvailableStorageSpace();
      expect(result).toBe(5 * 1024 * 1024); // 5MB
    }
  });
});

describe("createSafeStorage", () => {
  let mockLocalStorage: Record<string, string>;
  let originalLocalStorage: Storage | undefined;

  beforeEach(() => {
    mockLocalStorage = {};
    originalLocalStorage = global.localStorage;

    // Mock localStorage - getItem must actually read from mockLocalStorage
    const localStorageMock = {
      getItem: (key: string) => {
        try {
          return mockLocalStorage[key] || null;
        } catch {
          return null;
        }
      },
      setItem: (key: string, value: string) => {
        mockLocalStorage[key] = value;
      },
      removeItem: (key: string) => {
        delete mockLocalStorage[key];
      },
      clear: () => {
        mockLocalStorage = {};
      },
      key: (index: number) => Object.keys(mockLocalStorage)[index] || null,
      get length() {
        return Object.keys(mockLocalStorage).length;
      },
    };
    global.localStorage = localStorageMock as any;

    Object.defineProperty(global.localStorage, "length", {
      get: () => Object.keys(mockLocalStorage).length,
    });
  });

  afterEach(() => {
    if (originalLocalStorage) {
      global.localStorage = originalLocalStorage;
    }
    vi.clearAllMocks();
  });

  describe("getItem", () => {
    it("should return null when window is undefined", () => {
      const originalWindow = global.window;
      // @ts-ignore
      delete global.window;

      const storage = createSafeStorage();
      const result = storage.getItem("test");
      expect(result).toBeNull();

      global.window = originalWindow;
    });

    it("should return stored value", () => {
      mockLocalStorage["test"] = '{"key": "value"}';
      const storage = createSafeStorage();
      const result = storage.getItem("test");
      expect(result).toBe('{"key": "value"}');
    });

    it("should return null for non-existent key", () => {
      const storage = createSafeStorage();
      const result = storage.getItem("nonexistent");
      expect(result).toBeNull();
    });

    it("should detect and remove corrupted data", () => {
      mockLocalStorage["test"] = "[object Object]";
      const storage = createSafeStorage();
      const result = storage.getItem("test");
      expect(result).toBeNull();
      expect(global.localStorage.removeItem).toHaveBeenCalledWith("test");
    });

    it("should handle localStorage errors gracefully", () => {
      global.localStorage.getItem = vi.fn(() => {
        throw new Error("Storage error");
      });

      const storage = createSafeStorage();
      const result = storage.getItem("test");
      expect(result).toBeNull();
    });
  });

  describe("setItem", () => {
    it("should store value successfully", () => {
      const storage = createSafeStorage();
      storage.setItem("test", '{"key": "value"}');
      expect(global.localStorage.setItem).toHaveBeenCalledWith("test", '{"key": "value"}');
    });

    it("should warn when window is undefined", () => {
      const originalWindow = global.window;
      const consoleWarn = vi.spyOn(console, "warn").mockImplementation(() => {});
      // @ts-ignore
      delete global.window;

      const storage = createSafeStorage();
      storage.setItem("test", "value");

      expect(consoleWarn).toHaveBeenCalledWith("localStorage is not available");

      global.window = originalWindow;
      consoleWarn.mockRestore();
    });

    it("should throw StorageError on quota exceeded", () => {
      const quotaError = new DOMException("Quota exceeded", "QuotaExceededError");
      quotaError.code = 22;

      global.localStorage.setItem = vi.fn(() => {
        throw quotaError;
      });

      const storage = createSafeStorage();
      expect(() => storage.setItem("test", "value")).toThrow(StorageError);
      expect(() => storage.setItem("test", "value")).toThrow("quota exceeded");
    });

    it("should throw StorageError on security error", () => {
      const securityError = new DOMException("Security error", "SecurityError");
      securityError.code = 18;

      global.localStorage.setItem = vi.fn(() => {
        throw securityError;
      });

      const storage = createSafeStorage();
      expect(() => storage.setItem("test", "value")).toThrow(StorageError);
      expect(() => storage.setItem("test", "value")).toThrow("disabled");
    });

    it("should retry after cleanup on quota error", () => {
      let callCount = 0;
      const quotaError = new DOMException("Quota exceeded", "QuotaExceededError");
      quotaError.code = 22;

      global.localStorage.setItem = vi.fn(() => {
        callCount++;
        if (callCount === 1) {
          throw quotaError;
        }
        // Second call succeeds
      });

      // Mock getAllKeys to return some keys for cleanup
      const originalKey = global.localStorage.key;
      global.localStorage.key = vi.fn((index: number) => {
        const keys = ["results_1", "results_2"];
        return keys[index] || null;
      });
      Object.defineProperty(global.localStorage, "length", {
        get: () => 2,
      });

      const storage = createSafeStorage();
      storage.setItem("test", "value");

      // Should have tried cleanup and retried
      expect(global.localStorage.setItem).toHaveBeenCalledTimes(2);

      global.localStorage.key = originalKey;
    });

    it("should warn about large values", () => {
      const consoleWarn = vi.spyOn(console, "warn").mockImplementation(() => {});
      const largeValue = "x".repeat(5 * 1024 * 1024); // 5MB

      const storage = createSafeStorage();
      storage.setItem("test", largeValue);

      expect(consoleWarn).toHaveBeenCalledWith(
        expect.stringContaining("large"),
        expect.any(Number),
      );

      consoleWarn.mockRestore();
    });
  });

  describe("removeItem", () => {
    it("should remove item successfully", () => {
      mockLocalStorage["test"] = "value";
      const storage = createSafeStorage();
      storage.removeItem("test");
      expect(global.localStorage.removeItem).toHaveBeenCalledWith("test");
    });

    it("should handle errors gracefully", () => {
      global.localStorage.removeItem = vi.fn(() => {
        throw new Error("Remove error");
      });

      const storage = createSafeStorage();
      // Should not throw
      expect(() => storage.removeItem("test")).not.toThrow();
    });

    it("should return early when window is undefined", () => {
      const originalWindow = global.window;
      // @ts-ignore
      delete global.window;

      const storage = createSafeStorage();
      storage.removeItem("test");
      // Should not call localStorage.removeItem
      expect(global.localStorage.removeItem).not.toHaveBeenCalled();

      global.window = originalWindow;
    });
  });
});

describe("cleanupExpiredStorage", () => {
  let mockLocalStorage: Record<string, string>;
  let originalLocalStorage: Storage | undefined;

  beforeEach(() => {
    mockLocalStorage = {};
    originalLocalStorage = global.localStorage;

    global.localStorage = {
      getItem: vi.fn((key: string) => mockLocalStorage[key] || null),
      setItem: vi.fn(),
      removeItem: vi.fn((key: string) => {
        delete mockLocalStorage[key];
      }),
      clear: vi.fn(),
      key: vi.fn((index: number) => Object.keys(mockLocalStorage)[index] || null),
      length: 0,
    } as any;

    Object.defineProperty(global.localStorage, "length", {
      get: () => Object.keys(mockLocalStorage).length,
    });
  });

  afterEach(() => {
    if (originalLocalStorage) {
      global.localStorage = originalLocalStorage;
    }
    vi.clearAllMocks();
  });

  it("should cleanup expired entries", () => {
    const now = Date.now();
    const oldTimestamp = now - 31 * 24 * 60 * 60 * 1000; // 31 days ago
    const recentTimestamp = now - 10 * 24 * 60 * 60 * 1000; // 10 days ago

    mockLocalStorage["results_old"] = JSON.stringify({ timestamp: oldTimestamp });
    mockLocalStorage["results_recent"] = JSON.stringify({ timestamp: recentTimestamp });
    mockLocalStorage["other_key"] = "value";

    cleanupExpiredStorage(30 * 24 * 60 * 60 * 1000); // 30 days

    expect(global.localStorage.removeItem).toHaveBeenCalledWith("results_old");
    expect(global.localStorage.removeItem).not.toHaveBeenCalledWith("results_recent");
    expect(global.localStorage.removeItem).not.toHaveBeenCalledWith("other_key");
  });

  it("should only cleanup results_ keys", () => {
    const now = Date.now();
    const oldTimestamp = now - 31 * 24 * 60 * 60 * 1000;

    mockLocalStorage["results_old"] = JSON.stringify({ timestamp: oldTimestamp });
    mockLocalStorage["other_old"] = JSON.stringify({ timestamp: oldTimestamp });

    cleanupExpiredStorage(30 * 24 * 60 * 60 * 1000);

    expect(global.localStorage.removeItem).toHaveBeenCalledWith("results_old");
    expect(global.localStorage.removeItem).not.toHaveBeenCalledWith("other_old");
  });

  it("should handle invalid JSON gracefully", () => {
    mockLocalStorage["results_invalid"] = "invalid json";

    expect(() => cleanupExpiredStorage()).not.toThrow();
  });

  it("should handle entries without timestamp", () => {
    mockLocalStorage["results_no_timestamp"] = JSON.stringify({ data: "value" });

    expect(() => cleanupExpiredStorage()).not.toThrow();
    expect(global.localStorage.removeItem).not.toHaveBeenCalledWith("results_no_timestamp");
  });

  it("should return early when window is undefined", () => {
    const originalWindow = global.window;
    // @ts-ignore
    delete global.window;

    expect(() => cleanupExpiredStorage()).not.toThrow();

    global.window = originalWindow;
  });

  it("should use default maxAge of 30 days", () => {
    const now = Date.now();
    const oldTimestamp = now - 31 * 24 * 60 * 60 * 1000;

    mockLocalStorage["results_old"] = JSON.stringify({ timestamp: oldTimestamp });

    cleanupExpiredStorage(); // Use default

    expect(global.localStorage.removeItem).toHaveBeenCalledWith("results_old");
  });

  it("should handle errors during cleanup", () => {
    global.localStorage.getItem = vi.fn(() => {
      throw new Error("Storage error");
    });

    expect(() => cleanupExpiredStorage()).not.toThrow();
  });
});

describe("StorageError", () => {
  it("should create error with code", () => {
    const error = new StorageError("Test error", "QUOTA_EXCEEDED");
    expect(error.message).toBe("Test error");
    expect(error.code).toBe("QUOTA_EXCEEDED");
    expect(error.name).toBe("StorageError");
  });

  it("should support all error codes", () => {
    const codes = ["QUOTA_EXCEEDED", "DISABLED", "INVALID_DATA", "PARSE_ERROR", "UNKNOWN"] as const;
    codes.forEach((code) => {
      const error = new StorageError("Test", code);
      expect(error.code).toBe(code);
    });
  });
});
