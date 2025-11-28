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
  let mockGetItem: ReturnType<typeof vi.fn>;
  let mockSetItem: ReturnType<typeof vi.fn>;
  let mockRemoveItem: ReturnType<typeof vi.fn>;

  beforeEach(() => {
    mockLocalStorage = {};
    originalLocalStorage = global.localStorage;

    // Ensure window is defined (needed for isStorageAvailable check)
    if (typeof global.window === "undefined") {
      global.window = {} as any;
    }

    // Create spies that actually interact with mockLocalStorage
    // Note: isStorageAvailable() will try to set/remove "__storage_test__"
    mockGetItem = vi.fn((key: string) => {
      return mockLocalStorage[key] || null;
    });
    mockSetItem = vi.fn((key: string, value: string) => {
      mockLocalStorage[key] = value;
    });
    mockRemoveItem = vi.fn((key: string) => {
      delete mockLocalStorage[key];
    });

    const localStorageMock = {
      getItem: mockGetItem,
      setItem: mockSetItem,
      removeItem: mockRemoveItem,
      clear: vi.fn(() => {
        mockLocalStorage = {};
      }),
      key: vi.fn((index: number) => Object.keys(mockLocalStorage)[index] || null),
      get length() {
        return Object.keys(mockLocalStorage).length;
      },
    };
    global.localStorage = localStorageMock as any;

    Object.defineProperty(global.localStorage, "length", {
      get: () => Object.keys(mockLocalStorage).length,
      configurable: true,
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
      // Set value before creating storage
      mockLocalStorage["test"] = '{"key": "value"}';
      const storage = createSafeStorage();
      // isStorageAvailable() may have been called during createSafeStorage, but shouldn't affect our key
      // Re-set the value to be safe
      mockLocalStorage["test"] = '{"key": "value"}';
      const result = storage.getItem("test");
      expect(result).toBe('{"key": "value"}');
      expect(mockGetItem).toHaveBeenCalledWith("test");
    });

    it("should return null for non-existent key", () => {
      const storage = createSafeStorage();
      const result = storage.getItem("nonexistent");
      expect(result).toBeNull();
    });

    it("should detect and remove corrupted data", () => {
      mockLocalStorage["test"] = "[object Object]";
      const storage = createSafeStorage();
      // Re-set corrupted data after storage creation
      mockLocalStorage["test"] = "[object Object]";
      const result = storage.getItem("test");
      expect(result).toBeNull();
      // removeItem should be called to clean up corrupted data
      expect(mockRemoveItem).toHaveBeenCalledWith("test");
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
      expect(mockSetItem).toHaveBeenCalledWith("test", '{"key": "value"}');
      expect(mockLocalStorage["test"]).toBe('{"key": "value"}');
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
      const quotaError = Object.create(DOMException.prototype);
      Object.defineProperty(quotaError, "name", { value: "QuotaExceededError", writable: false });
      Object.defineProperty(quotaError, "message", { value: "Quota exceeded", writable: false });
      Object.defineProperty(quotaError, "code", { value: 22, writable: false });

      // isStorageAvailable() will call setItem with "__storage_test__" first
      // Then our actual call will throw the error
      mockSetItem.mockImplementation((key: string, value: string) => {
        if (key === "__storage_test__") {
          // Allow isStorageAvailable() to work
          mockLocalStorage[key] = value;
          return;
        }
        // Throw error for actual setItem calls
        throw quotaError;
      });

      const storage = createSafeStorage();
      expect(() => storage.setItem("test", "value")).toThrow(StorageError);
      expect(() => storage.setItem("test", "value")).toThrow("quota exceeded");
    });

    it("should throw StorageError on security error", () => {
      const securityError = Object.create(DOMException.prototype);
      Object.defineProperty(securityError, "name", { value: "SecurityError", writable: false });
      Object.defineProperty(securityError, "message", { value: "Security error", writable: false });
      Object.defineProperty(securityError, "code", { value: 18, writable: false });

      // isStorageAvailable() will call setItem with "__storage_test__" first
      mockSetItem.mockImplementation((key: string, value: string) => {
        if (key === "__storage_test__") {
          // Allow isStorageAvailable() to work
          mockLocalStorage[key] = value;
          return;
        }
        // Throw error for actual setItem calls
        throw securityError;
      });

      const storage = createSafeStorage();
      expect(() => storage.setItem("test", "value")).toThrow(StorageError);
      expect(() => storage.setItem("test", "value")).toThrow("disabled");
    });

    it("should retry after cleanup on quota error", () => {
      let callCount = 0;
      const quotaError = Object.create(DOMException.prototype);
      Object.defineProperty(quotaError, "name", { value: "QuotaExceededError", writable: false });
      Object.defineProperty(quotaError, "message", { value: "Quota exceeded", writable: false });
      Object.defineProperty(quotaError, "code", { value: 22, writable: false });

      mockSetItem.mockImplementation((key: string, value: string) => {
        if (key === "__storage_test__") {
          // Allow isStorageAvailable() to work
          mockLocalStorage[key] = value;
          return;
        }
        callCount++;
        if (callCount === 1) {
          throw quotaError;
        }
        // Second call succeeds
        mockLocalStorage["test"] = "value";
      });

      // Mock getAllKeys to return some keys for cleanup
      const mockKey = vi.fn((index: number) => {
        const keys = ["results_1", "results_2"];
        return keys[index] || null;
      });
      global.localStorage.key = mockKey;
      Object.defineProperty(global.localStorage, "length", {
        get: () => 2,
      });

      const storage = createSafeStorage();
      storage.setItem("test", "value");

      // Should have tried cleanup and retried
      // isStorageAvailable() calls setItem once, then our call (throws), then retry (succeeds) = 3 total
      expect(mockSetItem).toHaveBeenCalledTimes(3);
    });

    it("should warn about large values", () => {
      const consoleWarn = vi.spyOn(console, "warn").mockImplementation(() => {});
      // Use a value that's >90% of the estimated 5MB limit (5MB * 0.9 = 4.5MB)
      // We need >4.5MB to trigger the warning - use 4.6MB to be safe
      const largeValue = "x".repeat(Math.floor(5 * 1024 * 1024 * 0.92));

      const storage = createSafeStorage();
      storage.setItem("test", largeValue);

      // The warning should be called with the size message
      // Note: getAvailableStorageSpace() returns 5MB when window is defined
      // Check if any warning was called with "large" in it
      const warnCalls = consoleWarn.mock.calls;
      const hasLargeWarning = warnCalls.some(
        (call) => call[0] && typeof call[0] === "string" && call[0].includes("large"),
      );
      expect(hasLargeWarning).toBe(true);

      consoleWarn.mockRestore();
    });
  });

  describe("removeItem", () => {
    it("should remove item successfully", () => {
      mockLocalStorage["test"] = "value";
      const storage = createSafeStorage();
      storage.removeItem("test");
      expect(mockRemoveItem).toHaveBeenCalledWith("test");
      expect(mockLocalStorage["test"]).toBeUndefined();
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
      expect(mockRemoveItem).not.toHaveBeenCalled();

      global.window = originalWindow;
    });
  });
});

describe("cleanupExpiredStorage", () => {
  let mockLocalStorage: Record<string, string>;
  let originalLocalStorage: Storage | undefined;

  let mockGetItem: ReturnType<typeof vi.fn>;
  let mockSetItem: ReturnType<typeof vi.fn>;
  let mockRemoveItem: ReturnType<typeof vi.fn>;
  let mockKey: ReturnType<typeof vi.fn>;

  beforeEach(() => {
    mockLocalStorage = {};
    originalLocalStorage = global.localStorage;

    mockGetItem = vi.fn((key: string) => mockLocalStorage[key] || null);
    mockSetItem = vi.fn();
    mockRemoveItem = vi.fn((key: string) => {
      delete mockLocalStorage[key];
    });
    mockKey = vi.fn((index: number) => Object.keys(mockLocalStorage)[index] || null);

    global.localStorage = {
      getItem: mockGetItem,
      setItem: mockSetItem,
      removeItem: mockRemoveItem,
      clear: vi.fn(),
      key: mockKey,
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

    expect(mockRemoveItem).toHaveBeenCalledWith("results_old");
    expect(mockRemoveItem).not.toHaveBeenCalledWith("results_recent");
    expect(mockRemoveItem).not.toHaveBeenCalledWith("other_key");
  });

  it("should only cleanup results_ keys", () => {
    const now = Date.now();
    const oldTimestamp = now - 31 * 24 * 60 * 60 * 1000;

    mockLocalStorage["results_old"] = JSON.stringify({ timestamp: oldTimestamp });
    mockLocalStorage["other_old"] = JSON.stringify({ timestamp: oldTimestamp });

    cleanupExpiredStorage(30 * 24 * 60 * 60 * 1000);

    expect(mockRemoveItem).toHaveBeenCalledWith("results_old");
    expect(mockRemoveItem).not.toHaveBeenCalledWith("other_old");
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

    expect(mockRemoveItem).toHaveBeenCalledWith("results_old");
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
