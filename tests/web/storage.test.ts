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
  let setItemSpy: ReturnType<typeof vi.spyOn>;
  let getItemSpy: ReturnType<typeof vi.spyOn>;
  let removeItemSpy: ReturnType<typeof vi.spyOn>;
  let clearSpy: ReturnType<typeof vi.spyOn>;

  beforeEach(() => {
    mockLocalStorage = {};

    // Ensure window and localStorage exist
    if (typeof global.window === "undefined") {
      global.window = {} as any;
    }
    if (!global.window.localStorage) {
      global.window.localStorage = {
        getItem: (key: string) => mockLocalStorage[key] || null,
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
        length: 0,
      } as any;
    }

    // Sync global.localStorage with window.localStorage
    global.localStorage = global.window.localStorage;

    // Spy on the methods
    setItemSpy = vi.spyOn(global.localStorage, "setItem").mockImplementation((key, value) => {
      console.log(`[DEBUG] setItemSpy called with ${key}`);
      mockLocalStorage[key] = value;
    });
    getItemSpy = vi.spyOn(global.localStorage, "getItem").mockImplementation((key) => {
      return mockLocalStorage[key] || null;
    });
    removeItemSpy = vi.spyOn(global.localStorage, "removeItem").mockImplementation((key) => {
      console.log(`[DEBUG] removeItemSpy called with ${key}`);
      delete mockLocalStorage[key];
    });
    clearSpy = vi.spyOn(global.localStorage, "clear").mockImplementation(() => {
      mockLocalStorage = {};
    });

    // Mock length property
    Object.defineProperty(global.localStorage, "length", {
      get: () => Object.keys(mockLocalStorage).length,
      configurable: true,
    });

    // Mock key method
    vi.spyOn(global.localStorage, "key").mockImplementation((index) => {
      return Object.keys(mockLocalStorage)[index] || null;
    });
  });

  afterEach(() => {
    vi.restoreAllMocks();
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
      expect(getItemSpy).toHaveBeenCalledWith("test");
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
      expect(removeItemSpy).toHaveBeenCalledWith("test");
    });

    it("should handle localStorage errors gracefully", () => {
      getItemSpy.mockImplementation(() => {
        throw new Error("Storage error");
      });

      const storage = createSafeStorage();
      const result = storage.getItem("test");
      expect(result).toBeNull();
    });
  });

  describe("setItem", () => {
    it("debug isStorageAvailable", () => {
      const isAvailable = () => {
        if (typeof window === "undefined") return false;
        try {
          const test = "__storage_test__";
          localStorage.setItem(test, test);
          localStorage.removeItem(test);
          return true;
        } catch (e) {
          console.log("isStorageAvailable threw:", e);
          return false;
        }
      };
      expect(isAvailable()).toBe(true);
    });

    it("should store value successfully", () => {
      const storage = createSafeStorage();
      storage.setItem("test", '{"key": "value"}');
      expect(setItemSpy).toHaveBeenCalledWith("test", '{"key": "value"}');
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
    });

    it("should throw StorageError on quota exceeded", () => {
      const quotaError = new DOMException("Quota exceeded", "QuotaExceededError");

      setItemSpy.mockImplementation((key: string, value: string) => {
        if (key === "__storage_test__") {
          mockLocalStorage[key] = value;
          return;
        }
        throw quotaError;
      });

      const storage = createSafeStorage();
      expect(() => storage.setItem("test", "value")).toThrow(StorageError);
      expect(() => storage.setItem("test", "value")).toThrow("quota exceeded");
    });

    it("should throw StorageError on security error", () => {
      const securityError = new DOMException("Security error", "SecurityError");

      setItemSpy.mockImplementation((key: string, value: string) => {
        if (key === "__storage_test__") {
          mockLocalStorage[key] = value;
          return;
        }
        throw securityError;
      });

      const storage = createSafeStorage();
      expect(() => storage.setItem("test", "value")).toThrow(StorageError);
      expect(() => storage.setItem("test", "value")).toThrow("disabled");
    });

    it("should retry after cleanup on quota error", () => {
      let callCount = 0;
      const quotaError = new DOMException("Quota exceeded", "QuotaExceededError");

      setItemSpy.mockImplementation((key: string, value: string) => {
        if (key === "__storage_test__") {
          mockLocalStorage[key] = value;
          return;
        }
        callCount++;
        if (callCount === 1) {
          throw quotaError;
        }
        mockLocalStorage["test"] = "value";
      });

      // Mock getAllKeys to return some keys for cleanup
      vi.spyOn(global.localStorage, "key").mockImplementation((index) => {
        const keys = ["results_1", "results_2"];
        return keys[index] || null;
      });
      Object.defineProperty(global.localStorage, "length", {
        get: () => 2,
        configurable: true,
      });

      const storage = createSafeStorage();
      storage.setItem("test", "value");

      // isStorageAvailable calls setItem once (success)
      // storage.setItem calls setItem (fail)
      // retry calls setItem (success)
      // Total at least 3 calls
      expect(setItemSpy.mock.calls.length).toBeGreaterThanOrEqual(3);
    });

    it("should warn about large values", () => {
      const consoleWarn = vi.spyOn(console, "warn").mockImplementation(() => {});
      const largeValue = "x".repeat(Math.floor(5 * 1024 * 1024 * 0.92));

      const storage = createSafeStorage();
      storage.setItem("test", largeValue);

      const warnCalls = consoleWarn.mock.calls;
      const hasLargeWarning = warnCalls.some(
        (call) => call[0] && typeof call[0] === "string" && call[0].includes("large"),
      );
      expect(hasLargeWarning).toBe(true);
    });
  });

  describe("removeItem", () => {
    it("should remove item successfully", () => {
      mockLocalStorage["test"] = "value";
      const storage = createSafeStorage();
      storage.removeItem("test");
      // removeItem is called for __storage_test__ and then for "test"
      expect(removeItemSpy).toHaveBeenCalledWith("test");
      expect(mockLocalStorage["test"]).toBeUndefined();
    });

    it("should handle errors gracefully", () => {
      removeItemSpy.mockImplementation(() => {
        throw new Error("Remove error");
      });

      const storage = createSafeStorage();
      expect(() => storage.removeItem("test")).not.toThrow();
    });

    it("should return early when window is undefined", () => {
      const originalWindow = global.window;
      // @ts-ignore
      delete global.window;

      const storage = createSafeStorage();
      storage.removeItem("test");
      expect(() => storage.removeItem("test")).not.toThrow();

      global.window = originalWindow;
    });
  });
});

describe("cleanupExpiredStorage", () => {
  let mockLocalStorage: Record<string, string>;
  let removeItemSpy: ReturnType<typeof vi.spyOn>;

  beforeEach(() => {
    mockLocalStorage = {};

    if (typeof global.window === "undefined") {
      global.window = {} as any;
    }
    if (!global.window.localStorage) {
      global.window.localStorage = {
        getItem: (key: string) => mockLocalStorage[key] || null,
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
        length: 0,
      } as any;
    }
    global.localStorage = global.window.localStorage;

    vi.spyOn(global.localStorage, "getItem").mockImplementation(
      (key) => mockLocalStorage[key] || null,
    );
    removeItemSpy = vi.spyOn(global.localStorage, "removeItem").mockImplementation((key) => {
      delete mockLocalStorage[key];
    });
    vi.spyOn(global.localStorage, "key").mockImplementation(
      (index) => Object.keys(mockLocalStorage)[index] || null,
    );
    Object.defineProperty(global.localStorage, "length", {
      get: () => Object.keys(mockLocalStorage).length,
      configurable: true,
    });
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("should cleanup expired entries", () => {
    const now = Date.now();
    const oldTimestamp = now - 31 * 24 * 60 * 60 * 1000;
    const recentTimestamp = now - 10 * 24 * 60 * 60 * 1000;

    mockLocalStorage["results_old"] = JSON.stringify({ timestamp: oldTimestamp });
    mockLocalStorage["results_recent"] = JSON.stringify({ timestamp: recentTimestamp });
    mockLocalStorage["other_key"] = "value";

    cleanupExpiredStorage(30 * 24 * 60 * 60 * 1000);

    expect(removeItemSpy).toHaveBeenCalledWith("results_old");
    expect(removeItemSpy).not.toHaveBeenCalledWith("results_recent");
    expect(removeItemSpy).not.toHaveBeenCalledWith("other_key");
  });

  it("should only cleanup results_ keys", () => {
    const now = Date.now();
    const oldTimestamp = now - 31 * 24 * 60 * 60 * 1000;

    mockLocalStorage["results_old"] = JSON.stringify({ timestamp: oldTimestamp });
    mockLocalStorage["other_old"] = JSON.stringify({ timestamp: oldTimestamp });

    cleanupExpiredStorage(30 * 24 * 60 * 60 * 1000);

    expect(removeItemSpy).toHaveBeenCalledWith("results_old");
    expect(removeItemSpy).not.toHaveBeenCalledWith("other_old");
  });

  it("should handle invalid JSON gracefully", () => {
    mockLocalStorage["results_invalid"] = "invalid json";
    expect(() => cleanupExpiredStorage()).not.toThrow();
  });

  it("should handle entries without timestamp", () => {
    mockLocalStorage["results_no_timestamp"] = JSON.stringify({ data: "value" });
    expect(() => cleanupExpiredStorage()).not.toThrow();
    expect(removeItemSpy).not.toHaveBeenCalledWith("results_no_timestamp");
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

    cleanupExpiredStorage();

    expect(removeItemSpy).toHaveBeenCalledWith("results_old");
  });

  it("should handle errors during cleanup", () => {
    vi.spyOn(global.localStorage, "getItem").mockImplementation(() => {
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
