/**
 * Unit tests for error logger utility
 */

import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { errorLogger } from "../../apps/web/app/lib/utils/error-logger";

describe("ErrorLogger", () => {
  let mockLocalStorage: Record<string, string>;
  let originalLocalStorage: Storage | undefined;
  let originalWindow: Window | undefined;

  beforeEach(() => {
    mockLocalStorage = {};
    originalLocalStorage = global.localStorage;
    originalWindow = global.window;

    // Set window before any operations to ensure errorLogger.enabled is true
    global.window = {
      location: { href: "http://localhost:3000", pathname: "/test" },
      addEventListener: vi.fn(),
      removeEventListener: vi.fn(),
    } as any;

    // Don't set navigator - it's read-only in some environments
    // The code handles undefined navigator gracefully

    global.localStorage = {
      getItem: vi.fn((key: string) => mockLocalStorage[key] || null),
      setItem: vi.fn((key: string, value: string) => {
        mockLocalStorage[key] = value;
      }),
      removeItem: vi.fn((key: string) => {
        delete mockLocalStorage[key];
      }),
      clear: vi.fn(),
      key: vi.fn(),
      length: 0,
    } as any;

    process.env.NODE_ENV = "development";
  });

  afterEach(() => {
    if (originalLocalStorage) {
      global.localStorage = originalLocalStorage;
    }
    if (originalWindow) {
      global.window = originalWindow;
    } else {
      // @ts-ignore
      delete global.window;
    }
    vi.restoreAllMocks();
    vi.clearAllMocks();
  });

  describe("logError", () => {
    it("should log error with context", () => {
      const consoleError = vi.spyOn(console, "error").mockImplementation(() => {});
      const error = new Error("Test error");

      errorLogger.logError(error, { userId: "user-123", page: "/test" });

      expect(consoleError).toHaveBeenCalled();
      const call = consoleError.mock.calls[0];
      expect(call[0]).toContain("Error logged:");
      expect(call[1]).toBeDefined();
      expect(call[1].message).toBe("Test error");
      expect(call[1].context.userId).toBe("user-123");

      consoleError.mockRestore();
    });

    it("should store error in localStorage", () => {
      const error = new Error("Test error");
      errorLogger.logError(error, { userId: "user-123" });

      expect(global.localStorage.setItem).toHaveBeenCalledWith("writeo_errors", expect.any(String));

      const stored = JSON.parse(mockLocalStorage["writeo_errors"]);
      expect(stored).toHaveLength(1);
      expect(stored[0].message).toBe("Test error");
    });

    it("should convert non-Error to Error", () => {
      const consoleError = vi.spyOn(console, "error").mockImplementation(() => {});
      errorLogger.logError("String error", {});

      expect(consoleError).toHaveBeenCalled();
      const call = consoleError.mock.calls[0];
      expect(call[1].message).toBe("String error");

      consoleError.mockRestore();
    });

    it("should limit stored errors to 10", () => {
      for (let i = 0; i < 15; i++) {
        errorLogger.logError(new Error(`Error ${i}`));
      }

      const stored = JSON.parse(mockLocalStorage["writeo_errors"]);
      expect(stored.length).toBe(10);
      expect(stored[0].message).toBe("Error 14"); // Most recent first
    });

    it("should include environment and timestamp in context", () => {
      const error = new Error("Test");
      errorLogger.logError(error);

      const stored = JSON.parse(mockLocalStorage["writeo_errors"]);
      expect(stored[0].context.environment).toBe("development");
      expect(stored[0].context.timestamp).toBeDefined();
    });

    it("should not log when window is undefined", () => {
      // @ts-ignore
      delete global.window;

      const consoleError = vi.spyOn(console, "error").mockImplementation(() => {});
      errorLogger.logError(new Error("Test"));

      expect(consoleError).not.toHaveBeenCalled();
      expect(global.localStorage.setItem).not.toHaveBeenCalled();

      consoleError.mockRestore();
    });
  });

  describe("logWarning", () => {
    it("should log warning with context", () => {
      const consoleWarn = vi.spyOn(console, "warn").mockImplementation(() => {});
      errorLogger.logWarning("Test warning", { page: "/test" });

      expect(consoleWarn).toHaveBeenCalled();
      const call = consoleWarn.mock.calls[0];
      expect(call[0]).toContain("Warning logged:");

      consoleWarn.mockRestore();
    });

    it("should include context in warning", () => {
      const consoleWarn = vi.spyOn(console, "warn").mockImplementation(() => {});
      errorLogger.logWarning("Test warning", { userId: "user-123" });

      const call = consoleWarn.mock.calls[0];
      expect(call[1].context.userId).toBe("user-123");

      consoleWarn.mockRestore();
    });

    it("should not log when window is undefined", () => {
      // @ts-ignore
      delete global.window;

      const consoleWarn = vi.spyOn(console, "warn").mockImplementation(() => {});
      errorLogger.logWarning("Test warning");

      expect(consoleWarn).not.toHaveBeenCalled();

      consoleWarn.mockRestore();
    });
  });

  describe("getStoredErrors", () => {
    it("should return stored errors", () => {
      mockLocalStorage["writeo_errors"] = JSON.stringify([
        { message: "Error 1", context: {} },
        { message: "Error 2", context: {} },
      ]);

      const errors = errorLogger.getStoredErrors();
      expect(errors).toHaveLength(2);
      expect(errors[0].message).toBe("Error 1");
    });

    it("should return empty array when no errors stored", () => {
      const errors = errorLogger.getStoredErrors();
      expect(errors).toEqual([]);
    });

    it("should handle invalid JSON gracefully", () => {
      mockLocalStorage["writeo_errors"] = "invalid json";
      const errors = errorLogger.getStoredErrors();
      expect(errors).toEqual([]);
    });
  });

  describe("clearStoredErrors", () => {
    it("should remove errors from localStorage", () => {
      mockLocalStorage["writeo_errors"] = JSON.stringify([{ message: "Error", context: {} }]);
      errorLogger.clearStoredErrors();

      expect(global.localStorage.removeItem).toHaveBeenCalledWith("writeo_errors");
    });

    it("should handle errors gracefully", () => {
      global.localStorage.removeItem = vi.fn(() => {
        throw new Error("Storage error");
      });

      expect(() => errorLogger.clearStoredErrors()).not.toThrow();
    });
  });
});
