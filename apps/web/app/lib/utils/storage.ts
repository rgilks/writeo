/**
 * Shared storage utilities for Zustand stores
 * Provides error handling, quota management, and cleanup
 */

import { StateStorage } from "zustand/middleware";

export class StorageError extends Error {
  constructor(
    message: string,
    public code: "QUOTA_EXCEEDED" | "DISABLED" | "INVALID_DATA" | "PARSE_ERROR" | "UNKNOWN",
  ) {
    super(message);
    this.name = "StorageError";
  }
}

/**
 * Get available storage space (approximate)
 * Returns bytes available, or null if unable to determine
 * Note: Most browsers have ~5-10MB limit for localStorage
 */
export function getAvailableStorageSpace(): number | null {
  if (typeof window === "undefined") {
    return null;
  }
  // Conservative 5MB estimate (actual limit varies by browser)
  return 5 * 1024 * 1024;
}

function getValueSize(value: string): number {
  return new Blob([value]).size;
}

function isCorruptedData(value: string): boolean {
  return value === "[object Object]" || (value.startsWith("[object ") && value.endsWith("]"));
}

function getAllKeys(excludeKey?: string): string[] {
  const keys: string[] = [];
  for (let i = 0; i < localStorage.length; i++) {
    const key = localStorage.key(i);
    if (key && key !== excludeKey) {
      keys.push(key);
    }
  }
  return keys;
}

function isQuotaExceededError(error: unknown): boolean {
  if (error instanceof DOMException) {
    return error.name === "QuotaExceededError" || error.code === 22;
  }
  return false;
}

function isSecurityError(error: unknown): boolean {
  if (error instanceof DOMException) {
    return error.name === "SecurityError" || error.code === 18;
  }
  return false;
}

function isStorageAvailable(): boolean {
  if (typeof window === "undefined") return false;

  try {
    const test = "__storage_test__";
    localStorage.setItem(test, test);
    localStorage.removeItem(test);
    return true;
  } catch {
    return false;
  }
}

/**
 * Enhanced base storage with error handling and quota management
 */
export function createSafeStorage(): StateStorage {
  return {
    getItem: (name: string): string | null => {
      if (typeof window === "undefined" || !isStorageAvailable()) {
        return null;
      }

      try {
        const value = localStorage.getItem(name);
        if (!value) return null;

        if (isCorruptedData(value)) {
          console.warn(`Corrupted storage data detected for ${name}, clearing it`);
          localStorage.removeItem(name);
          return null;
        }

        return value;
      } catch (error) {
        console.error(`Failed to get item ${name} from localStorage:`, error);
        return null;
      }
    },

    setItem: (name: string, value: string): void => {
      if (typeof window === "undefined" || !isStorageAvailable()) {
        console.warn("localStorage is not available");
        return;
      }

      try {
        const valueSize = getValueSize(value);
        const availableSpace = getAvailableStorageSpace();

        if (availableSpace !== null && valueSize > availableSpace * 0.9) {
          console.warn(`Storage value for ${name} is large (${valueSize} bytes), may exceed quota`);
        }

        localStorage.setItem(name, value);
      } catch (error) {
        if (isQuotaExceededError(error)) {
          console.error(`Storage quota exceeded for ${name}:`, error);
          cleanupOldStorageData(name);
          try {
            localStorage.setItem(name, value);
          } catch (retryError) {
            console.error(`Failed to set item ${name} after cleanup:`, retryError);
            throw new StorageError(
              "Storage quota exceeded. Please clear some data or use incognito mode.",
              "QUOTA_EXCEEDED",
            );
          }
        } else if (isSecurityError(error)) {
          console.warn(`Storage disabled for ${name}:`, error);
          throw new StorageError("Storage is disabled (e.g., private browsing mode)", "DISABLED");
        } else {
          console.error(`Failed to set item ${name} in localStorage:`, error);
          throw error;
        }
      }
    },

    removeItem: (name: string): void => {
      if (typeof window === "undefined") return;
      if (!isStorageAvailable()) return;

      try {
        localStorage.removeItem(name);
      } catch (error) {
        console.error(`Failed to remove item ${name} from localStorage:`, error);
      }
    },
  };
}

function cleanupOldStorageData(excludeKey?: string): void {
  if (typeof window === "undefined") return;

  try {
    const keys = getAllKeys(excludeKey);
    const resultsKeys = keys.filter((k) => k.startsWith("results_"));

    if (resultsKeys.length > 0) {
      const toRemove = Math.max(1, Math.floor(resultsKeys.length * 0.1));
      for (let i = 0; i < toRemove; i++) {
        localStorage.removeItem(resultsKeys[i]);
      }
      console.log(`Cleaned up ${toRemove} old results entries`);
    }
  } catch (error) {
    console.error("Failed to cleanup old storage data:", error);
  }
}

/**
 * Cleanup expired storage entries
 * Checks for entries with timestamps and removes old ones
 * @param maxAgeMs - Maximum age in milliseconds (default: 30 days)
 */
export function cleanupExpiredStorage(maxAgeMs: number = 30 * 24 * 60 * 60 * 1000): void {
  if (typeof window === "undefined") return;

  try {
    const now = Date.now();
    const keysToRemove: string[] = [];
    const keys = getAllKeys();

    for (const key of keys) {
      if (!key.startsWith("results_")) continue;

      try {
        const value = localStorage.getItem(key);
        if (!value) continue;

        const parsed = JSON.parse(value);
        if (parsed?.timestamp && typeof parsed.timestamp === "number") {
          if (now - parsed.timestamp > maxAgeMs) {
            keysToRemove.push(key);
          }
        }
      } catch {
        // Skip invalid entries
      }
    }

    keysToRemove.forEach((key) => localStorage.removeItem(key));
    if (keysToRemove.length > 0) {
      console.log(`Cleaned up ${keysToRemove.length} expired storage entries`);
    }
  } catch (error) {
    console.error("Failed to cleanup expired storage:", error);
  }
}
