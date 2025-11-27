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
 */
export function getAvailableStorageSpace(): number | null {
  if (
    typeof window === "undefined" ||
    !("storage" in navigator) ||
    !("estimate" in navigator.storage)
  ) {
    return null;
  }

  try {
    // This is async, but we'll use a sync approximation
    // Most browsers have ~5-10MB limit for localStorage
    return 5 * 1024 * 1024; // 5MB conservative estimate
  } catch {
    return null;
  }
}

/**
 * Get approximate size of a value in bytes
 */
function getValueSize(value: string): number {
  return new Blob([value]).size;
}

/**
 * Check if storage is available
 */
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
      if (typeof window === "undefined") return null;
      if (!isStorageAvailable()) return null;

      try {
        const value = localStorage.getItem(name);
        if (!value) return null;

        // Check for corrupted data
        if (value === "[object Object]" || (value.startsWith("[object ") && value.endsWith("]"))) {
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
      if (typeof window === "undefined") return;
      if (!isStorageAvailable()) {
        console.warn("localStorage is not available");
        return;
      }

      try {
        // Check value size
        const valueSize = getValueSize(value);
        const availableSpace = getAvailableStorageSpace();

        if (availableSpace !== null && valueSize > availableSpace * 0.9) {
          // If value is >90% of available space, warn but still try
          console.warn(`Storage value for ${name} is large (${valueSize} bytes), may exceed quota`);
        }

        localStorage.setItem(name, value);
      } catch (error) {
        if (error instanceof DOMException) {
          if (error.name === "QuotaExceededError" || error.code === 22) {
            console.error(`Storage quota exceeded for ${name}:`, error);
            // Try to clean up old data
            cleanupOldStorageData(name);
            // Try again once
            try {
              localStorage.setItem(name, value);
            } catch (retryError) {
              console.error(`Failed to set item ${name} after cleanup:`, retryError);
              throw new StorageError(
                `Storage quota exceeded. Please clear some data or use incognito mode.`,
                "QUOTA_EXCEEDED",
              );
            }
          } else if (error.name === "SecurityError" || error.code === 18) {
            console.warn(`Storage disabled for ${name}:`, error);
            throw new StorageError("Storage is disabled (e.g., private browsing mode)", "DISABLED");
          }
        }
        console.error(`Failed to set item ${name} in localStorage:`, error);
        throw error;
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

/**
 * Cleanup old storage data when quota is exceeded
 * Removes oldest entries based on timestamp if available
 */
function cleanupOldStorageData(excludeKey?: string): void {
  if (typeof window === "undefined") return;

  try {
    // Get all keys
    const keys: string[] = [];
    for (let i = 0; i < localStorage.length; i++) {
      const key = localStorage.key(i);
      if (key && key !== excludeKey) {
        keys.push(key);
      }
    }

    // Try to remove oldest results_* entries first (they're largest)
    const resultsKeys = keys.filter((k) => k.startsWith("results_"));
    if (resultsKeys.length > 0) {
      // Remove oldest 10% of results
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
 */
export function cleanupExpiredStorage(maxAgeMs: number = 30 * 24 * 60 * 60 * 1000): void {
  // 30 days default
  if (typeof window === "undefined") return;

  try {
    const now = Date.now();
    const keysToRemove: string[] = [];

    for (let i = 0; i < localStorage.length; i++) {
      const key = localStorage.key(i);
      if (!key) continue;

      // Only cleanup results_* entries for now
      if (key.startsWith("results_")) {
        try {
          const value = localStorage.getItem(key);
          if (value) {
            const parsed = JSON.parse(value);
            // Check if it has a timestamp
            if (parsed?.timestamp && now - parsed.timestamp > maxAgeMs) {
              keysToRemove.push(key);
            }
          }
        } catch {
          // Skip invalid entries
        }
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
