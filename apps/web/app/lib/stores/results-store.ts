/**
 * Zustand store for managing assessment results in localStorage
 * Replaces direct localStorage access with centralized, type-safe storage
 */

import { create } from "zustand";
import { devtools, persist, createJSONStorage } from "zustand/middleware";
import { immer } from "zustand/middleware/immer";
import type { AssessmentResults } from "@writeo/shared";
import { createSafeStorage, cleanupExpiredStorage } from "../utils/storage";

interface StoredResult {
  results: AssessmentResults;
  timestamp: number;
  // parentSubmissionId removed - it's already in results.meta.parentSubmissionId
}

interface ResultsStore {
  // Results by submission ID
  results: Record<string, StoredResult>;

  // Actions
  setResult: (submissionId: string, results: AssessmentResults) => void;
  getResult: (submissionId: string) => AssessmentResults | null;
  getParentSubmissionId: (submissionId: string) => string | null;
  removeResult: (submissionId: string) => void;
  clearAllResults: () => void;
  cleanupOldResults: (maxAgeMs?: number) => void;

  // Computed
  getAllSubmissionIds: () => string[];
  getResultsCount: () => number;
}

const STORAGE_KEY = "writeo-results-store";

export const useResultsStore = create<ResultsStore>()(
  devtools(
    persist(
      immer((set, get) => ({
        results: {},

        setResult: (submissionId, results) => {
          set((state) => {
            state.results[submissionId] = {
              results,
              timestamp: Date.now(),
            };
          });
        },

        getResult: (submissionId) => {
          const stored = get().results[submissionId];
          return stored?.results || null;
        },

        getParentSubmissionId: (submissionId) => {
          // Read parentSubmissionId from results.meta instead of separate storage
          const stored = get().results[submissionId];
          return (stored?.results?.meta?.parentSubmissionId as string | undefined) || null;
        },

        removeResult: (submissionId) => {
          set((state) => {
            delete state.results[submissionId];
          });
        },

        clearAllResults: () => {
          set((state) => {
            state.results = {};
          });
        },

        cleanupOldResults: (maxAgeMs = 30 * 24 * 60 * 60 * 1000) => {
          // 30 days default
          const now = Date.now();
          set((state) => {
            Object.keys(state.results).forEach((submissionId) => {
              const stored = state.results[submissionId];
              if (stored && now - stored.timestamp > maxAgeMs) {
                delete state.results[submissionId];
              }
            });
          });
          // Also cleanup direct localStorage entries for backwards compatibility
          cleanupExpiredStorage(maxAgeMs);
        },

        getAllSubmissionIds: () => {
          return Object.keys(get().results);
        },

        getResultsCount: () => {
          return Object.keys(get().results).length;
        },
      })),
      {
        name: STORAGE_KEY,
        storage: createJSONStorage(() => createSafeStorage()),
        // Cleanup old results on rehydration
        onRehydrateStorage: () => (state) => {
          if (state && typeof window !== "undefined") {
            // Cleanup results older than 30 days
            state.cleanupOldResults(30 * 24 * 60 * 60 * 1000);
          }
        },
      }
    ),
    { name: "ResultsStore" }
  )
);

// Initialize cleanup on module load (runs once per page load)
if (typeof window !== "undefined") {
  // Cleanup old results on app start
  setTimeout(() => {
    useResultsStore.getState().cleanupOldResults(30 * 24 * 60 * 60 * 1000);
  }, 1000); // Wait 1s after page load
}
