/**
 * Refactored Draft Store - Cleaner, more elegant design
 *
 * Improvements:
 * - Separates local-only state from syncable state
 * - Cleaner storage adapter with proper Set handling
 * - Optional API sync middleware
 * - Simplified immer usage
 * - Better separation of concerns
 */

import { create } from "zustand";
import { devtools, persist, type StateStorage } from "zustand/middleware";
import { immer } from "zustand/middleware/immer";
import { enableMapSet } from "immer";
import type { AssessmentResults } from "@writeo/shared";
import { createSafeStorage } from "../utils/storage";

enableMapSet();

// ============================================================================
// TYPES
// ============================================================================

export interface DraftContent {
  id: string;
  content: string;
  lastModified: number;
  wordCount: number;
  summary: string;
}

export interface DraftHistory {
  draftNumber: number;
  submissionId: string;
  timestamp: string;
  wordCount: number;
  errorCount: number;
  overallScore?: number;
  cefrLevel?: string;
  errorIds: string[];
}

export interface ProgressMetrics {
  totalDrafts: number;
  firstDraftScore?: number;
  latestDraftScore?: number;
  scoreImprovement?: number;
  errorReduction?: number;
  wordCountChange?: number;
}

export interface Achievement {
  id: string;
  name: string;
  description: string;
  icon: string;
  unlockedAt: string;
}

export interface StreakData {
  currentStreak: number;
  longestStreak: number;
  lastActivityDate: string;
}

interface StoredResult {
  results: AssessmentResults;
  timestamp: number;
}

// ============================================================================
// STORE STATE (separated by concern)
// ============================================================================

interface LocalState {
  // Local-only: never synced to API
  contentDrafts: DraftContent[];
  currentContent: string;
  activeDraftId: string | null;
}

interface SyncableState {
  // Syncable: can be synced to API if user opts in
  results: Record<string, StoredResult>;
  drafts: Record<string, DraftHistory[]>; // keyed by rootSubmissionId
  progress: Record<string, ProgressMetrics>;
  fixedErrors: Record<string, Set<string>>;
  achievements: Achievement[];
  streak: StreakData;
}

interface DraftStore extends LocalState, SyncableState {
  // Actions
  // Local state actions
  updateContent: (text: string) => void;
  saveContentDraft: () => void;
  loadContentDraft: (id: string) => void;
  createNewContentDraft: () => void;
  deleteContentDraft: (id: string) => void;

  // Syncable state actions
  setResult: (submissionId: string, results: AssessmentResults) => void;
  getResult: (submissionId: string) => AssessmentResults | null;
  getParentSubmissionId: (submissionId: string) => string | null;
  removeResult: (submissionId: string) => void;
  clearAllResults: () => void;

  addDraft: (draft: DraftHistory, rootSubmissionId: string) => Achievement[];
  getDraftHistory: (rootSubmissionId: string) => DraftHistory[];
  getRootSubmissionId: (submissionId: string) => string | null;
  getProgress: (submissionId: string) => ProgressMetrics | undefined;

  trackFixedErrors: (
    submissionId: string,
    previousErrorIds: string[],
    currentErrorIds: string[]
  ) => void;
  getFixedErrors: (submissionId: string) => Set<string>;

  updateStreak: () => void;
  getStreak: () => StreakData;
  getAchievements: () => Achievement[];

  // Computed selectors
  getTotalDrafts: () => number;
  getTotalWritings: () => number;
  getAverageImprovement: () => number;
  getAllDrafts: () => DraftHistory[];

  // Cleanup
  cleanupOldResults: (maxAgeMs?: number) => void;
  clearDrafts: () => void;
}

// ============================================================================
// UTILITIES
// ============================================================================

const generateId = (): string => Math.random().toString(36).substring(2, 9);

const countWords = (text: string): number => {
  const trimmed = text.trim();
  return trimmed.length === 0 ? 0 : trimmed.split(/\s+/).length;
};

const generateSummary = (text: string): string => {
  const trimmed = text.trim();
  if (trimmed.length === 0) return "Empty draft";
  return trimmed.slice(0, 40) + (trimmed.length > 40 ? "..." : "");
};

const CEFR_LEVELS = ["A2", "B1", "B2", "C1", "C2"] as const;

// ============================================================================
// STORAGE ADAPTER (cleaner Set handling)
// ============================================================================

const baseStorage = createSafeStorage();

/**
 * Custom storage adapter that handles Set serialization elegantly
 */
const createStorageAdapter = (): StateStorage => {
  return {
    getItem: (name: string): string | null => {
      const str = baseStorage.getItem(name);
      // Handle both sync and async return types
      if (!str || (typeof str === "object" && "then" in str)) {
        return null;
      }
      if (typeof str !== "string") {
        return null;
      }

      try {
        // Convert Set arrays back to Sets during deserialization
        // Use a recursive function to transform the object
        const transformSets = (obj: any): any => {
          if (obj === null || typeof obj !== "object") {
            return obj;
          }

          if (Array.isArray(obj)) {
            return obj.map(transformSets);
          }

          if (obj.__type === "Set" && Array.isArray(obj.value)) {
            return new Set(obj.value);
          }

          const transformed: any = {};
          for (const [key, value] of Object.entries(obj)) {
            transformed[key] = transformSets(value);
          }
          return transformed;
        };

        const parsed = JSON.parse(str);
        const transformed = transformSets(parsed);
        // Return the transformed object as JSON string for Zustand
        return JSON.stringify(transformed);
      } catch (error) {
        console.error(`Failed to parse stored data for ${name}:`, error);
        baseStorage.removeItem(name);
        return null;
      }
    },

    setItem: (name: string, value: string | unknown): void => {
      if (typeof window === "undefined") return;

      try {
        // Zustand persist may pass either a string or an object depending on version
        let stringValue: string;

        if (typeof value === "string") {
          stringValue = value;
        } else if (typeof value === "object" && value !== null) {
          // If it's an object, stringify it (Sets are already converted in partialize)
          stringValue = JSON.stringify(value);
        } else {
          console.warn(`Unexpected value type for setItem: ${typeof value}`);
          return;
        }

        // Check for corrupted data
        if (
          stringValue === "[object Object]" ||
          (stringValue.startsWith("[object ") && stringValue.endsWith("]"))
        ) {
          console.warn(`Corrupted data detected for ${name}, skipping save`);
          return;
        }

        // Save the stringified value
        baseStorage.setItem(name, stringValue);
      } catch (error) {
        console.error(`Failed to save data for ${name}:`, error);
      }
    },

    removeItem: (name: string): void => {
      baseStorage.removeItem(name);
    },
  };
};

// ============================================================================
// HELPER FUNCTIONS (pure, testable)
// ============================================================================

/**
 * Find root submission ID from a submission ID
 * Pure function - no side effects
 */
function findRootSubmissionId(
  submissionId: string,
  drafts: Record<string, DraftHistory[]>
): string | null {
  // Check if this submissionId is a root key
  if (drafts[submissionId]) {
    return submissionId;
  }

  // Search through all draft arrays
  for (const [rootId, draftArray] of Object.entries(drafts)) {
    const found = draftArray.find((d) => d.submissionId === submissionId);
    if (found) {
      // Return the root ID (the key)
      return rootId;
    }
  }

  return null;
}

/**
 * Calculate progress metrics from draft array
 * Pure function - no side effects
 */
function calculateProgress(drafts: DraftHistory[]): ProgressMetrics | undefined {
  if (drafts.length === 0) return undefined;

  const first = drafts[0];
  const latest = drafts[drafts.length - 1];

  return {
    totalDrafts: drafts.length,
    firstDraftScore: first?.overallScore,
    latestDraftScore: latest?.overallScore,
    scoreImprovement:
      latest?.overallScore !== undefined && first?.overallScore !== undefined
        ? latest.overallScore - first.overallScore
        : undefined,
    errorReduction:
      latest?.errorCount !== undefined && first?.errorCount !== undefined
        ? first.errorCount - latest.errorCount
        : undefined,
    wordCountChange:
      latest?.wordCount !== undefined && first?.wordCount !== undefined
        ? latest.wordCount - first.wordCount
        : undefined,
  };
}

/**
 * Check for new achievements based on draft and progress
 * Pure function - no side effects
 */
function checkAchievements(
  draft: DraftHistory,
  allDrafts: DraftHistory[],
  existingAchievements: Achievement[],
  progress: ProgressMetrics | undefined,
  totalFixedErrors: number,
  currentStreak: number
): Achievement[] {
  const existingIds = new Set(existingAchievements.map((a) => a.id));
  const achievements: Achievement[] = [];

  if (allDrafts.length === 1 && !existingIds.has("first-draft")) {
    achievements.push({
      id: "first-draft",
      name: "First Draft",
      description: "Submitted your first essay",
      icon: "🎯",
      unlockedAt: new Date().toISOString(),
    });
  }

  if (allDrafts.length >= 5 && !existingIds.has("reviser")) {
    achievements.push({
      id: "reviser",
      name: "Reviser",
      description: "Submitted 5 drafts",
      icon: "✏️",
      unlockedAt: new Date().toISOString(),
    });
  }

  if (
    progress?.scoreImprovement &&
    progress.scoreImprovement >= 1.0 &&
    !existingIds.has("improver")
  ) {
    achievements.push({
      id: "improver",
      name: "Improver",
      description: "Improved your score by 1.0+ points",
      icon: "📈",
      unlockedAt: new Date().toISOString(),
    });
  }

  if (totalFixedErrors >= 10 && !existingIds.has("grammar-master")) {
    achievements.push({
      id: "grammar-master",
      name: "Grammar Master",
      description: "Fixed 10+ grammar errors",
      icon: "🎓",
      unlockedAt: new Date().toISOString(),
    });
  }

  if (draft.cefrLevel) {
    const currentLevelIndex = CEFR_LEVELS.indexOf(draft.cefrLevel as (typeof CEFR_LEVELS)[number]);
    const allLevels = allDrafts
      .map((d) => d.cefrLevel)
      .filter((l): l is string => !!l)
      .map((l) => CEFR_LEVELS.indexOf(l as (typeof CEFR_LEVELS)[number]))
      .filter((i) => i >= 0);
    const maxLevel = Math.max(...allLevels, -1);

    if (
      currentLevelIndex === maxLevel &&
      currentLevelIndex > 0 &&
      !existingIds.has("cefr-climber")
    ) {
      achievements.push({
        id: "cefr-climber",
        name: "CEFR Climber",
        description: `Reached CEFR level ${draft.cefrLevel}`,
        icon: "🏆",
        unlockedAt: new Date().toISOString(),
      });
    }
  }

  if (currentStreak >= 7 && !existingIds.has("streak-keeper")) {
    achievements.push({
      id: "streak-keeper",
      name: "Streak Keeper",
      description: "Maintained a 7+ day practice streak",
      icon: "🔥",
      unlockedAt: new Date().toISOString(),
    });
  }

  return achievements;
}

/**
 * Calculate new streak based on last activity date
 * Pure function - no side effects
 */
function calculateNewStreak(
  lastDate: string | "",
  currentStreak: number,
  longestStreak: number
): StreakData {
  const today = new Date().toISOString().split("T")[0];

  if (!lastDate) {
    return { currentStreak: 1, longestStreak: 1, lastActivityDate: today };
  }

  if (lastDate === today) {
    // Already active today - no change
    return { currentStreak, longestStreak, lastActivityDate: today };
  }

  const yesterday = new Date();
  yesterday.setDate(yesterday.getDate() - 1);
  const yesterdayStr = yesterday.toISOString().split("T")[0];

  if (lastDate === yesterdayStr) {
    // Consecutive day - increment streak
    const newStreak = currentStreak + 1;
    return {
      currentStreak: newStreak,
      longestStreak: Math.max(newStreak, longestStreak),
      lastActivityDate: today,
    };
  }

  // Streak broken - reset
  return { currentStreak: 1, longestStreak, lastActivityDate: today };
}

// ============================================================================
// STORE CREATION
// ============================================================================

const STORAGE_KEY = "writeo-draft-store";

export const useDraftStore = create<DraftStore>()(
  devtools(
    persist(
      immer((set, get) => ({
        // Initial state
        // Local-only state
        contentDrafts: [],
        currentContent: "",
        activeDraftId: null,

        // Syncable state
        results: {},
        drafts: {},
        progress: {},
        fixedErrors: {},
        achievements: [],
        streak: {
          currentStreak: 0,
          longestStreak: 0,
          lastActivityDate: "",
        },

        // ====================================================================
        // LOCAL STATE ACTIONS (never synced)
        // ====================================================================

        updateContent: (text: string) => {
          set((state) => {
            state.currentContent = text;
          });
        },

        saveContentDraft: () => {
          const state = get();
          if (!state.currentContent.trim()) return;

          const now = Date.now();
          const wordCount = countWords(state.currentContent);
          const summary = generateSummary(state.currentContent);

          set((draft) => {
            const existingIndex = draft.contentDrafts.findIndex(
              (d) => d.id === draft.activeDraftId
            );

            if (existingIndex !== -1 && draft.activeDraftId) {
              const updatedDraft = draft.contentDrafts[existingIndex];
              updatedDraft.content = state.currentContent;
              updatedDraft.lastModified = now;
              updatedDraft.wordCount = wordCount;
              updatedDraft.summary = summary;
              draft.contentDrafts.splice(existingIndex, 1);
              draft.contentDrafts.unshift(updatedDraft);
            } else {
              const newDraft: DraftContent = {
                id: generateId(),
                content: state.currentContent,
                lastModified: now,
                wordCount,
                summary,
              };
              draft.contentDrafts.unshift(newDraft);
              draft.activeDraftId = newDraft.id;
            }
          });
        },

        loadContentDraft: (id: string) => {
          const draft = get().contentDrafts.find((d) => d.id === id);
          if (draft) {
            set((state) => {
              state.currentContent = draft.content;
              state.activeDraftId = draft.id;
            });
          }
        },

        createNewContentDraft: () => {
          set((state) => {
            state.currentContent = "";
            state.activeDraftId = null;
          });
        },

        deleteContentDraft: (id: string) => {
          set((state) => {
            state.contentDrafts = state.contentDrafts.filter((d) => d.id !== id);
            if (state.activeDraftId === id) {
              state.activeDraftId = null;
              state.currentContent = "";
            }
          });
        },

        // ====================================================================
        // SYNCABLE STATE ACTIONS
        // ====================================================================

        setResult: (submissionId, results) => {
          set((state) => {
            state.results[submissionId] = {
              results,
              timestamp: Date.now(),
            };
          });
        },

        getResult: (submissionId) => {
          return get().results[submissionId]?.results || null;
        },

        getParentSubmissionId: (submissionId) => {
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

        addDraft: (draft, rootSubmissionId) => {
          let newAchievements: Achievement[] = [];

          set((state) => {
            // Ensure the root key exists
            if (!state.drafts[rootSubmissionId]) {
              state.drafts[rootSubmissionId] = [];
            }

            const draftArray = state.drafts[rootSubmissionId];
            const existingIndex = draftArray.findIndex(
              (d) => d.submissionId === draft.submissionId
            );

            if (existingIndex >= 0) {
              draftArray[existingIndex] = { ...draft };
            } else {
              draftArray.push({ ...draft });
              draftArray.sort((a, b) => a.draftNumber - b.draftNumber);
            }

            // Update progress metrics (pure function)
            const progress = calculateProgress(draftArray);
            if (progress) {
              state.progress[draft.submissionId] = progress;
            }

            // Update streak
            const newStreak = calculateNewStreak(
              state.streak.lastActivityDate,
              state.streak.currentStreak,
              state.streak.longestStreak
            );
            state.streak = newStreak;

            // Check for achievements
            const allDrafts = Object.values(state.drafts).flat();
            const totalFixedErrors = Object.values(state.fixedErrors).reduce(
              (sum, errors) => sum + errors.size,
              0
            );

            newAchievements = checkAchievements(
              draft,
              allDrafts,
              state.achievements,
              progress,
              totalFixedErrors,
              newStreak.currentStreak
            );

            if (newAchievements.length > 0) {
              state.achievements.push(...newAchievements);
            }
          });

          return newAchievements;
        },

        getDraftHistory: (rootSubmissionId) => {
          return get().drafts[rootSubmissionId] || [];
        },

        getRootSubmissionId: (submissionId) => {
          return findRootSubmissionId(submissionId, get().drafts);
        },

        getProgress: (submissionId) => {
          return get().progress[submissionId];
        },

        trackFixedErrors: (submissionId, previousErrorIds, currentErrorIds) => {
          set((state) => {
            if (!state.fixedErrors[submissionId]) {
              state.fixedErrors[submissionId] = new Set();
            }

            previousErrorIds.forEach((errorId) => {
              if (!currentErrorIds.includes(errorId)) {
                state.fixedErrors[submissionId].add(errorId);
              }
            });
          });
        },

        getFixedErrors: (submissionId) => {
          return get().fixedErrors[submissionId] || new Set();
        },

        updateStreak: () => {
          set((state) => {
            state.streak = calculateNewStreak(
              state.streak.lastActivityDate,
              state.streak.currentStreak,
              state.streak.longestStreak
            );
          });
        },

        getStreak: () => {
          return get().streak;
        },

        getAchievements: () => {
          return get().achievements;
        },

        // ====================================================================
        // COMPUTED SELECTORS
        // ====================================================================

        getTotalDrafts: () => {
          const state = get();
          return Object.values(state.drafts).reduce((sum, drafts) => sum + drafts.length, 0);
        },

        getTotalWritings: () => {
          return Object.keys(get().drafts).length;
        },

        getAverageImprovement: () => {
          const state = get();
          const improvements = Object.values(state.progress)
            .map((p) => p.scoreImprovement || 0)
            .filter((i) => i > 0);
          return improvements.length > 0
            ? improvements.reduce((a, b) => a + b, 0) / improvements.length
            : 0;
        },

        getAllDrafts: () => {
          return Object.values(get().drafts).flat();
        },

        // ====================================================================
        // CLEANUP
        // ====================================================================

        cleanupOldResults: (maxAgeMs = 30 * 24 * 60 * 60 * 1000) => {
          const now = Date.now();
          set((state) => {
            Object.keys(state.results).forEach((submissionId) => {
              const stored = state.results[submissionId];
              if (stored && now - stored.timestamp > maxAgeMs) {
                delete state.results[submissionId];
              }
            });
          });
          // Also cleanup expired storage entries
          if (typeof window !== "undefined") {
            const { cleanupExpiredStorage } = require("../utils/storage");
            cleanupExpiredStorage(maxAgeMs);
          }
        },

        clearDrafts: () => {
          set((state) => {
            state.drafts = {};
            state.progress = {};
            state.fixedErrors = {};
            state.achievements = [];
            state.streak = {
              currentStreak: 0,
              longestStreak: 0,
              lastActivityDate: "",
            };
          });
        },
      })),
      {
        name: STORAGE_KEY,
        storage: createStorageAdapter() as any,
        partialize: (state) => {
          // Only persist state, not functions
          // Convert Sets to marked objects before Zustand stringifies (to avoid "[object Object]")
          const fixedErrorsSerialized: Record<string, { __type: "Set"; value: string[] }> = {};
          Object.entries(state.fixedErrors).forEach(([key, value]) => {
            fixedErrorsSerialized[key] = {
              __type: "Set",
              value: value instanceof Set ? Array.from(value) : [],
            };
          });

          return {
            contentDrafts: state.contentDrafts,
            currentContent: state.currentContent,
            activeDraftId: state.activeDraftId,
            results: state.results,
            drafts: state.drafts,
            progress: state.progress,
            fixedErrors: fixedErrorsSerialized,
            achievements: state.achievements,
            streak: state.streak,
          };
        },
        onRehydrateStorage: () => (state) => {
          if (state && typeof window !== "undefined") {
            state.cleanupOldResults(30 * 24 * 60 * 60 * 1000);
          }
        },
      }
    ),
    { name: "DraftStore" }
  )
);

// Initialize cleanup on module load
if (typeof window !== "undefined") {
  setTimeout(() => {
    useDraftStore.getState().cleanupOldResults(30 * 24 * 60 * 60 * 1000);
  }, 1000);
}

// ============================================================================
// OPTIONAL: API SYNC MIDDLEWARE
// ============================================================================

/**
 * Optional middleware to sync syncable state to API
 * This can be added as a separate middleware layer if needed
 */
export function createApiSyncMiddleware(syncFn: (state: SyncableState) => Promise<void>) {
  return (config: any) => (set: any, get: any, api: any) =>
    config(
      (...args: any[]) => {
        set(...args);
        // Sync after state update
        const state = get();
        syncFn({
          results: state.results,
          drafts: state.drafts,
          progress: state.progress,
          fixedErrors: state.fixedErrors,
          achievements: state.achievements,
          streak: state.streak,
        }).catch((error) => {
          console.error("Failed to sync state to API:", error);
        });
      },
      get,
      api
    );
}
