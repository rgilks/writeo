import { create } from "zustand";
import { devtools, persist, StateStorage } from "zustand/middleware";
import { immer } from "zustand/middleware/immer";
import { createSafeStorage } from "../utils/storage";

export interface DraftHistory {
  draftNumber: number;
  submissionId: string;
  timestamp: string;
  wordCount: number;
  errorCount: number;
  overallScore?: number;
  cefrLevel?: string;
  errorIds: string[]; // Track which errors were present in this draft
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
  currentStreak: number; // Consecutive days
  longestStreak: number;
  lastActivityDate: string; // ISO date string
}

interface DraftStore {
  drafts: Record<string, DraftHistory[]>; // Key: parentSubmissionId or submissionId
  progress: Record<string, ProgressMetrics>; // Key: submissionId
  fixedErrors: Record<string, Set<string>>; // Key: submissionId, Value: Set of error IDs that were fixed
  achievements: Achievement[]; // Unlocked achievements
  streak: StreakData; // Daily practice streak

  // Actions
  addDraft: (draft: DraftHistory, parentSubmissionId?: string) => Achievement[];
  getDraftHistory: (submissionId: string) => DraftHistory[];
  getRootSubmissionId: (submissionId: string) => string | null; // Add this line
  getProgress: (submissionId: string) => ProgressMetrics | undefined;
  trackFixedErrors: (
    submissionId: string,
    previousErrorIds: string[],
    currentErrorIds: string[]
  ) => void;
  getFixedErrors: (submissionId: string) => Set<string>;
  updateStreak: () => void;
  getStreak: () => StreakData;
  checkAndUnlockAchievements: (draft: DraftHistory, allDrafts: DraftHistory[]) => Achievement[];
  getAchievements: () => Achievement[];
  clearDrafts: () => void;

  // Computed selectors
  getTotalDrafts: () => number;
  getTotalWritings: () => number;
  getAverageImprovement: () => number;
  getAllDrafts: () => DraftHistory[];
}

const STORAGE_KEY = "writeo-draft-store";

// Custom storage that handles Set serialization/deserialization
// Based on Zustand docs: https://zustand.docs.pmnd.rs/integrations/persisting-store-data#how-do-i-use-it-with-map-and-set
// We need to convert Sets to/from arrays for JSON serialization
const baseStorage = createSafeStorage();

// Custom storage adapter that handles Set conversion
// Zustand persist wraps state in { state: {...}, version: number }
const storageWithSetHandling: StateStorage = {
  getItem: (name: string): string | null => {
    // baseStorage uses synchronous localStorage, so result is always string | null (not Promise)
    const str = baseStorage.getItem(name) as string | null;
    if (!str) return null;

    // Check if the stored value is corrupted (e.g., "[object Object]")
    if (str === "[object Object]" || (str.startsWith("[object ") && str.endsWith("]"))) {
      console.warn("Corrupted draft store data detected, clearing it");
      baseStorage.removeItem(name);
      return null;
    }

    try {
      const parsed = JSON.parse(str);
      // Convert fixedErrors arrays back to Sets
      if (parsed?.state?.fixedErrors) {
        const fixedErrors: Record<string, Set<string>> = {};
        Object.entries(parsed.state.fixedErrors).forEach(([key, value]) => {
          // Handle both array and Set cases (defensive)
          if (Array.isArray(value)) {
            fixedErrors[key] = new Set(value);
          } else if (value instanceof Set) {
            fixedErrors[key] = value;
          } else {
            console.warn(
              `Unexpected fixedErrors value type for key ${key} when loading:`,
              typeof value
            );
            fixedErrors[key] = new Set();
          }
        });
        parsed.state.fixedErrors = fixedErrors;
      }
      return JSON.stringify(parsed);
    } catch (error) {
      console.error("Failed to parse stored draft store, clearing corrupted data:", error);
      // Clear corrupted data to prevent future errors
      baseStorage.removeItem(name);
      return null;
    }
  },
  setItem: (name: string, value: string): void => {
    if (typeof window === "undefined") return;

    try {
      // Handle case where value might already be an object (defensive programming)
      let parsed: any;
      if (typeof value === "string") {
        try {
          parsed = JSON.parse(value);
        } catch (parseError) {
          // If parsing fails, value might be corrupted - try to recover or skip
          console.warn("Failed to parse draft store value, skipping save:", parseError);
          return;
        }
      } else if (typeof value === "object" && value !== null) {
        // If value is already an object, use it directly
        parsed = value;
      } else {
        console.warn("Unexpected value type for draft store:", typeof value);
        return;
      }

      // Convert Sets to arrays for JSON serialization
      if (parsed?.state?.fixedErrors) {
        const fixedErrors: Record<string, string[]> = {};
        Object.entries(parsed.state.fixedErrors).forEach(([key, setValue]) => {
          // Handle both Set and array cases (defensive)
          if (setValue instanceof Set) {
            fixedErrors[key] = Array.from(setValue);
          } else if (Array.isArray(setValue)) {
            fixedErrors[key] = setValue;
          } else {
            console.warn(`Unexpected fixedErrors value type for key ${key}:`, typeof setValue);
          }
        });
        parsed.state.fixedErrors = fixedErrors;
      }

      baseStorage.setItem(name, JSON.stringify(parsed));
    } catch (error) {
      console.error("Failed to save draft store:", error, { valueType: typeof value, value });
      // Don't throw - allow the app to continue functioning even if persistence fails
    }
  },
  removeItem: (name: string): void => {
    baseStorage.removeItem(name);
  },
};

export const useDraftStore = create<DraftStore>()(
  devtools(
    persist(
      immer((set, get) => {
        return {
          drafts: {},
          progress: {},
          fixedErrors: {},
          achievements: [],
          streak: {
            currentStreak: 0,
            longestStreak: 0,
            lastActivityDate: "",
          },

          addDraft: (draft, parentSubmissionId) => {
            let newAchievements: Achievement[] = [];

            set((state) => {
              // Use root submissionId: parentSubmissionId if provided, otherwise submissionId (for draft 1)
              // This is simpler than the old logic and matches how results.meta.parentSubmissionId works
              const key = parentSubmissionId || draft.submissionId;

              // Initialize array if it doesn't exist
              if (!state.drafts[key]) {
                state.drafts[key] = [];
              }

              // Check if this draft already exists (by submissionId)
              const existingIndex = state.drafts[key].findIndex(
                (d) => d.submissionId === draft.submissionId
              );

              if (existingIndex >= 0) {
                // Update existing draft
                state.drafts[key][existingIndex] = { ...draft };
              } else {
                // Add new draft
                state.drafts[key].push({ ...draft });
                // Sort by draft number
                state.drafts[key].sort((a, b) => a.draftNumber - b.draftNumber);
              }

              // Update progress metrics
              // Access through state directly (don't destructure to maintain Immer proxy)
              const draftsArray = state.drafts[key];
              if (draftsArray.length > 0) {
                // Read values directly from state to maintain proxy chain
                const firstDraftScore = draftsArray[0]?.overallScore;
                const latestDraftScore = draftsArray[draftsArray.length - 1]?.overallScore;
                const firstErrorCount = draftsArray[0]?.errorCount;
                const latestErrorCount = draftsArray[draftsArray.length - 1]?.errorCount;
                const firstWordCount = draftsArray[0]?.wordCount;
                const latestWordCount = draftsArray[draftsArray.length - 1]?.wordCount;

                state.progress[draft.submissionId] = {
                  totalDrafts: draftsArray.length,
                  firstDraftScore,
                  latestDraftScore,
                  scoreImprovement:
                    latestDraftScore !== undefined && firstDraftScore !== undefined
                      ? latestDraftScore - firstDraftScore
                      : undefined,
                  errorReduction:
                    latestErrorCount !== undefined && firstErrorCount !== undefined
                      ? firstErrorCount - latestErrorCount
                      : undefined,
                  wordCountChange:
                    latestWordCount !== undefined && firstWordCount !== undefined
                      ? latestWordCount - firstWordCount
                      : undefined,
                };
              }

              // Update streak when draft is added (within immer for atomicity)
              const today = new Date().toISOString().split("T")[0];
              // Read directly from state (reading is safe, but don't destructure for mutation)
              const lastDate = state.streak.lastActivityDate;

              if (!lastDate) {
                state.streak.currentStreak = 1;
                state.streak.lastActivityDate = today;
                state.streak.longestStreak = 1;
              } else if (lastDate !== today) {
                const yesterday = new Date();
                yesterday.setDate(yesterday.getDate() - 1);
                const yesterdayStr = yesterday.toISOString().split("T")[0];

                if (lastDate === yesterdayStr) {
                  state.streak.currentStreak += 1;
                  state.streak.lastActivityDate = today;
                  if (state.streak.currentStreak > state.streak.longestStreak) {
                    state.streak.longestStreak = state.streak.currentStreak;
                  }
                } else {
                  state.streak.currentStreak = 1;
                  state.streak.lastActivityDate = today;
                }
              }

              // Check for new achievements (within immer to access current state)
              // Reading from state is safe - we're creating new arrays/sets for computation
              const allDrafts = Object.values(state.drafts).flat();
              const existingIds = new Set(state.achievements.map((a) => a.id));
              const achievementsToAdd: Achievement[] = [];

              // First Draft achievement
              if (allDrafts.length === 1 && !existingIds.has("first-draft")) {
                achievementsToAdd.push({
                  id: "first-draft",
                  name: "First Draft",
                  description: "Submitted your first essay",
                  icon: "🎯",
                  unlockedAt: new Date().toISOString(),
                });
              }

              // Reviser achievement (5 drafts)
              if (allDrafts.length >= 5 && !existingIds.has("reviser")) {
                achievementsToAdd.push({
                  id: "reviser",
                  name: "Reviser",
                  description: "Submitted 5 drafts",
                  icon: "✏️",
                  unlockedAt: new Date().toISOString(),
                });
              }

              // Improver achievement (score improved by 1.0+)
              // Read directly from state (reading is safe)
              const progressEntry = state.progress[draft.submissionId];
              if (
                progressEntry?.scoreImprovement &&
                progressEntry.scoreImprovement >= 1.0 &&
                !existingIds.has("improver")
              ) {
                achievementsToAdd.push({
                  id: "improver",
                  name: "Improver",
                  description: "Improved your score by 1.0+ points",
                  icon: "📈",
                  unlockedAt: new Date().toISOString(),
                });
              }

              // Grammar Master (fixed 10+ errors total)
              // Reading from state is safe - we're computing a value, not mutating
              const totalFixedErrors = Object.values(state.fixedErrors).reduce(
                (sum, errors) => sum + errors.size,
                0
              );
              if (totalFixedErrors >= 10 && !existingIds.has("grammar-master")) {
                achievementsToAdd.push({
                  id: "grammar-master",
                  name: "Grammar Master",
                  description: "Fixed 10+ grammar errors",
                  icon: "🎓",
                  unlockedAt: new Date().toISOString(),
                });
              }

              // CEFR Climber (reached new CEFR level)
              const cefrLevels = ["A2", "B1", "B2", "C1", "C2"];
              if (draft.cefrLevel) {
                const currentLevelIndex = cefrLevels.indexOf(draft.cefrLevel);
                const allLevels = allDrafts
                  .map((d) => d.cefrLevel)
                  .filter((l): l is string => !!l)
                  .map((l) => cefrLevels.indexOf(l))
                  .filter((i) => i >= 0);
                const maxLevel = Math.max(...allLevels, -1);
                if (
                  currentLevelIndex === maxLevel &&
                  currentLevelIndex > 0 &&
                  !existingIds.has("cefr-climber")
                ) {
                  achievementsToAdd.push({
                    id: "cefr-climber",
                    name: "CEFR Climber",
                    description: `Reached CEFR level ${draft.cefrLevel}`,
                    icon: "🏆",
                    unlockedAt: new Date().toISOString(),
                  });
                }
              }

              // Streak Keeper (7+ day streak)
              if (state.streak.currentStreak >= 7 && !existingIds.has("streak-keeper")) {
                achievementsToAdd.push({
                  id: "streak-keeper",
                  name: "Streak Keeper",
                  description: "Maintained a 7+ day practice streak",
                  icon: "🔥",
                  unlockedAt: new Date().toISOString(),
                });
              }

              // Add new achievements
              if (achievementsToAdd.length > 0) {
                state.achievements.push(...achievementsToAdd);
                newAchievements = achievementsToAdd.map((a) => ({ ...a }));
              }
            });

            // Return new achievements for notification
            // Note: Persistence is handled automatically by persist middleware
            return newAchievements;
          },

          getDraftHistory: (submissionId) => {
            const state = get();
            // First try direct lookup by submissionId (for draft 1)
            if (state.drafts[submissionId]) {
              return state.drafts[submissionId];
            }
            // Try to find in any draft array (for drafts 2+)
            for (const [key, drafts] of Object.entries(state.drafts)) {
              const found = drafts.find((d) => d.submissionId === submissionId);
              if (found) {
                // Return all drafts in the same group (key is the root submission ID)
                return drafts;
              }
            }
            return [];
          },

          // Add a new helper function to find the root submission ID
          getRootSubmissionId: (submissionId) => {
            const state = get();
            // First try direct lookup by submissionId (for draft 1)
            if (state.drafts[submissionId]) {
              return submissionId; // This is the root
            }
            // Try to find in any draft array (for drafts 2+)
            for (const [key, drafts] of Object.entries(state.drafts)) {
              const found = drafts.find((d) => d.submissionId === submissionId);
              if (found) {
                // The key is the root submission ID, but verify by finding draft 1
                const draft1 = drafts.find((d) => d.draftNumber === 1);
                return draft1?.submissionId || key;
              }
            }
            return null;
          },

          getProgress: (submissionId) => {
            return get().progress[submissionId];
          },

          trackFixedErrors: (submissionId, previousErrorIds, currentErrorIds) => {
            set((state) => {
              if (!state.fixedErrors[submissionId]) {
                state.fixedErrors[submissionId] = new Set();
              }

              // Find errors that were in previous draft but not in current
              previousErrorIds.forEach((errorId) => {
                if (!currentErrorIds.includes(errorId)) {
                  state.fixedErrors[submissionId].add(errorId);
                }
              });
            });
            // Note: Persistence is handled automatically by persist middleware
          },

          getFixedErrors: (submissionId) => {
            return get().fixedErrors[submissionId] || new Set();
          },

          updateStreak: () => {
            set((state) => {
              const today = new Date().toISOString().split("T")[0]; // YYYY-MM-DD
              const lastDate = state.streak.lastActivityDate;

              if (!lastDate) {
                // First activity
                state.streak.currentStreak = 1;
                state.streak.lastActivityDate = today;
                state.streak.longestStreak = 1;
              } else if (lastDate === today) {
                // Already active today, don't increment
                // Keep current streak
              } else {
                // Check if yesterday
                const yesterday = new Date();
                yesterday.setDate(yesterday.getDate() - 1);
                const yesterdayStr = yesterday.toISOString().split("T")[0];

                if (lastDate === yesterdayStr) {
                  // Consecutive day - increment streak
                  state.streak.currentStreak += 1;
                  state.streak.lastActivityDate = today;
                  if (state.streak.currentStreak > state.streak.longestStreak) {
                    state.streak.longestStreak = state.streak.currentStreak;
                  }
                } else {
                  // Streak broken - reset
                  state.streak.currentStreak = 1;
                  state.streak.lastActivityDate = today;
                }
              }
            });
            // Note: Persistence is handled automatically by persist middleware
          },

          getStreak: () => {
            return get().streak;
          },

          checkAndUnlockAchievements: (draft, allDrafts) => {
            // This method is now handled within addDraft for better atomicity
            // Keeping for backwards compatibility but it's no longer used
            return [];
          },

          getAchievements: () => {
            return get().achievements;
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
            // Note: Persistence is handled automatically by persist middleware
            // The clearDrafts action will automatically persist the cleared state
          },

          // Computed selectors
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
        };
      }),
      {
        name: STORAGE_KEY,
        storage: storageWithSetHandling as any, // Custom storage with Set handling - compatible with persist
      }
    ),
    { name: "DraftStore" }
  )
);
