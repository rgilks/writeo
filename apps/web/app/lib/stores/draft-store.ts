import { create } from "zustand";
import { devtools } from "zustand/middleware";
import { produce } from "immer";

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

// Load from localStorage
const loadFromStorage = (): Partial<DraftStore> => {
  if (typeof window === "undefined") return {};

  try {
    const stored = localStorage.getItem(STORAGE_KEY);
    if (stored) {
      const parsed = JSON.parse(stored);
      // Convert fixedErrors Sets back from arrays
      const fixedErrors: Record<string, Set<string>> = {};
      if (parsed.fixedErrors) {
        Object.entries(parsed.fixedErrors).forEach(([key, value]) => {
          fixedErrors[key] = new Set(value as string[]);
        });
      }
      return {
        ...parsed,
        fixedErrors,
      };
    }
  } catch (error) {
    console.error("Failed to load draft store from localStorage:", error);
  }
  return {};
};

// Save to localStorage
const saveToStorage = (state: DraftStore) => {
  if (typeof window === "undefined") return;

  try {
    // Convert Sets to arrays for JSON serialization
    const fixedErrors: Record<string, string[]> = {};
    Object.entries(state.fixedErrors).forEach(([key, value]) => {
      fixedErrors[key] = Array.from(value);
    });

    const toStore = {
      ...state,
      fixedErrors,
    };
    localStorage.setItem(STORAGE_KEY, JSON.stringify(toStore));
  } catch (error) {
    console.error("Failed to save draft store to localStorage:", error);
  }
};

export const useDraftStore = create<DraftStore>()(
  devtools(
    (set, get) => {
      const initialState = loadFromStorage();

      return {
        drafts: initialState.drafts || {},
        progress: initialState.progress || {},
        fixedErrors: initialState.fixedErrors || {},
        achievements: (initialState.achievements as Achievement[]) || [],
        streak: (initialState.streak as StreakData) || {
          currentStreak: 0,
          longestStreak: 0,
          lastActivityDate: "",
        },

        addDraft: (draft, parentSubmissionId) => {
          let newAchievements: Achievement[] = [];

          set(
            produce((draftState: DraftStore) => {
              const key = parentSubmissionId || draft.submissionId;

              // Initialize array if it doesn't exist
              if (!draftState.drafts[key]) {
                draftState.drafts[key] = [];
              }

              // Check if this draft already exists (by submissionId)
              const existingIndex = draftState.drafts[key].findIndex(
                (d) => d.submissionId === draft.submissionId
              );

              if (existingIndex >= 0) {
                // Update existing draft
                draftState.drafts[key][existingIndex] = { ...draft };
              } else {
                // Add new draft
                draftState.drafts[key].push({ ...draft });
                // Sort by draft number
                draftState.drafts[key].sort((a, b) => a.draftNumber - b.draftNumber);
              }

              // Update progress metrics
              // Access through draftState directly (don't destructure to maintain Immer proxy)
              const draftsArray = draftState.drafts[key];
              if (draftsArray.length > 0) {
                // Read values directly from draftState to maintain proxy chain
                const firstDraftScore = draftsArray[0]?.overallScore;
                const latestDraftScore = draftsArray[draftsArray.length - 1]?.overallScore;
                const firstErrorCount = draftsArray[0]?.errorCount;
                const latestErrorCount = draftsArray[draftsArray.length - 1]?.errorCount;
                const firstWordCount = draftsArray[0]?.wordCount;
                const latestWordCount = draftsArray[draftsArray.length - 1]?.wordCount;

                draftState.progress[draft.submissionId] = {
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

              // Update streak when draft is added (within produce for atomicity)
              const today = new Date().toISOString().split("T")[0];
              // Read directly from draftState (reading is safe, but don't destructure for mutation)
              const lastDate = draftState.streak.lastActivityDate;

              if (!lastDate) {
                draftState.streak.currentStreak = 1;
                draftState.streak.lastActivityDate = today;
                draftState.streak.longestStreak = 1;
              } else if (lastDate !== today) {
                const yesterday = new Date();
                yesterday.setDate(yesterday.getDate() - 1);
                const yesterdayStr = yesterday.toISOString().split("T")[0];

                if (lastDate === yesterdayStr) {
                  draftState.streak.currentStreak += 1;
                  draftState.streak.lastActivityDate = today;
                  if (draftState.streak.currentStreak > draftState.streak.longestStreak) {
                    draftState.streak.longestStreak = draftState.streak.currentStreak;
                  }
                } else {
                  draftState.streak.currentStreak = 1;
                  draftState.streak.lastActivityDate = today;
                }
              }

              // Check for new achievements (within produce to access current state)
              // Reading from draftState is safe - we're creating new arrays/sets for computation
              const allDrafts = Object.values(draftState.drafts).flat();
              const existingIds = new Set(draftState.achievements.map((a) => a.id));
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
              // Read directly from draftState (reading is safe)
              const progressEntry = draftState.progress[draft.submissionId];
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
              // Reading from draftState is safe - we're computing a value, not mutating
              const totalFixedErrors = Object.values(draftState.fixedErrors).reduce(
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
              if (draftState.streak.currentStreak >= 7 && !existingIds.has("streak-keeper")) {
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
                draftState.achievements.push(...achievementsToAdd);
                newAchievements = achievementsToAdd.map((a) => ({ ...a }));
              }
            }),
            false,
            "addDraft"
          );

          // Save to localStorage after state update
          saveToStorage(get());

          // Return new achievements for notification
          return newAchievements;
        },

        getDraftHistory: (submissionId) => {
          const state = get();
          // Try to find in any draft array
          for (const drafts of Object.values(state.drafts)) {
            const found = drafts.find((d) => d.submissionId === submissionId);
            if (found) {
              // Return all drafts in the same group
              const key = Object.keys(state.drafts).find((k) => state.drafts[k].includes(found));
              return key ? state.drafts[key] : [found];
            }
          }
          return [];
        },

        getProgress: (submissionId) => {
          return get().progress[submissionId];
        },

        trackFixedErrors: (submissionId, previousErrorIds, currentErrorIds) => {
          set(
            produce((draftState: DraftStore) => {
              if (!draftState.fixedErrors[submissionId]) {
                draftState.fixedErrors[submissionId] = new Set();
              }

              // Find errors that were in previous draft but not in current
              previousErrorIds.forEach((errorId) => {
                if (!currentErrorIds.includes(errorId)) {
                  draftState.fixedErrors[submissionId].add(errorId);
                }
              });
            }),
            false,
            "trackFixedErrors"
          );

          saveToStorage(get());
        },

        getFixedErrors: (submissionId) => {
          return get().fixedErrors[submissionId] || new Set();
        },

        updateStreak: () => {
          set(
            produce((draftState: DraftStore) => {
              const today = new Date().toISOString().split("T")[0]; // YYYY-MM-DD
              const lastDate = draftState.streak.lastActivityDate;

              if (!lastDate) {
                // First activity
                draftState.streak.currentStreak = 1;
                draftState.streak.lastActivityDate = today;
                draftState.streak.longestStreak = 1;
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
                  draftState.streak.currentStreak += 1;
                  draftState.streak.lastActivityDate = today;
                  if (draftState.streak.currentStreak > draftState.streak.longestStreak) {
                    draftState.streak.longestStreak = draftState.streak.currentStreak;
                  }
                } else {
                  // Streak broken - reset
                  draftState.streak.currentStreak = 1;
                  draftState.streak.lastActivityDate = today;
                }
              }
            }),
            false,
            "updateStreak"
          );
          saveToStorage(get());
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
          set(
            produce((draftState: DraftStore) => {
              draftState.drafts = {};
              draftState.progress = {};
              draftState.fixedErrors = {};
              draftState.achievements = [];
              draftState.streak = {
                currentStreak: 0,
                longestStreak: 0,
                lastActivityDate: "",
              };
            }),
            false,
            "clearDrafts"
          );
          if (typeof window !== "undefined") {
            localStorage.removeItem(STORAGE_KEY);
          }
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
    },
    { name: "DraftStore" }
  )
);
