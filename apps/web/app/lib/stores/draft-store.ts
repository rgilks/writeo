import { create } from "zustand";
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

export const useDraftStore = create<DraftStore>((set, get) => {
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
          const drafts = draftState.drafts[key];
          if (drafts.length > 0) {
            const firstDraft = drafts[0];
            const latestDraft = drafts[drafts.length - 1];

            draftState.progress[draft.submissionId] = {
              totalDrafts: drafts.length,
              firstDraftScore: firstDraft.overallScore,
              latestDraftScore: latestDraft.overallScore,
              scoreImprovement:
                latestDraft.overallScore && firstDraft.overallScore
                  ? latestDraft.overallScore - firstDraft.overallScore
                  : undefined,
              errorReduction:
                latestDraft.errorCount !== undefined && firstDraft.errorCount !== undefined
                  ? firstDraft.errorCount - latestDraft.errorCount
                  : undefined,
              wordCountChange:
                latestDraft.wordCount !== undefined && firstDraft.wordCount !== undefined
                  ? latestDraft.wordCount - firstDraft.wordCount
                  : undefined,
            };
          }
        })
      );

      // Get updated state after produce
      const updatedState = get();

      // Update streak when draft is added
      updatedState.updateStreak();

      // Check for new achievements - create fresh copy of draft to avoid Immer issues
      const allDrafts = Object.values(updatedState.drafts).flat();
      const freshDraft = {
        ...draft,
        errorIds: Array.isArray(draft.errorIds) ? [...draft.errorIds] : [],
      };
      const newAchievements = updatedState.checkAndUnlockAchievements(freshDraft, allDrafts);

      // Save to localStorage after state update
      saveToStorage(updatedState);

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
        })
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
        })
      );
      saveToStorage(get());
    },

    getStreak: () => {
      return get().streak;
    },

    checkAndUnlockAchievements: (draft, allDrafts) => {
      const newAchievements: Achievement[] = [];
      const existingIds = new Set(get().achievements.map((a) => a.id));

      // First Draft achievement
      if (allDrafts.length === 1 && !existingIds.has("first-draft")) {
        newAchievements.push({
          id: "first-draft",
          name: "First Draft",
          description: "Submitted your first essay",
          icon: "🎯",
          unlockedAt: new Date().toISOString(),
        });
      }

      // Reviser achievement (5 drafts)
      if (allDrafts.length >= 5 && !existingIds.has("reviser")) {
        newAchievements.push({
          id: "reviser",
          name: "Reviser",
          description: "Submitted 5 drafts",
          icon: "✏️",
          unlockedAt: new Date().toISOString(),
        });
      }

      // Improver achievement (score improved by 1.0+)
      const progress = get().progress[draft.submissionId];
      if (
        progress?.scoreImprovement &&
        progress.scoreImprovement >= 1.0 &&
        !existingIds.has("improver")
      ) {
        newAchievements.push({
          id: "improver",
          name: "Improver",
          description: "Improved your score by 1.0+ points",
          icon: "📈",
          unlockedAt: new Date().toISOString(),
        });
      }

      // Grammar Master (fixed 10+ errors total)
      const totalFixedErrors = Object.values(get().fixedErrors).reduce(
        (sum, errors) => sum + errors.size,
        0
      );
      if (totalFixedErrors >= 10 && !existingIds.has("grammar-master")) {
        newAchievements.push({
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
        // Check if this is a new highest level
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
          newAchievements.push({
            id: "cefr-climber",
            name: "CEFR Climber",
            description: `Reached CEFR level ${draft.cefrLevel}`,
            icon: "🏆",
            unlockedAt: new Date().toISOString(),
          });
        }
      }

      // Streak Keeper (7+ day streak)
      const streak = get().streak;
      if (streak.currentStreak >= 7 && !existingIds.has("streak-keeper")) {
        newAchievements.push({
          id: "streak-keeper",
          name: "Streak Keeper",
          description: "Maintained a 7+ day practice streak",
          icon: "🔥",
          unlockedAt: new Date().toISOString(),
        });
      }

      // Add new achievements to store
      if (newAchievements.length > 0) {
        set(
          produce((draftState: DraftStore) => {
            draftState.achievements.push(...newAchievements.map((a) => ({ ...a })));
          })
        );
        saveToStorage(get());
      }

      return newAchievements;
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
        })
      );
      if (typeof window !== "undefined") {
        localStorage.removeItem(STORAGE_KEY);
      }
    },
  };
});
