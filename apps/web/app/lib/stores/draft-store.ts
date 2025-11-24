import { create } from "zustand";
import { devtools, persist, StateStorage, createJSONStorage } from "zustand/middleware";
import { immer } from "zustand/middleware/immer";
import type { AssessmentResults } from "@writeo/shared";
import { createSafeStorage, cleanupExpiredStorage } from "../utils/storage";

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

interface DraftStore {
  // Draft content (before submission)
  contentDrafts: DraftContent[];
  currentContent: string;
  activeDraftId: string | null;

  // Assessment results (after submission)
  results: Record<string, StoredResult>;

  // Submission history & progress (after submission)
  drafts: Record<string, DraftHistory[]>;
  progress: Record<string, ProgressMetrics>;
  fixedErrors: Record<string, Set<string>>;
  achievements: Achievement[];
  streak: StreakData;

  // Actions
  updateContent: (text: string) => void;
  saveContentDraft: () => void;
  loadContentDraft: (id: string) => void;
  createNewContentDraft: () => void;
  deleteContentDraft: (id: string) => void;

  setResult: (submissionId: string, results: AssessmentResults) => void;
  getResult: (submissionId: string) => AssessmentResults | null;
  getParentSubmissionId: (submissionId: string) => string | null;
  removeResult: (submissionId: string) => void;
  clearAllResults: () => void;
  cleanupOldResults: (maxAgeMs?: number) => void;

  addDraft: (draft: DraftHistory, parentSubmissionId?: string) => Achievement[];
  getDraftHistory: (submissionId: string) => DraftHistory[];
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
  clearDrafts: () => void;

  // Computed selectors
  getTotalDrafts: () => number;
  getTotalWritings: () => number;
  getAverageImprovement: () => number;
  getAllDrafts: () => DraftHistory[];
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
// STORAGE ADAPTER (handles Set serialization for fixedErrors)
// ============================================================================

const baseStorage = createSafeStorage();

const storageWithSetHandling: StateStorage = {
  getItem: (name: string): string | null => {
    const str = baseStorage.getItem(name);
    if (!str || typeof str !== "string") return null;

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
          fixedErrors[key] = Array.isArray(value) ? new Set(value) : new Set();
        });
        parsed.state.fixedErrors = fixedErrors;
      }
      return JSON.stringify(parsed);
    } catch (error) {
      console.error("Failed to parse stored draft store, clearing corrupted data:", error);
      baseStorage.removeItem(name);
      return null;
    }
  },
  setItem: (name: string, value: string): void => {
    if (typeof window === "undefined") return;

    try {
      const parsed = JSON.parse(value);
      // Convert Sets to arrays for JSON serialization
      if (parsed?.state?.fixedErrors) {
        const fixedErrors: Record<string, string[]> = {};
        Object.entries(parsed.state.fixedErrors).forEach(([key, setValue]) => {
          fixedErrors[key] = setValue instanceof Set ? Array.from(setValue) : [];
        });
        parsed.state.fixedErrors = fixedErrors;
      }
      baseStorage.setItem(name, JSON.stringify(parsed));
    } catch (error) {
      console.error("Failed to save draft store:", error);
    }
  },
  removeItem: (name: string): void => {
    baseStorage.removeItem(name);
  },
};

// ============================================================================
// STORE CREATION
// ============================================================================

const STORAGE_KEY = "writeo-draft-store";

export const useDraftStore = create<DraftStore>()(
  devtools(
    persist(
      immer((set, get) => ({
        // Initial state
        contentDrafts: [],
        currentContent: "",
        activeDraftId: null,
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

        // Draft content actions
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

        // Assessment results actions
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
          cleanupExpiredStorage(maxAgeMs);
        },

        // Submission history actions
        getDraftHistory: (submissionId) => {
          const state = get();
          if (state.drafts[submissionId]) {
            return state.drafts[submissionId];
          }
          for (const drafts of Object.values(state.drafts)) {
            const found = drafts.find((d) => d.submissionId === submissionId);
            if (found) return drafts;
          }
          return [];
        },

        getRootSubmissionId: (submissionId) => {
          const state = get();
          if (state.drafts[submissionId]) {
            return submissionId;
          }
          for (const [key, drafts] of Object.entries(state.drafts)) {
            const found = drafts.find((d) => d.submissionId === submissionId);
            if (found) {
              const draft1 = drafts.find((d) => d.draftNumber === 1);
              return draft1?.submissionId || key;
            }
          }
          return null;
        },

        getProgress: (submissionId) => {
          return get().progress[submissionId];
        },

        addDraft: (draft, parentSubmissionId) => {
          let newAchievements: Achievement[] = [];

          set((state) => {
            const key = parentSubmissionId || draft.submissionId;

            if (!state.drafts[key]) {
              state.drafts[key] = [];
            }

            const existingIndex = state.drafts[key].findIndex(
              (d) => d.submissionId === draft.submissionId
            );

            if (existingIndex >= 0) {
              state.drafts[key][existingIndex] = { ...draft };
            } else {
              state.drafts[key].push({ ...draft });
              state.drafts[key].sort((a, b) => a.draftNumber - b.draftNumber);
            }

            // Update progress metrics
            const draftsArray = state.drafts[key];
            if (draftsArray.length > 0) {
              const first = draftsArray[0];
              const latest = draftsArray[draftsArray.length - 1];

              state.progress[draft.submissionId] = {
                totalDrafts: draftsArray.length,
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
              state.progress[draft.submissionId],
              totalFixedErrors,
              state.streak.currentStreak
            );

            if (newAchievements.length > 0) {
              state.achievements.push(...newAchievements);
            }
          });

          return newAchievements;
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
      })),
      {
        name: STORAGE_KEY,
        storage: storageWithSetHandling as any,
        partialize: (state) => ({
          // Only persist state, not functions
          contentDrafts: state.contentDrafts,
          currentContent: state.currentContent,
          activeDraftId: state.activeDraftId,
          results: state.results,
          drafts: state.drafts,
          progress: state.progress,
          fixedErrors: state.fixedErrors,
          achievements: state.achievements,
          streak: state.streak,
        }),
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
