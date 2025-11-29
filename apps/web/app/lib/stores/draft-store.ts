import { create } from "zustand";
import { devtools, persist, createJSONStorage } from "zustand/middleware";
import { immer } from "zustand/middleware/immer";
import type { AssessmentResults } from "@writeo/shared";
import { createSafeStorage } from "../utils/storage";

const STORAGE_KEY = "writeo-draft-store";
const DEFAULT_RESULT_MAX_AGE_MS = 30 * 24 * 60 * 60 * 1000;
const SUMMARY_MAX_LENGTH = 40;
const MIN_SCORE_IMPROVEMENT_FOR_ACHIEVEMENT = 1.0;
const MIN_FIXED_ERRORS_FOR_ACHIEVEMENT = 10;
const MIN_STREAK_FOR_ACHIEVEMENT = 7;
const MIN_DRAFTS_FOR_REVISER_ACHIEVEMENT = 5;

const CEFR_LEVELS = ["A2", "B1", "B2", "C1", "C2"] as const;
type CefrLevel = (typeof CEFR_LEVELS)[number];

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
  contentDrafts: DraftContent[];
  currentContent: string;
  activeDraftId: string | null;

  results: Record<string, StoredResult>;
  drafts: Record<string, DraftHistory[]>;
  progress: Record<string, ProgressMetrics>;
  fixedErrors: Record<string, string[]>;
  achievements: Achievement[];
  streak: StreakData;
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

  addDraft: (draft: DraftHistory, rootSubmissionId: string) => Achievement[];
  getDraftHistory: (rootSubmissionId: string) => DraftHistory[];
  getRootSubmissionId: (submissionId: string) => string | null;
  getProgress: (submissionId: string) => ProgressMetrics | undefined;

  trackFixedErrors: (
    submissionId: string,
    previousErrorIds: string[],
    currentErrorIds: string[],
  ) => void;
  getFixedErrors: (submissionId: string) => string[];

  updateStreak: () => void;

  getTotalDrafts: () => number;
  getTotalWritings: () => number;
  getAverageImprovement: () => number;
  getAllDrafts: () => DraftHistory[];

  cleanupOldResults: (maxAgeMs?: number) => void;
  clearDrafts: () => void;
}

const generateId = (): string => Math.random().toString(36).substring(2, 9);

const countWords = (text: string): number => {
  const trimmed = text.trim();
  return trimmed.length === 0 ? 0 : trimmed.split(/\s+/).length;
};

const generateSummary = (text: string): string => {
  const trimmed = text.trim();
  if (trimmed.length === 0) return "Empty draft";
  return trimmed.slice(0, SUMMARY_MAX_LENGTH) + (trimmed.length > SUMMARY_MAX_LENGTH ? "..." : "");
};

const isCefrLevel = (level: string | undefined): level is CefrLevel => {
  return level !== undefined && CEFR_LEVELS.includes(level as CefrLevel);
};

const getCefrLevelIndex = (level: string | undefined): number => {
  if (!isCefrLevel(level)) return -1;
  return CEFR_LEVELS.indexOf(level);
};

function findRootSubmissionId(
  submissionId: string,
  drafts: Record<string, DraftHistory[]>,
): string | null {
  if (drafts[submissionId]) {
    return submissionId;
  }

  for (const [rootId, draftArray] of Object.entries(drafts)) {
    const found = draftArray.find((d) => d.submissionId === submissionId);
    if (found) {
      return rootId;
    }
  }

  return null;
}

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

function checkAchievements(
  draft: DraftHistory,
  allDrafts: DraftHistory[],
  existingAchievements: Achievement[],
  progress: ProgressMetrics | undefined,
  totalFixedErrors: number,
  currentStreak: number,
): Achievement[] {
  const existingIds = new Set(existingAchievements.map((a) => a.id));
  const achievements: Achievement[] = [];
  const now = new Date().toISOString();

  const createAchievement = (
    id: string,
    name: string,
    description: string,
    icon: string,
  ): Achievement => ({
    id,
    name,
    description,
    icon,
    unlockedAt: now,
  });

  if (allDrafts.length === 1 && !existingIds.has("first-draft")) {
    achievements.push(
      createAchievement("first-draft", "First Draft", "Submitted your first essay", "🎯"),
    );
  }

  if (allDrafts.length >= MIN_DRAFTS_FOR_REVISER_ACHIEVEMENT && !existingIds.has("reviser")) {
    achievements.push(createAchievement("reviser", "Reviser", "Submitted 5 drafts", "✏️"));
  }

  if (
    progress?.scoreImprovement &&
    progress.scoreImprovement >= MIN_SCORE_IMPROVEMENT_FOR_ACHIEVEMENT &&
    !existingIds.has("improver")
  ) {
    achievements.push(
      createAchievement("improver", "Improver", "Improved your score by 1.0+ points", "📈"),
    );
  }

  if (totalFixedErrors >= MIN_FIXED_ERRORS_FOR_ACHIEVEMENT && !existingIds.has("grammar-master")) {
    achievements.push(
      createAchievement("grammar-master", "Grammar Master", "Fixed 10+ grammar errors", "🎓"),
    );
  }

  if (isCefrLevel(draft.cefrLevel)) {
    const currentLevelIndex = getCefrLevelIndex(draft.cefrLevel);
    const allLevelIndices = allDrafts
      .map((d) => getCefrLevelIndex(d.cefrLevel))
      .filter((i) => i >= 0);
    const maxLevel = Math.max(...allLevelIndices, -1);

    if (
      currentLevelIndex === maxLevel &&
      currentLevelIndex > 0 &&
      !existingIds.has("cefr-climber")
    ) {
      achievements.push(
        createAchievement(
          "cefr-climber",
          "CEFR Climber",
          `Reached CEFR level ${draft.cefrLevel}`,
          "🏆",
        ),
      );
    }
  }

  if (currentStreak >= MIN_STREAK_FOR_ACHIEVEMENT && !existingIds.has("streak-keeper")) {
    achievements.push(
      createAchievement(
        "streak-keeper",
        "Streak Keeper",
        "Maintained a 7+ day practice streak",
        "🔥",
      ),
    );
  }

  return achievements;
}

function calculateNewStreak(
  lastDate: string | "",
  currentStreak: number,
  longestStreak: number,
): StreakData {
  const today = new Date().toISOString().split("T")[0];

  if (!lastDate) {
    return { currentStreak: 1, longestStreak: 1, lastActivityDate: today };
  }

  if (lastDate === today) {
    return { currentStreak, longestStreak, lastActivityDate: today };
  }

  const yesterday = new Date();
  yesterday.setDate(yesterday.getDate() - 1);
  const yesterdayStr = yesterday.toISOString().split("T")[0];

  if (lastDate === yesterdayStr) {
    const newStreak = currentStreak + 1;
    return {
      currentStreak: newStreak,
      longestStreak: Math.max(newStreak, longestStreak),
      lastActivityDate: today,
    };
  }

  return { currentStreak: 1, longestStreak, lastActivityDate: today };
}

export const useDraftStore = create<DraftStore>()(
  devtools(
    persist(
      immer((set, get) => ({
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
              (d) => d.id === draft.activeDraftId,
            );

            if (existingIndex !== -1 && draft.activeDraftId) {
              const updatedDraft = {
                ...draft.contentDrafts[existingIndex],
                content: state.currentContent,
                lastModified: now,
                wordCount,
                summary,
              };
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
            if (!state.drafts[rootSubmissionId]) {
              state.drafts[rootSubmissionId] = [];
            }

            const draftArray = state.drafts[rootSubmissionId];
            const existingIndex = draftArray.findIndex(
              (d) => d.submissionId === draft.submissionId,
            );

            if (existingIndex >= 0) {
              draftArray[existingIndex] = { ...draft };
            } else {
              draftArray.push({ ...draft });
              draftArray.sort((a, b) => a.draftNumber - b.draftNumber);
            }

            const progress = calculateProgress(draftArray);
            if (progress) {
              state.progress[rootSubmissionId] = progress;
            }

            const newStreak = calculateNewStreak(
              state.streak.lastActivityDate,
              state.streak.currentStreak,
              state.streak.longestStreak,
            );
            state.streak = newStreak;

            const allDrafts = Object.values(state.drafts).flat();
            const totalFixedErrors = Object.values(state.fixedErrors).reduce(
              (sum, errors) => sum + errors.length,
              0,
            );

            newAchievements = checkAchievements(
              draft,
              allDrafts,
              state.achievements,
              progress,
              totalFixedErrors,
              newStreak.currentStreak,
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
          const progress = get().progress[submissionId];
          if (progress) return progress;

          const rootId = findRootSubmissionId(submissionId, get().drafts);
          return rootId ? get().progress[rootId] : undefined;
        },

        trackFixedErrors: (submissionId, previousErrorIds, currentErrorIds) => {
          set((state) => {
            if (!state.fixedErrors[submissionId]) {
              state.fixedErrors[submissionId] = [];
            }

            const fixedSet = new Set(state.fixedErrors[submissionId]);
            const currentSet = new Set(currentErrorIds);

            previousErrorIds.forEach((errorId) => {
              if (!currentSet.has(errorId) && !fixedSet.has(errorId)) {
                state.fixedErrors[submissionId].push(errorId);
              }
            });
          });
        },

        getFixedErrors: (submissionId) => {
          return get().fixedErrors[submissionId] || [];
        },

        updateStreak: () => {
          set((state) => {
            state.streak = calculateNewStreak(
              state.streak.lastActivityDate,
              state.streak.currentStreak,
              state.streak.longestStreak,
            );
          });
        },

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

        // Cleanup
        cleanupOldResults: (maxAgeMs = DEFAULT_RESULT_MAX_AGE_MS) => {
          const now = Date.now();
          set((state) => {
            Object.keys(state.results).forEach((submissionId) => {
              const stored = state.results[submissionId];
              if (stored && now - stored.timestamp > maxAgeMs) {
                delete state.results[submissionId];
              }
            });
          });
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
        storage: createJSONStorage(() => createSafeStorage()),
      },
    ),
    { name: "DraftStore" },
  ),
);
