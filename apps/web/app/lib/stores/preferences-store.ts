import { create } from "zustand";
import { devtools, persist, createJSONStorage } from "zustand/middleware";
import { createSafeStorage } from "../utils/storage";

// ============================================================================
// CONSTANTS
// ============================================================================

const STORAGE_KEY = "writeo-preferences";
const DEFAULT_VIEW_MODE = "learner" as const;
const DEFAULT_STORE_RESULTS = false;

// ============================================================================
// TYPES
// ============================================================================

export type ViewMode = "learner" | "developer";

interface PreferencesStore {
  // View preferences
  viewMode: ViewMode;
  setViewMode: (mode: ViewMode) => void;

  // Storage preferences
  storeResults: boolean;
  setStoreResults: (value: boolean) => void;
}

// ============================================================================
// STORE CREATION
// ============================================================================

export const usePreferencesStore = create<PreferencesStore>()(
  devtools(
    persist(
      (set) => ({
        viewMode: DEFAULT_VIEW_MODE,
        storeResults: DEFAULT_STORE_RESULTS,

        setViewMode: (mode) => set({ viewMode: mode }),

        setStoreResults: (value) => set({ storeResults: value }),
      }),
      {
        name: STORAGE_KEY,
        storage: createJSONStorage(() => createSafeStorage()),
      },
    ),
    { name: "PreferencesStore" },
  ),
);
