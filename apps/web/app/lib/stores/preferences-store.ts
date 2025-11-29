import { create } from "zustand";
import { devtools, persist, createJSONStorage } from "zustand/middleware";
import { createSafeStorage } from "../utils/storage";

const STORAGE_KEY = "writeo-preferences";
const DEFAULT_VIEW_MODE = "learner" as const;
const DEFAULT_STORE_RESULTS = false;

export type ViewMode = "learner" | "developer";

interface PreferencesStore {
  viewMode: ViewMode;
  setViewMode: (mode: ViewMode) => void;
  storeResults: boolean;
  setStoreResults: (value: boolean) => void;
}

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
