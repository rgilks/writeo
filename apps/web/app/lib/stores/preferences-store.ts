import { create } from "zustand";
import { devtools, persist, createJSONStorage } from "zustand/middleware";
import { immer } from "zustand/middleware/immer";
import { createSafeStorage } from "../utils/storage";

export type ViewMode = "learner" | "developer";

interface PreferencesStore {
  // View preferences
  viewMode: ViewMode;
  setViewMode: (mode: ViewMode) => void;

  // Storage preferences
  storeResults: boolean;
  setStoreResults: (value: boolean) => void;
}

const STORAGE_KEY = "writeo-preferences";

export const usePreferencesStore = create<PreferencesStore>()(
  devtools(
    persist(
      immer((set) => ({
        viewMode: "learner",
        storeResults: false,

        setViewMode: (mode) => {
          set((state) => {
            state.viewMode = mode;
          });
        },

        setStoreResults: (value) => {
          set((state) => {
            state.storeResults = value;
          });
        },
      })),
      {
        name: STORAGE_KEY,
        storage: createJSONStorage(() => createSafeStorage()),
      },
    ),
    { name: "PreferencesStore" },
  ),
);
