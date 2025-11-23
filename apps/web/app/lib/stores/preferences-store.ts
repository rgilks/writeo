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

// Load from localStorage with migration from old keys
const loadFromStorage = (): Partial<PreferencesStore> => {
  if (typeof window === "undefined") return {};

  try {
    // Try new unified storage first
    const stored = localStorage.getItem(STORAGE_KEY);
    if (stored) {
      return JSON.parse(stored);
    }

    // Migrate from old separate keys for backwards compatibility
    const oldViewMode = localStorage.getItem("writeo-view-mode");
    const oldStoreResults = localStorage.getItem("writeo-store-results");

    const migrated: Partial<PreferencesStore> = {};
    if (oldViewMode === "developer" || oldViewMode === "learner") {
      migrated.viewMode = oldViewMode;
    }
    if (oldStoreResults === "true") {
      migrated.storeResults = true;
    }

    // If we found old data, save it in new format and clean up
    if (Object.keys(migrated).length > 0) {
      const toStore = {
        viewMode: migrated.viewMode || "learner",
        storeResults: migrated.storeResults ?? false,
      };
      localStorage.setItem(STORAGE_KEY, JSON.stringify(toStore));
      // Clean up old keys
      localStorage.removeItem("writeo-view-mode");
      localStorage.removeItem("writeo-store-results");
      return toStore;
    }
  } catch (error) {
    console.error("Failed to load preferences from localStorage:", error);
  }
  return {};
};

export const usePreferencesStore = create<PreferencesStore>()(
  devtools(
    persist(
      immer((set) => {
        const initialState = loadFromStorage();

        return {
          viewMode: (initialState.viewMode as ViewMode) || "learner",
          storeResults: initialState.storeResults ?? false,

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
        };
      }),
      {
        name: STORAGE_KEY,
        storage: createJSONStorage(() => {
          const safeStorage = createSafeStorage();
          return {
            getItem: safeStorage.getItem,
            setItem: safeStorage.setItem,
            removeItem: safeStorage.removeItem,
          };
        }),
        // Migrate from old separate keys
        onRehydrateStorage: () => (state) => {
          if (state && typeof window !== "undefined") {
            // Migrate from old separate keys for backwards compatibility
            const oldViewMode = localStorage.getItem("writeo-view-mode");
            const oldStoreResults = localStorage.getItem("writeo-store-results");

            if (oldViewMode === "developer" || oldViewMode === "learner") {
              state.viewMode = oldViewMode;
            }
            if (oldStoreResults === "true") {
              state.storeResults = true;
            }

            // Clean up old keys if they exist
            if (oldViewMode || oldStoreResults) {
              localStorage.removeItem("writeo-view-mode");
              localStorage.removeItem("writeo-store-results");
            }
          }
        },
      }
    ),
    { name: "PreferencesStore" }
  )
);
