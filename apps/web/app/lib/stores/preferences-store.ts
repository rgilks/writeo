import { create } from "zustand";
import { devtools } from "zustand/middleware";
import { produce } from "immer";

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

// Save to localStorage
const saveToStorage = (state: PreferencesStore) => {
  if (typeof window === "undefined") return;

  try {
    const toStore = {
      viewMode: state.viewMode,
      storeResults: state.storeResults,
    };
    localStorage.setItem(STORAGE_KEY, JSON.stringify(toStore));
  } catch (error) {
    console.error("Failed to save preferences to localStorage:", error);
  }
};

export const usePreferencesStore = create<PreferencesStore>()(
  devtools(
    (set, get) => {
      const initialState = loadFromStorage();

      return {
        viewMode: (initialState.viewMode as ViewMode) || "learner",
        storeResults: initialState.storeResults ?? false,

        setViewMode: (mode) => {
          set(
            produce((draft) => {
              draft.viewMode = mode;
            }),
            false,
            "setViewMode"
          );
          // Save after mutation completes (get final state)
          saveToStorage(get());
        },

        setStoreResults: (value) => {
          set(
            produce((draft) => {
              draft.storeResults = value;
            }),
            false,
            "setStoreResults"
          );
          // Save after mutation completes (get final state)
          saveToStorage(get());
        },
      };
    },
    { name: "PreferencesStore" }
  )
);
