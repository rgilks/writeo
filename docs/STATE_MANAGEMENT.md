# Frontend State Management

**Status:** ✅ Production Ready

---

## Overview

Writeo uses a modern, performant state management architecture:

- **Zustand** for global state (draft tracking, user preferences, assessment results)
- **Zustand Persist Middleware** for automatic localStorage persistence
- **Zustand Immer Middleware** for immutable state updates with direct mutations
- **Safe Storage Utilities** for error handling, quota management, and cleanup
- **useState** for component-specific UI state

## Architecture

### Global State → Zustand Stores

#### Draft Store (`apps/web/app/lib/stores/draft-store.ts`)

Manages draft history, progress tracking, achievements, and streaks.

**State:**

- `drafts`: Record of draft histories by submission ID
- `progress`: Progress metrics per submission
- `fixedErrors`: Tracked fixed errors per submission
- `achievements`: Unlocked achievements
- `streak`: Daily practice streak data

**Actions:**

- `addDraft()`: Add/update draft with automatic progress calculation
- `getDraftHistory()`: Retrieve draft history for a submission
- `trackFixedErrors()`: Track which errors were fixed between drafts
- `updateStreak()`: Update daily practice streak
- `checkAndUnlockAchievements()`: Check and unlock new achievements

**Computed Selectors:**

- `getTotalDrafts()`: Total number of drafts across all submissions
- `getTotalWritings()`: Total number of writing sessions
- `getAverageImprovement()`: Average score improvement
- `getAllDrafts()`: All drafts flattened

**Persistence:**

- Uses Zustand's `persist` middleware for automatic localStorage persistence
- Uses `createSafeStorage()` utility for error handling and quota management
- Custom storage adapter handles `Set<string>` serialization/deserialization for `fixedErrors`
- Automatically saves on every state change
- Automatically hydrates on store initialization

#### Preferences Store (`apps/web/app/lib/stores/preferences-store.ts`)

Manages user preferences that persist across sessions.

**State:**

- `viewMode`: "learner" | "developer"
- `storeResults`: Whether to store results on server (boolean)

**Actions:**

- `setViewMode()`: Update view mode preference
- `setStoreResults()`: Update server storage preference

**Persistence:**

- Uses Zustand's `persist` middleware for automatic localStorage persistence
- Uses `createSafeStorage()` utility for error handling and quota management
- Automatic migration from old separate keys (`writeo-view-mode`, `writeo-store-results`)
- Migration handled in `onRehydrateStorage` callback

#### Results Store (`apps/web/app/lib/stores/results-store.ts`)

Manages assessment results storage in localStorage. Replaces direct localStorage access with centralized, type-safe storage.

**State:**

- `results`: Record of assessment results by submission ID
  - Each entry includes: `results` (AssessmentResults), `timestamp`
  - **Note**: `parentSubmissionId` is stored in `results.meta.parentSubmissionId`, not as a separate field

**Actions:**

- `setResult()`: Store assessment results (parentSubmissionId is already in results.meta)
- `getResult()`: Retrieve assessment results by submission ID
- `getParentSubmissionId()`: Get parent submission ID from results.meta for draft tracking
- `removeResult()`: Remove a specific result
- `clearAllResults()`: Clear all stored results
- `cleanupOldResults()`: Remove results older than specified age (default: 30 days)

**Computed Selectors:**

- `getAllSubmissionIds()`: Get all stored submission IDs
- `getResultsCount()`: Get total number of stored results

**Persistence:**

- Uses Zustand's `persist` middleware with `createSafeStorage()` utility
- Automatic cleanup of old results (30 days) on rehydration
- Automatic cleanup on app start
- Type-safe API prevents storage errors

### Custom Hooks

Custom hooks encapsulate store logic and provide clean APIs for components:

- **`useDraftStorage`**: Automatically stores draft data when results arrive
- **`useDraftHistory`**: Computes and memoizes draft history display (uses `useMemo`)
- **`useResubmit`**: Handles draft resubmission with proper state management
- **`useDraftNavigation`**: Calculates navigation URLs for draft buttons

**Pattern:**

```typescript
export function useDraftHistory(...) {
  const getDraftHistory = useDraftStore((state) => state.getDraftHistory);

  // Memoize expensive computation
  const displayDraftHistory = useMemo(() => {
    // ... expensive computation
  }, [deps]);

  return { displayDraftHistory, draftNumber, parentSubmissionId };
}
```

### Component State → useState

**Appropriate Usage:**

- Form inputs: `answer`, `editedText`, `reflection` - Ephemeral, component-specific
- UI toggles: `isExpanded`, `showQuestion` - Component-specific visibility
- Loading states: `loading`, `isResubmitting` - Component-specific async state
- Component lifecycle: `mounted` - Hydration fixes

## Best Practices

### Zustand Selectors

**✅ DO: Use granular selectors**

```typescript
// Subscribe only to needed state slices
const drafts = useDraftStore((state) => state.drafts);
const progress = useDraftStore((state) => state.progress);
```

**❌ DON'T: Subscribe to entire store**

```typescript
// This causes unnecessary re-renders
const store = useDraftStore();
```

### Immer Middleware Usage

We use Zustand's `immer` middleware, which allows direct mutations in `set()` callbacks.

**✅ DO: Mutate state directly in set() callback**

```typescript
set((state) => {
  state.property = newValue;
  state.nested.array.push(item);
});
```

**❌ DON'T: Wrap in produce() manually**

```typescript
// Don't do this - immer middleware handles it automatically
set(
  produce((draft) => {
    draft.property = newValue;
  })
);
```

**❌ DON'T: Destructure for mutation**

```typescript
set((state) => {
  const { items } = state; // Breaks proxy
  items.push(item); // Won't track changes
});
```

**✅ DO: Access state directly (don't destructure)**

```typescript
set((state) => {
  // Reading is safe - access directly
  const currentValue = state.items.length;
  // Mutating is safe - immer middleware handles it
  state.items.push(newItem);
});
```

### Component Patterns

**✅ DO: Use selectors for state, functions for actions**

```typescript
const drafts = useDraftStore((state) => state.drafts);
const addDraft = useDraftStore((state) => state.addDraft);
```

**✅ DO: Use computed selectors for expensive calculations**

```typescript
const totalDrafts = useDraftStore((state) => state.getTotalDrafts());
```

**✅ DO: Use useMemo for expensive derived values**

```typescript
const displayDraftHistory = useMemo(() => {
  // Expensive computation
  return processDrafts(drafts);
}, [drafts, otherDeps]);
```

**✅ DO: Exclude stable store functions from dependency arrays**

```typescript
useEffect(() => {
  // Store functions are stable - don't need to be in deps
  addDraft(draftData);
  // eslint-disable-next-line react-hooks/exhaustive-deps
}, [draftData]); // Only include actual data dependencies
```

**✅ DO: Use hook selectors instead of getState()**

```typescript
// Good - component re-renders if preference changes
const storeResults = usePreferencesStore((state) => state.storeResults);

// Only use getState() for one-time reads in async functions
const currentValue = usePreferencesStore.getState().storeResults;
```

## Performance

- **Selective Subscriptions**: Components only re-render when their selected state changes
- **Computed Selectors**: Expensive calculations done in store, not components
- **Memoization**: `useMemo` for derived values in components (e.g., `useDraftHistory`)
- **Stable References**: Store functions are stable - safe to exclude from dependency arrays
- **Automatic Persistence**: Zustand persist middleware optimizes when persistence happens
- **Structural Sharing**: Immer middleware provides structural sharing out of the box

## Storage Utilities

### Safe Storage (`apps/web/app/lib/utils/storage.ts`)

Provides centralized storage utilities with error handling, quota management, and cleanup:

- **Error Handling**: Catches and handles `QuotaExceededError`, `SecurityError`, and other storage errors
- **Quota Management**: Warns when storage is approaching limits and attempts cleanup
- **Corruption Detection**: Automatically detects and clears corrupted data (e.g., `"[object Object]"`)
- **Cleanup Utilities**: Functions to clean up expired storage entries

**Usage:**

```typescript
import { createSafeStorage } from "@/app/lib/utils/storage";

// Use with Zustand persist
storage: createJSONStorage(() => createSafeStorage());
```

## DevTools

Zustand DevTools enabled for all stores:

- `DraftStore` - Debug draft tracking and progress
- `PreferencesStore` - Debug user preferences
- `ResultsStore` - Debug assessment results storage

Install Redux DevTools browser extension to use.

## Migration Notes

The preferences store automatically migrates from old localStorage keys:

- `writeo-view-mode` → `writeo-preferences.viewMode`
- `writeo-store-results` → `writeo-preferences.storeResults`

Old keys are cleaned up after migration.

## Store Structure

### Middleware Stack Order

Both stores use the same middleware stack order (important for proper functionality):

```typescript
create<StoreType>()(
  devtools(           // 3. DevTools (outermost)
    persist(          // 2. Persistence
      immer(          // 1. Immer (innermost)
        (set, get) => ({ ... })
      ),
      { name: "storage-key" }
    ),
    { name: "StoreName" }
  )
)
```

**Why this order matters:**

- `immer` must be innermost to handle mutations
- `persist` wraps immer to serialize state
- `devtools` wraps everything for debugging

### Draft Store Example

```typescript
export const useDraftStore = create<DraftStore>()(
  devtools(
    persist(
      immer((set, get) => ({
        drafts: {},
        progress: {},
        fixedErrors: {},
        achievements: [],
        streak: { currentStreak: 0, longestStreak: 0, lastActivityDate: "" },

        addDraft: (draft, parentSubmissionId) => {
          set((state) => {
            // Direct mutations - immer middleware handles immutability
            const key = parentSubmissionId || draft.submissionId;
            if (!state.drafts[key]) {
              state.drafts[key] = [];
            }
            state.drafts[key].push({ ...draft });
            // ... more mutations
          });
        },
      })),
      {
        name: "writeo-draft-store",
        storage: storageWithSetHandling, // Custom adapter for Set serialization
      }
    ),
    { name: "DraftStore" }
  )
);
```

### Preferences Store Example

```typescript
import { createSafeStorage } from "@/app/lib/utils/storage";

export const usePreferencesStore = create<PreferencesStore>()(
  devtools(
    persist(
      immer((set) => ({
        viewMode: "learner",
        storeResults: false,

        setViewMode: (mode) => {
          set((state) => {
            state.viewMode = mode; // Direct mutation
          });
        },
      })),
      {
        name: "writeo-preferences",
        storage: createJSONStorage(() => createSafeStorage()),
        onRehydrateStorage: () => (state) => {
          // Migration logic
        },
      }
    ),
    { name: "PreferencesStore" }
  )
);
```

### Results Store Example

```typescript
import { createSafeStorage } from "@/app/lib/utils/storage";

export const useResultsStore = create<ResultsStore>()(
  devtools(
    persist(
      immer((set, get) => ({
        results: {},

        setResult: (submissionId, results) => {
          set((state) => {
            state.results[submissionId] = {
              results,
              timestamp: Date.now(),
              // parentSubmissionId is stored in results.meta.parentSubmissionId
            };
          });
        },

        getResult: (submissionId) => {
          return get().results[submissionId]?.results || null;
        },
      })),
      {
        name: "writeo-results-store",
        storage: createJSONStorage(() => createSafeStorage()),
        onRehydrateStorage: () => (state) => {
          // Cleanup old results
          state?.cleanupOldResults(30 * 24 * 60 * 60 * 1000);
        },
      }
    ),
    { name: "ResultsStore" }
  )
);
```

## Implementation Details

### Draft Store Persistence

The draft store uses a custom storage adapter to handle `Set<string>` serialization:

```typescript
// Sets are converted to arrays when saving
fixedErrors: Record<string, Set<string>> → Record<string, string[]>

// Arrays are converted back to Sets when loading
Record<string, string[]> → Record<string, Set<string>>
```

This is necessary because `Set` cannot be directly serialized to JSON. The custom adapter handles this conversion automatically.

### Preferences Store Migration

The preferences store automatically migrates from old localStorage keys during hydration:

1. Checks for new unified key: `writeo-preferences`
2. Falls back to old keys: `writeo-view-mode`, `writeo-store-results`
3. Migrates data to new format
4. Cleans up old keys

Migration happens in the `onRehydrateStorage` callback of the persist middleware.

## Testing

Tests can directly set localStorage values for preferences (migration handles it), but prefer using the store when possible:

```typescript
// Works (migration handles it)
await page.evaluate(() => {
  const prefs = { viewMode: "learner", storeResults: false };
  localStorage.setItem("writeo-preferences", JSON.stringify(prefs));
});

// Better (uses store directly)
await page.evaluate(() => {
  usePreferencesStore.getState().setStoreResults(true);
});
```

### Storage Keys

**Zustand Stores (managed automatically):**

- `writeo-draft-store` - Draft store (Zustand persist)
- `writeo-preferences` - Preferences store (Zustand persist)
- `writeo-results-store` - Results store (Zustand persist)

**Legacy Keys (for backwards compatibility):**

- `results_{submissionId}` - Assessment results (now managed by Results Store)
- Note: `parentSubmissionId` is stored in `results.meta.parentSubmissionId`, not as a separate key

**Testing Notes:**

Tests can still use direct localStorage access for backwards compatibility, but prefer using stores:

```typescript
// Works (backwards compatible)
await page.evaluate((submissionId) => {
  localStorage.setItem(`results_${submissionId}`, JSON.stringify(results));
});

// Better (uses Results Store)
await page.evaluate(
  (submissionId, results) => {
    useResultsStore.getState().setResult(submissionId, results);
  },
  submissionId,
  results
);
```
