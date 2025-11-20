# Frontend State Management

**Last Updated:** January 2025  
**Status:** ✅ Production Ready

---

## Overview

Writeo uses a modern, performant state management architecture:

- **Zustand** for global state (draft tracking, user preferences)
- **Immer** for immutable state updates
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

**Persistence:** localStorage with automatic serialization/deserialization

#### Preferences Store (`apps/web/app/lib/stores/preferences-store.ts`)

Manages user preferences that persist across sessions.

**State:**

- `viewMode`: "learner" | "developer"
- `storeResults`: Whether to store results on server (boolean)

**Actions:**

- `setViewMode()`: Update view mode preference
- `setStoreResults()`: Update server storage preference

**Persistence:** localStorage with migration from old keys

### Component State → useState

**Appropriate Usage:**

- Form inputs: `answer`, `editedText`, `reflection` - Ephemeral, component-specific
- UI toggles: `isExpanded`, `showQuestion` - Component-specific visibility
- Loading states: `loading`, `isSubmitting` - Component-specific async state
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

### Immer Usage

**✅ DO: Mutate draft directly**

```typescript
produce((draft) => {
  draft.property = newValue;
  draft.nested.array.push(item);
});
```

**❌ DON'T: Destructure for mutation**

```typescript
produce((draft) => {
  const { items } = draft; // Breaks proxy
  items.push(item); // Won't track changes
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

## Performance

- **Selective Subscriptions**: Components only re-render when their selected state changes
- **Computed Selectors**: Expensive calculations done in store, not components
- **Memoization**: `useMemo` for derived values in components
- **Stable References**: Store functions are stable, safe for dependency arrays

## DevTools

Zustand DevTools enabled for both stores:

- `DraftStore` - Debug draft tracking and progress
- `PreferencesStore` - Debug user preferences

Install Redux DevTools browser extension to use.

## Migration Notes

The preferences store automatically migrates from old localStorage keys:

- `writeo-view-mode` → `writeo-preferences.viewMode`
- `writeo-store-results` → `writeo-preferences.storeResults`

Old keys are cleaned up after migration.

## Testing

Tests can directly set localStorage values for preferences (migration handles it), but prefer using the store when possible:

```typescript
// Works (migration handles it)
localStorage.setItem("writeo-store-results", "true");

// Better (uses store directly)
await page.evaluate(() => {
  usePreferencesStore.getState().setStoreResults(true);
});
```
