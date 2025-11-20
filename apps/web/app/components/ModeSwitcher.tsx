"use client";

import { usePreferencesStore } from "@/app/lib/stores/preferences-store";

export function ModeSwitcher() {
  const mode = usePreferencesStore((state) => state.viewMode);
  const setMode = usePreferencesStore((state) => state.setViewMode);
  const isDeveloper = mode === "developer";

  const toggleMode = () => {
    setMode(isDeveloper ? "learner" : "developer");
  };

  return (
    <button
      type="button"
      onClick={toggleMode}
      aria-pressed={isDeveloper}
      className={`developer-toggle${isDeveloper ? " developer-toggle--active" : ""}`}
    >
      <span className="developer-toggle__icon" aria-hidden="true">
        🔬
      </span>
      <span className="developer-toggle__meta">
        <span className="developer-toggle__label">Dev</span>
        <span className="developer-toggle__hint">{isDeveloper ? "on" : "off"}</span>
      </span>
      <span className="developer-toggle__pill" aria-hidden="true">
        <span className="developer-toggle__thumb" />
      </span>
    </button>
  );
}
