"use client";

import { useMode } from "@/app/lib/mode-context";

export function ModeSwitcher() {
  const { mode, setMode } = useMode();
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
