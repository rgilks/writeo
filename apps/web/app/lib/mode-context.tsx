"use client";

import React, { createContext, useContext, useState, useEffect } from "react";

export type ViewMode = "learner" | "developer";

interface ModeContextType {
  mode: ViewMode;
  setMode: (mode: ViewMode) => void;
}

const ModeContext = createContext<ModeContextType | undefined>(undefined);

export function ModeProvider({ children }: { children: React.ReactNode }) {
  const [mode, setModeState] = useState<ViewMode>("learner");

  // Load mode from localStorage on mount
  useEffect(() => {
    const saved = localStorage.getItem("writeo-view-mode");
    if (saved === "developer" || saved === "learner") {
      setModeState(saved);
    }
  }, []);

  const setMode = (newMode: ViewMode) => {
    setModeState(newMode);
    localStorage.setItem("writeo-view-mode", newMode);
  };

  return <ModeContext.Provider value={{ mode, setMode }}>{children}</ModeContext.Provider>;
}

export function useMode() {
  const context = useContext(ModeContext);
  if (context === undefined) {
    throw new Error("useMode must be used within a ModeProvider");
  }
  return context;
}
