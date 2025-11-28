"use client";

import { useState, useEffect, useRef } from "react";

export function useStoreHydration(store: {
  persist: {
    hasHydrated: () => boolean;
    onFinishHydration: (callback: () => void) => () => void;
  };
}): boolean {
  const persistRef = useRef(store.persist);
  persistRef.current = store.persist;

  const [isHydrated, setIsHydrated] = useState(() => persistRef.current.hasHydrated());

  useEffect(() => {
    if (persistRef.current.hasHydrated()) {
      setIsHydrated(true);
      return;
    }

    const unsubscribe = persistRef.current.onFinishHydration(() => {
      setIsHydrated(true);
    });

    return unsubscribe;
  }, []);

  return isHydrated;
}
