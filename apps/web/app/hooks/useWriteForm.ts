"use client";

import { useState, useEffect, useCallback } from "react";
import { useDraftStore } from "@/app/lib/stores/draft-store";
import { useStoreHydration } from "@/app/hooks/useStoreHydration";

export function useWriteForm() {
  const currentContent = useDraftStore((state) => state.currentContent);
  const updateContent = useDraftStore((state) => state.updateContent);
  const activeDraftId = useDraftStore((state) => state.activeDraftId);
  const contentDrafts = useDraftStore((state) => state.contentDrafts);
  const loadContentDraft = useDraftStore((state) => state.loadContentDraft);

  const isHydrated = useStoreHydration(useDraftStore);
  const [localAnswer, setLocalAnswer] = useState<string | null>(null);

  const answer = localAnswer !== null ? localAnswer : isHydrated ? currentContent : "";

  const handleAnswerChange = useCallback(
    (e: React.ChangeEvent<HTMLTextAreaElement>) => {
      const newValue = e.target.value;
      setLocalAnswer(newValue);
      updateContent(newValue);
    },
    [updateContent],
  );

  useEffect(() => {
    if (isHydrated && currentContent && localAnswer === null) {
      setLocalAnswer(currentContent);
    }
  }, [isHydrated, currentContent, localAnswer]);

  // Load draft if one is active (e.g., user clicked "Continue Editing" from history)
  useEffect(() => {
    if (!isHydrated) return;

    if (!currentContent && activeDraftId && contentDrafts.length > 0) {
      const draft = contentDrafts.find((d) => d.id === activeDraftId);
      if (draft) {
        loadContentDraft(activeDraftId);
      }
    }
  }, [isHydrated, currentContent, activeDraftId, contentDrafts, loadContentDraft]);

  return {
    answer,
    isHydrated,
    activeDraftId,
    handleAnswerChange,
  };
}
