"use client";

import { useState, useEffect, useCallback, useRef } from "react";
import { useDraftStore } from "@/app/lib/stores/draft-store";
import { useStoreHydration } from "@/app/hooks/useStoreHydration";

const AUTO_SAVE_DELAY = 2000;

export function useWriteForm() {
  const currentContent = useDraftStore((state) => state.currentContent);
  const updateContent = useDraftStore((state) => state.updateContent);
  const saveDraft = useDraftStore((state) => state.saveContentDraft);
  const activeDraftId = useDraftStore((state) => state.activeDraftId);
  const contentDrafts = useDraftStore((state) => state.contentDrafts);
  const loadContentDraft = useDraftStore((state) => state.loadContentDraft);

  const isHydrated = useStoreHydration(useDraftStore);
  const [localAnswer, setLocalAnswer] = useState<string | null>(null);
  const autoSaveTimeoutRef = useRef<NodeJS.Timeout | null>(null);

  const answer = localAnswer !== null ? localAnswer : isHydrated ? currentContent : "";

  const handleAnswerChange = useCallback(
    (e: React.ChangeEvent<HTMLTextAreaElement>) => {
      const newValue = e.target.value;
      setLocalAnswer(newValue);
      updateContent(newValue);

      if (autoSaveTimeoutRef.current) {
        clearTimeout(autoSaveTimeoutRef.current);
      }

      autoSaveTimeoutRef.current = setTimeout(() => {
        if (newValue.trim().length > 0) {
          saveDraft();
        }
      }, AUTO_SAVE_DELAY);
    },
    [updateContent, saveDraft],
  );

  useEffect(() => {
    if (isHydrated && currentContent && localAnswer === null) {
      setLocalAnswer(currentContent);
    }
  }, [isHydrated, currentContent, localAnswer]);

  useEffect(() => {
    if (!isHydrated) return;

    if (!currentContent && activeDraftId && contentDrafts.length > 0) {
      const draft = contentDrafts.find((d) => d.id === activeDraftId);
      if (draft) {
        loadContentDraft(activeDraftId);
      }
    }
  }, [isHydrated, currentContent, activeDraftId, contentDrafts, loadContentDraft]);

  useEffect(() => {
    return () => {
      if (autoSaveTimeoutRef.current) {
        clearTimeout(autoSaveTimeoutRef.current);
      }
    };
  }, []);

  return {
    answer,
    isHydrated,
    activeDraftId,
    handleAnswerChange,
  };
}
