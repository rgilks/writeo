"use client";

import { useState } from "react";
import { useParams } from "next/navigation";
import Link from "next/link";
import { countWords, MIN_ESSAY_WORDS, MAX_ESSAY_WORDS } from "@writeo/shared";
import { TASK_DATA } from "@/app/lib/constants/tasks";
import { usePreferencesStore } from "@/app/lib/stores/preferences-store";
import { useWriteForm } from "@/app/hooks/useWriteForm";
import { useEssaySubmission } from "@/app/hooks/useEssaySubmission";
import { QuestionCard } from "@/app/components/write/QuestionCard";
import { AnswerForm } from "@/app/components/write/AnswerForm";
import { LiveRegion } from "@/app/components/LiveRegion";

export default function WritePage() {
  const params = useParams();
  const taskId = params.id as string;
  const isCustom = taskId === "custom";

  const [customQuestion, setCustomQuestion] = useState("");
  const task = isCustom
    ? {
        title: "Custom Question",
        prompt: customQuestion.trim() || "",
      }
    : TASK_DATA[taskId] || {
        title: "Writing Practice",
        prompt: "Write your essay here.",
      };

  const [selfEval, setSelfEval] = useState({
    answeredAllParts: false,
    supportedOpinion: false,
    variedStructure: false,
  });

  const storeResults = usePreferencesStore((state) => state.storeResults);
  const setStoreResults = usePreferencesStore((state) => state.setStoreResults);

  const { answer, activeDraftId, handleAnswerChange } = useWriteForm();
  const { submit, loading, error } = useEssaySubmission();

  const wordCount = countWords(answer);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();

    const questionText = task.prompt.trim() || "";
    await submit(
      questionText,
      answer,
      taskId,
      wordCount,
      MIN_ESSAY_WORDS,
      MAX_ESSAY_WORDS,
      storeResults,
    );
  };

  const handleCustomQuestionChange = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    setCustomQuestion(e.target.value);
  };

  const handleSelfEvalChange = (updates: Partial<typeof selfEval>) => {
    setSelfEval((prev) => ({ ...prev, ...updates }));
  };

  const handleStoreResultsChange = (checked: boolean) => {
    setStoreResults(checked);
  };

  return (
    <>
      <LiveRegion
        message={
          loading
            ? "Submitting your essay for analysis, please wait"
            : error
              ? `Error: ${error}`
              : null
        }
        priority={error ? "assertive" : "polite"}
      />
      <header className="header" lang="en">
        <div className="header-content">
          <div className="logo-group">
            <Link href="/" className="logo">
              Writeo
            </Link>
          </div>
          <nav className="header-actions" aria-label="Writing actions">
            <Link href="/" className="nav-back-link">
              <span aria-hidden="true">←</span> Back to Home
            </Link>
          </nav>
        </div>
      </header>

      <div className="container" style={{ overflowY: "auto" }}>
        <div style={{ marginBottom: "var(--spacing-xl)" }}>
          <h1 className="page-title">{task.title}</h1>
          <p className="page-subtitle">
            Write your essay and get detailed feedback to improve your writing.
          </p>
        </div>

        <div className="writing-container">
          <QuestionCard
            isCustom={isCustom}
            customQuestion={customQuestion}
            prompt={task.prompt}
            onCustomQuestionChange={handleCustomQuestionChange}
            disabled={loading}
          />

          <AnswerForm
            answer={answer}
            wordCount={wordCount}
            error={error}
            loading={loading}
            isCustom={isCustom}
            customQuestion={customQuestion}
            selfEval={selfEval}
            storeResults={storeResults}
            activeDraftId={activeDraftId}
            onAnswerChange={handleAnswerChange}
            onSubmit={handleSubmit}
            onSelfEvalChange={handleSelfEvalChange}
            onStoreResultsChange={handleStoreResultsChange}
          />
        </div>

        {error && (
          <div
            id="answer-error"
            className="error"
            role="alert"
            aria-live="assertive"
            style={{ marginTop: "var(--spacing-md)" }}
          >
            <strong>
              <span aria-hidden="true">⚠️</span> {error}
            </strong>
          </div>
        )}
      </div>
    </>
  );
}
