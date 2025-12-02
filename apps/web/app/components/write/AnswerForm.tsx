"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { MIN_ESSAY_WORDS, MAX_ESSAY_WORDS } from "@writeo/shared";
import { WordCountDisplay } from "./WordCountDisplay";
import { SelfEvaluationChecklist } from "./SelfEvaluationChecklist";
import { ServerStorageOption } from "./ServerStorageOption";

interface AnswerFormProps {
  answer: string;
  wordCount: number;
  error: string | null;
  loading: boolean;
  isCustom: boolean;
  customQuestion: string;
  selfEval: {
    answeredAllParts: boolean;
    supportedOpinion: boolean;
    variedStructure: boolean;
  };
  storeResults: boolean;
  onAnswerChange: (e: React.ChangeEvent<HTMLTextAreaElement>) => void;
  onSubmit: (e: React.FormEvent) => void;
  onSelfEvalChange: (updates: Partial<AnswerFormProps["selfEval"]>) => void;
  onStoreResultsChange: (checked: boolean) => void;
}

export function AnswerForm({
  answer,
  wordCount,
  error,
  loading,
  isCustom,
  customQuestion,
  selfEval,
  storeResults,
  onAnswerChange,
  onSubmit,
  onSelfEvalChange,
  onStoreResultsChange,
}: AnswerFormProps) {
  const [isHydrated, setIsHydrated] = useState(false);

  useEffect(() => {
    setIsHydrated(true);
  }, []);

  const isSubmitDisabled = loading || !answer.trim() || wordCount < MIN_ESSAY_WORDS;

  const placeholder =
    isCustom && !customQuestion.trim()
      ? "Write your essay here. Minimum 250 words required. This is free writing practice - write about any topic you choose."
      : "Write your essay here. Minimum 250 words required. Aim for 250-300 words and address all parts of the question.";

  return (
    <div
      className="card answer-card"
      data-testid="answer-form"
      data-hydrated={isHydrated ? "true" : "false"}
    >
      <form onSubmit={onSubmit}>
        <label htmlFor="answer" className="label">
          Your Answer
          <WordCountDisplay wordCount={wordCount} />
        </label>
        <textarea
          id="answer"
          data-testid="answer-textarea"
          data-word-count={wordCount}
          className="textarea notranslate"
          value={answer}
          onChange={onAnswerChange}
          aria-describedby={error ? "answer-error" : "answer-help"}
          aria-invalid={!!error || wordCount < MIN_ESSAY_WORDS || wordCount > MAX_ESSAY_WORDS}
          placeholder={placeholder}
          rows={20}
          disabled={loading}
          autoFocus={false}
          translate="no"
        />
        <div id="answer-help" className="sr-only">
          Minimum {MIN_ESSAY_WORDS} words required. Maximum {MAX_ESSAY_WORDS} words recommended.
        </div>

        {answer.trim().length > 50 && (
          <SelfEvaluationChecklist
            selfEval={selfEval}
            onSelfEvalChange={onSelfEvalChange}
            showAnsweredAllParts={!isCustom || !!customQuestion.trim()}
          />
        )}

        <ServerStorageOption
          storeResults={storeResults}
          onStoreResultsChange={onStoreResultsChange}
        />

        <div
          style={{
            marginTop: "var(--spacing-md)",
            display: "flex",
            gap: "var(--spacing-md)",
            alignItems: "center",
          }}
        >
          <div
            style={{
              flex: 1,
              display: "flex",
              alignItems: "center",
              gap: "var(--spacing-sm)",
            }}
          ></div>
          <button
            type="submit"
            data-testid="submit-button"
            className="btn btn-primary"
            disabled={isSubmitDisabled}
            title={
              wordCount < MIN_ESSAY_WORDS
                ? `Please write at least ${MIN_ESSAY_WORDS} words (currently ${wordCount} words)`
                : "We value your privacy – see our policy"
            }
          >
            {loading ? (
              <span
                style={{
                  display: "flex",
                  alignItems: "center",
                  gap: "var(--spacing-sm)",
                }}
              >
                <span className="spinner"></span>
                Analyzing your writing…
              </span>
            ) : (
              "Get Feedback →"
            )}
          </button>
          <Link href="/" className="btn btn-secondary">
            Cancel
          </Link>
        </div>
        <p
          style={{
            fontSize: "14px",
            color: "var(--text-secondary)",
            fontStyle: "italic",
            marginTop: "var(--spacing-sm)",
            lineHeight: "1.5",
          }}
        >
          Your text is processed by an AI model; no one else reads it.{" "}
          <Link
            href="/privacy"
            style={{
              color: "var(--primary-color)",
              textDecoration: "underline",
            }}
          >
            See our privacy policy
          </Link>
          .
        </p>
      </form>
    </div>
  );
}
