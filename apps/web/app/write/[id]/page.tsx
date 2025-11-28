"use client";

import { useState, useEffect, useCallback, useRef } from "react";
import { useRouter, useParams } from "next/navigation";
import Link from "next/link";
import type { AssessmentResults } from "@writeo/shared";
import { submitEssay } from "@/app/lib/actions";
import { usePreferencesStore } from "@/app/lib/stores/preferences-store";
import { useDraftStore } from "@/app/lib/stores/draft-store";
import { countWords, MIN_ESSAY_WORDS, MAX_ESSAY_WORDS } from "@writeo/shared";
import { TASK_DATA } from "@/app/lib/constants/tasks";
import { getErrorMessage } from "@/app/lib/utils/error-messages";
import { errorLogger } from "@/app/lib/utils/error-logger";
import { LiveRegion } from "@/app/components/LiveRegion";

const SUBMISSION_TIMEOUT = 60000;
const AUTO_SAVE_DELAY = 2000;

function getFriendlyErrorMessage(error: unknown): string {
  if (error instanceof Error) {
    return getErrorMessage(error, "write");
  }
  if (
    typeof error === "string" &&
    error.length < 200 &&
    !error.includes("Error:") &&
    !error.includes("at ")
  ) {
    return error;
  }
  return getErrorMessage(new Error("Submission failed"), "write");
}

export default function WritePage() {
  const params = useParams();
  const router = useRouter();
  const setResult = useDraftStore((state) => state.setResult);
  const taskId = params.id as string;
  const isCustom = taskId === "custom";

  const currentContent = useDraftStore((state) => state.currentContent);
  const updateContent = useDraftStore((state) => state.updateContent);
  const saveDraft = useDraftStore((state) => state.saveContentDraft);
  const activeDraftId = useDraftStore((state) => state.activeDraftId);
  const contentDrafts = useDraftStore((state) => state.contentDrafts);
  const loadContentDraft = useDraftStore((state) => state.loadContentDraft);

  const [isHydrated, setIsHydrated] = useState(() => useDraftStore.persist.hasHydrated());

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

  const [localAnswer, setLocalAnswer] = useState<string | null>(null);
  const answer = localAnswer !== null ? localAnswer : isHydrated ? currentContent : "";
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [selfEval, setSelfEval] = useState({
    answeredAllParts: false,
    supportedOpinion: false,
    variedStructure: false,
  });

  const autoSaveTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  const storeResults = usePreferencesStore((state) => state.storeResults);
  const setStoreResults = usePreferencesStore((state) => state.setStoreResults);

  const handleStoreResultsChange = (checked: boolean) => {
    setStoreResults(checked);
  };

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
    if (useDraftStore.persist.hasHydrated()) {
      setIsHydrated(true);
      if (currentContent && localAnswer === null) {
        setLocalAnswer(currentContent);
      }
      return;
    }

    const unsubscribe = useDraftStore.persist.onFinishHydration(() => {
      setIsHydrated(true);
      if (currentContent && localAnswer === null) {
        setLocalAnswer(currentContent);
      }
    });

    return () => {
      unsubscribe();
    };
  }, [currentContent, localAnswer]);

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
    if (isHydrated && currentContent && localAnswer === null) {
      setLocalAnswer(currentContent);
    }
  }, [isHydrated, currentContent, localAnswer]);

  useEffect(() => {
    return () => {
      if (autoSaveTimeoutRef.current) {
        clearTimeout(autoSaveTimeoutRef.current);
      }
    };
  }, []);

  const handleCustomQuestionChange = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    setCustomQuestion(e.target.value);
  };

  const getPrompt = (basePrompt: string) => {
    return basePrompt;
  };

  const wordCount = countWords(answer);
  const MIN_WORDS = MIN_ESSAY_WORDS;
  const MAX_WORDS = MAX_ESSAY_WORDS; // Soft cap - warn but allow

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();

    if (!answer.trim()) {
      setError("Please write your essay before submitting. Add your answer to receive feedback.");
      return;
    }

    // Validate word count
    if (wordCount < MIN_WORDS) {
      setError(
        `Your essay is too short. Please write at least ${MIN_WORDS} words (currently ${wordCount} words).`,
      );
      return;
    }

    if (wordCount > MAX_WORDS) {
      setError(
        `Your essay is too long. Please keep it under ${MAX_WORDS} words (currently ${wordCount} words).`,
      );
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const questionText = task.prompt.trim() || "";

      const submitPromise = submitEssay(questionText, answer, undefined, storeResults);
      const timeoutPromise = new Promise<never>((_, reject) => {
        setTimeout(
          () => reject(new Error("Request timed out. Please try again.")),
          SUBMISSION_TIMEOUT,
        );
      });

      const { submissionId, results } = await Promise.race([submitPromise, timeoutPromise]);

      if (!submissionId || !results) {
        throw new Error("No submission ID or results returned");
      }

      if (
        typeof results !== "object" ||
        results === null ||
        !("status" in results) ||
        !("template" in results)
      ) {
        throw new Error("Invalid results format");
      }

      const resultsObj = results as AssessmentResults;

      let resultsToStore = resultsObj;
      if (resultsObj && typeof window !== "undefined") {
        const answerTexts = resultsObj.meta?.answerTexts as Record<string, string> | undefined;
        const answerId = answerTexts ? Object.keys(answerTexts)[0] : undefined;

        if (answerId && questionText !== undefined) {
          if (!resultsObj.meta?.questionTexts) {
            resultsToStore = {
              ...resultsObj,
              meta: {
                ...resultsObj.meta,
                questionTexts: {
                  [answerId]: questionText,
                },
              },
            };
          } else {
            const existingQuestionTexts = resultsObj.meta.questionTexts as Record<string, string>;
            if (!existingQuestionTexts[answerId]) {
              resultsToStore = {
                ...resultsObj,
                meta: {
                  ...resultsObj.meta,
                  questionTexts: {
                    ...existingQuestionTexts,
                    [answerId]: questionText,
                  },
                },
              };
            }
          }
        }
      }

      setResult(submissionId, resultsToStore);
      router.push(`/results/${submissionId}`);
    } catch (err) {
      errorLogger.logError(err, {
        page: "write",
        action: "submit_essay",
        taskId,
        wordCount,
      });
      setError(getFriendlyErrorMessage(err));
      setLoading(false);
    }
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
          <div className="card question-card">
            <h2
              style={{
                fontSize: "20px",
                marginBottom: "var(--spacing-md)",
                display: "flex",
                alignItems: "center",
                gap: "var(--spacing-sm)",
              }}
            >
              <span>📝</span> {isCustom ? "Your Question (Optional)" : "Question"}
            </h2>
            {isCustom ? (
              <textarea
                className="textarea notranslate"
                value={customQuestion}
                onChange={handleCustomQuestionChange}
                placeholder="Enter your question here, or leave blank for free writing practice..."
                rows={4}
                disabled={loading}
                translate="no"
                style={{
                  width: "100%",
                  minHeight: "80px",
                  resize: "vertical",
                }}
              />
            ) : (
              <div
                className="prompt-box notranslate"
                style={{ whiteSpace: "pre-wrap" }}
                translate="no"
              >
                {getPrompt(task.prompt)}
              </div>
            )}
            {isCustom && !customQuestion.trim() && (
              <p
                style={{
                  marginTop: "var(--spacing-sm)",
                  fontSize: "14px",
                  color: "var(--text-secondary)",
                  fontStyle: "italic",
                }}
              >
                💡 Leave blank to practice free writing without answering a specific question.
              </p>
            )}
          </div>

          <div className="card answer-card">
            <form onSubmit={handleSubmit}>
              <label htmlFor="answer" className="label">
                Your Answer
                <div
                  style={{
                    display: "flex",
                    gap: "var(--spacing-md)",
                    alignItems: "center",
                    fontSize: "14px",
                    color: "var(--text-secondary)",
                  }}
                >
                  <span aria-live="polite" aria-atomic="true">
                    {wordCount} {wordCount === 1 ? "word" : "words"}
                  </span>
                  {wordCount < MIN_WORDS && (
                    <span
                      style={{ color: "var(--error-color)", fontWeight: 600 }}
                      role="status"
                      aria-live="polite"
                    >
                      (Need at least {MIN_WORDS} words)
                    </span>
                  )}
                  {wordCount >= MIN_WORDS && wordCount <= MAX_WORDS && (
                    <span
                      style={{ color: "var(--secondary-accent)" }}
                      aria-label="Word count valid"
                    >
                      ✓
                    </span>
                  )}
                  {wordCount > MAX_WORDS && (
                    <span
                      style={{ color: "var(--error-color)", fontWeight: 600 }}
                      role="status"
                      aria-live="polite"
                    >
                      (Too long - maximum {MAX_WORDS} words)
                    </span>
                  )}
                </div>
              </label>
              <textarea
                id="answer"
                className="textarea notranslate"
                value={answer}
                onChange={handleAnswerChange}
                aria-describedby={error ? "answer-error" : "answer-help"}
                aria-invalid={!!error || wordCount < MIN_WORDS || wordCount > MAX_WORDS}
                placeholder={
                  isCustom && !customQuestion.trim()
                    ? "Write your essay here. Minimum 250 words required. This is free writing practice - write about any topic you choose."
                    : "Write your essay here. Minimum 250 words required. Aim for 250-300 words and address all parts of the question."
                }
                rows={20}
                disabled={loading}
                autoFocus={false}
                translate="no"
              />
              <div id="answer-help" className="sr-only">
                Minimum {MIN_WORDS} words required. Maximum {MAX_WORDS} words recommended.
              </div>

              {/* Self-Evaluation Checklist */}
              {answer.trim().length > 50 && (
                <div
                  style={{
                    marginTop: "var(--spacing-md)",
                    padding: "var(--spacing-md)",
                    backgroundColor: "rgba(102, 126, 234, 0.1)",
                    borderRadius: "var(--border-radius)",
                  }}
                >
                  <p
                    style={{
                      marginBottom: "var(--spacing-sm)",
                      fontSize: "14px",
                      fontWeight: 600,
                    }}
                  >
                    ✓ Self-Evaluation Checklist (optional)
                  </p>
                  <div
                    style={{
                      display: "flex",
                      flexDirection: "column",
                      gap: "var(--spacing-sm)",
                    }}
                  >
                    {(!isCustom || customQuestion.trim()) && (
                      <label
                        style={{
                          display: "flex",
                          alignItems: "center",
                          gap: "var(--spacing-sm)",
                          fontSize: "14px",
                          cursor: "pointer",
                          lineHeight: "1.5",
                        }}
                      >
                        <input
                          type="checkbox"
                          checked={selfEval.answeredAllParts}
                          onChange={(e) =>
                            setSelfEval({
                              ...selfEval,
                              answeredAllParts: e.target.checked,
                            })
                          }
                          style={{ cursor: "pointer" }}
                        />
                        Did I answer all parts of the question?
                      </label>
                    )}
                    <label
                      style={{
                        display: "flex",
                        alignItems: "center",
                        gap: "var(--spacing-sm)",
                        fontSize: "14px",
                        cursor: "pointer",
                        lineHeight: "1.5",
                      }}
                    >
                      <input
                        type="checkbox"
                        checked={selfEval.supportedOpinion}
                        onChange={(e) =>
                          setSelfEval({
                            ...selfEval,
                            supportedOpinion: e.target.checked,
                          })
                        }
                        style={{ cursor: "pointer" }}
                      />
                      Did I support my opinion with at least two reasons?
                    </label>
                    <label
                      style={{
                        display: "flex",
                        alignItems: "center",
                        gap: "var(--spacing-sm)",
                        fontSize: "14px",
                        cursor: "pointer",
                        lineHeight: "1.5",
                      }}
                    >
                      <input
                        type="checkbox"
                        checked={selfEval.variedStructure}
                        onChange={(e) =>
                          setSelfEval({
                            ...selfEval,
                            variedStructure: e.target.checked,
                          })
                        }
                        style={{ cursor: "pointer" }}
                      />
                      Did I vary my sentence structure?
                    </label>
                  </div>
                </div>
              )}

              {/* Server Storage Opt-in */}
              <div
                style={{
                  marginTop: "var(--spacing-md)",
                  padding: "var(--spacing-md)",
                  backgroundColor: "rgba(102, 126, 234, 0.05)",
                  borderRadius: "var(--border-radius)",
                  border: "1px solid rgba(102, 126, 234, 0.2)",
                }}
              >
                <label
                  style={{
                    display: "flex",
                    alignItems: "flex-start",
                    gap: "var(--spacing-sm)",
                    fontSize: "14px",
                    cursor: "pointer",
                    lineHeight: "1.5",
                  }}
                >
                  <input
                    type="checkbox"
                    checked={storeResults}
                    onChange={(e) => handleStoreResultsChange(e.target.checked)}
                    style={{ cursor: "pointer", marginTop: "2px" }}
                  />
                  <span>
                    <strong>Save results on server (optional)</strong>
                    <br />
                    <span style={{ fontSize: "13px", color: "var(--text-secondary)" }}>
                      By default, your results are only saved in your browser. Check this box to
                      enable server storage so you can access your results from any device. Your
                      data will be stored for 90 days.
                    </span>
                  </span>
                </label>
              </div>

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
                >
                  {activeDraftId && (
                    <span
                      style={{
                        fontSize: "0.875rem",
                        color: "var(--text-secondary)",
                      }}
                    >
                      ✓ Auto-saved
                    </span>
                  )}
                </div>
                <button
                  type="submit"
                  className="btn btn-primary"
                  disabled={loading || !answer.trim() || wordCount < MIN_WORDS}
                  title={
                    wordCount < MIN_WORDS
                      ? `Please write at least ${MIN_WORDS} words (currently ${wordCount} words)`
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
