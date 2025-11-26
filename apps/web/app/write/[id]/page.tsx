"use client";

import { useState, useEffect, useCallback, useRef } from "react";
import { useRouter, useParams } from "next/navigation";
import Link from "next/link";
import { submitEssay } from "@/app/lib/actions";
import { usePreferencesStore } from "@/app/lib/stores/preferences-store";
import { useDraftStore } from "@/app/lib/stores/draft-store";
import { countWords, MIN_ESSAY_WORDS, MAX_ESSAY_WORDS } from "@writeo/shared";

// Task data - matches tasks from home page
const taskData: Record<string, { title: string; prompt: string }> = {
  "1": {
    title: "Education: Practical vs Theoretical",
    prompt:
      "Some people believe that universities should focus more on practical skills rather than theoretical knowledge. To what extent do you agree or disagree?",
  },
  "2": {
    title: "Technology: Social Media Impact",
    prompt:
      "Some people think that social media has a negative impact on society. Others believe it brings people together and has positive effects. Discuss both views and give your own opinion.",
  },
  "3": {
    title: "Environment: Individual vs Government",
    prompt:
      "Some people think that individuals should be responsible for protecting the environment. Others believe that governments should take the lead. What is your view?",
  },
  "4": {
    title: "Work: Remote Working",
    prompt:
      "More and more people are working from home rather than in offices. What are the advantages and disadvantages of this trend?",
  },
  "5": {
    title: "Health: Fast Food Problem",
    prompt:
      "Fast food consumption is increasing worldwide, leading to health problems. What problems does this cause, and what solutions can you suggest?",
  },
  "6": {
    title: "Society: Ageing Population",
    prompt:
      "In many countries, the population is ageing. What are the causes of this trend, and what effects might it have on society?",
  },
  "7": {
    title: "Culture: Global vs Local",
    prompt:
      "Some people think that globalization means losing local culture and traditions. Others believe it enriches culture by bringing people together. To what extent do you agree or disagree?",
  },
  "8": {
    title: "Crime: Punishment vs Rehabilitation",
    prompt:
      "Some people think that criminals should be punished harshly to deter crime. Others believe that rehabilitation programs are more effective. Discuss both views and give your opinion.",
  },
};

export default function WritePage() {
  const params = useParams();
  const router = useRouter();
  const setResult = useDraftStore((state) => state.setResult);
  const taskId = params.id as string;
  const isCustom = taskId === "custom";

  // Draft store (consolidated)
  const currentContent = useDraftStore((state) => state.currentContent);
  const updateContent = useDraftStore((state) => state.updateContent);
  const saveDraft = useDraftStore((state) => state.saveContentDraft);
  const activeDraftId = useDraftStore((state) => state.activeDraftId);
  const contentDrafts = useDraftStore((state) => state.contentDrafts);
  const loadContentDraft = useDraftStore((state) => state.loadContentDraft);

  const [customQuestion, setCustomQuestion] = useState("");
  const task = isCustom
    ? {
        title: "Custom Question",
        prompt: customQuestion.trim() || "",
      }
    : taskData[taskId] || {
        title: "Writing Practice",
        prompt: "Write your essay here.",
      };

  // Use draft store content instead of local state
  const answer = currentContent;
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [selfEval, setSelfEval] = useState({
    answeredAllParts: false,
    supportedOpinion: false,
    variedStructure: false,
  });

  // Auto-save debouncing
  const autoSaveTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  const AUTO_SAVE_DELAY = 2000; // 2 seconds after user stops typing

  // Use preferences store for storeResults (persists across sessions)
  const storeResults = usePreferencesStore((state) => state.storeResults);
  const setStoreResults = usePreferencesStore((state) => state.setStoreResults);

  const handleStoreResultsChange = (checked: boolean) => {
    setStoreResults(checked);
  };

  // Handle textarea change with explicit state update and auto-save debouncing
  const handleAnswerChange = useCallback(
    (e: React.ChangeEvent<HTMLTextAreaElement>) => {
      const newValue = e.target.value;
      updateContent(newValue);

      // Clear existing timeout
      if (autoSaveTimeoutRef.current) {
        clearTimeout(autoSaveTimeoutRef.current);
      }

      // Set new timeout for auto-save
      autoSaveTimeoutRef.current = setTimeout(() => {
        if (newValue.trim().length > 0) {
          saveDraft();
        }
      }, AUTO_SAVE_DELAY);
    },
    [updateContent, saveDraft]
  );

  // Load active draft on mount if currentContent is empty but activeDraftId exists
  useEffect(() => {
    if (!currentContent && activeDraftId && contentDrafts.length > 0) {
      loadContentDraft(activeDraftId);
    }
  }, []); // Only run on mount

  // Cleanup timeout on unmount
  useEffect(() => {
    return () => {
      if (autoSaveTimeoutRef.current) {
        clearTimeout(autoSaveTimeoutRef.current);
      }
    };
  }, []);

  // Handle custom question change
  const handleCustomQuestionChange = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    setCustomQuestion(e.target.value);
  };

  // Return prompt as-is (no additional reminder text)
  const getPrompt = (basePrompt: string) => {
    return basePrompt;
  };

  // Calculate word count
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
        `Your essay is too short. Please write at least ${MIN_WORDS} words (currently ${wordCount} words).`
      );
      return;
    }

    if (wordCount > MAX_WORDS) {
      setError(
        `Your essay is too long. Please keep it under ${MAX_WORDS} words (currently ${wordCount} words).`
      );
      return;
    }

    setLoading(true);
    setError(null);

    try {
      // Use custom question if provided, otherwise use empty string for free writing
      const questionText = task.prompt.trim() || "";

      // Wrap Server Action call with timeout to prevent hanging
      const submitPromise = submitEssay(questionText, answer, undefined, storeResults);
      const timeoutPromise = new Promise<never>((_, reject) => {
        setTimeout(() => reject(new Error("Request timed out. Please try again.")), 60000);
      });

      const { submissionId, results } = await Promise.race([submitPromise, timeoutPromise]);

      if (!submissionId || !results) {
        throw new Error("No submission ID or results returned");
      }

      // Ensure questionTexts are in metadata for draft tracking
      // The API should include questionTexts, but ensure they're present
      let resultsToStore = results;
      if (results && typeof window !== "undefined") {
        // Get answerId from answerTexts
        const answerTexts = results.meta?.answerTexts as Record<string, string> | undefined;
        const answerId = answerTexts ? Object.keys(answerTexts)[0] : undefined;

        // Store question text (including empty string for free writing) if we have answerId
        if (answerId && questionText !== undefined) {
          if (!results.meta?.questionTexts) {
            // Create a new results object to avoid mutation
            resultsToStore = {
              ...results,
              meta: {
                ...results.meta,
                questionTexts: {
                  [answerId]: questionText,
                },
              },
            };
          } else {
            const existingQuestionTexts = results.meta.questionTexts as Record<string, string>;
            if (!existingQuestionTexts[answerId]) {
              // Create a new results object to avoid mutation
              resultsToStore = {
                ...results,
                meta: {
                  ...results.meta,
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

      // Store results in draft store (Zustand persist handles localStorage automatically)
      setResult(submissionId, resultsToStore);

      // Redirect to results page
      router.push(`/results/${submissionId}`);
    } catch (err) {
      console.error("Submission error:", err);
      // Extract error message safely and make it user-friendly
      let errorMessage = "We couldn't submit your essay. Please try again.";

      if (err instanceof Error) {
        const message = err.message;
        // Handle Server Component errors (production builds omit details)
        if (
          message.includes("Server Components render") ||
          message.includes("omitted in production builds") ||
          message.includes("digest property")
        ) {
          errorMessage =
            "We encountered an issue while processing your submission. Please try again.";
        } else if (
          message.includes("Server configuration error") ||
          message.includes("API_KEY") ||
          message.includes("API_BASE_URL")
        ) {
          errorMessage = "There's a server configuration issue. Please try again later.";
        } else if (message.includes("timeout") || message.includes("timed out")) {
          errorMessage = "The request took too long. Please try again.";
        } else if (
          message.includes("network") ||
          message.includes("fetch") ||
          message.includes("Failed to fetch")
        ) {
          errorMessage =
            "Unable to connect to our servers. Please check your internet connection and try again.";
        } else if (
          message.length > 0 &&
          message.length < 200 &&
          !message.includes("Error:") &&
          !message.includes("at ")
        ) {
          // Use the error message if it's user-friendly (short, no stack traces)
          errorMessage = message;
        }
      } else if (typeof err === "string") {
        // Check if it's a user-friendly string
        if (err.length < 200 && !err.includes("Error:") && !err.includes("at ")) {
          errorMessage = err;
        }
      }

      setError(errorMessage);
      setLoading(false);
    }
  };

  return (
    <>
      <header className="header" lang="en">
        <div className="header-content">
          <div className="logo-group">
            <Link href="/" className="logo" lang="en">
              Writeo
            </Link>
          </div>
          <nav className="header-actions" aria-label="Writing actions" lang="en">
            <Link href="/" className="nav-back-link" lang="en">
              <span aria-hidden="true">←</span> Back to Tasks
            </Link>
          </nav>
        </div>
      </header>

      <div className="container" style={{ overflowY: "auto" }}>
        <div style={{ marginBottom: "32px" }} lang="en">
          <h1 className="page-title">{task.title}</h1>
          <p className="page-subtitle">
            Write your essay and get detailed feedback to improve your writing.
          </p>
        </div>

        <div className="writing-container">
          <div className="card question-card">
            <h2
              lang="en"
              style={{
                fontSize: "20px",
                marginBottom: "16px",
                display: "flex",
                alignItems: "center",
                gap: "8px",
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
                lang="en"
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
                lang="en"
              >
                {getPrompt(task.prompt)}
              </div>
            )}
            {isCustom && !customQuestion.trim() && (
              <p
                style={{
                  marginTop: "12px",
                  fontSize: "14px",
                  color: "var(--text-secondary)",
                  fontStyle: "italic",
                }}
                lang="en"
              >
                💡 Leave blank to practice free writing without answering a specific question.
              </p>
            )}
          </div>

          <div className="card answer-card">
            <form onSubmit={handleSubmit}>
              <label htmlFor="answer" className="label" lang="en">
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
                  <span>
                    {wordCount} {wordCount === 1 ? "word" : "words"}
                  </span>
                  {wordCount < MIN_WORDS && (
                    <span style={{ color: "var(--error-color)", fontWeight: 600 }}>
                      (Need at least {MIN_WORDS} words)
                    </span>
                  )}
                  {wordCount >= MIN_WORDS && wordCount <= MAX_WORDS && (
                    <span style={{ color: "var(--secondary-accent)" }}>✓</span>
                  )}
                  {wordCount > MAX_WORDS && (
                    <span style={{ color: "var(--error-color)", fontWeight: 600 }}>
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
                onInput={handleAnswerChange}
                placeholder={
                  isCustom && !customQuestion.trim()
                    ? "Write your essay here. Minimum 250 words required. This is free writing practice - write about any topic you choose."
                    : "Write your essay here. Minimum 250 words required. Aim for 250-300 words and address all parts of the question."
                }
                rows={20}
                disabled={loading}
                autoFocus={false}
                translate="no"
                lang="en"
              />

              {/* Self-Evaluation Checklist */}
              {answer.trim().length > 50 && (
                <div
                  lang="en"
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
                    lang="en"
                  >
                    ✓ Self-Evaluation Checklist (optional)
                  </p>
                  <div
                    style={{
                      display: "flex",
                      flexDirection: "column",
                      gap: "var(--spacing-sm)",
                    }}
                    lang="en"
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
                        lang="en"
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
                      lang="en"
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
                      lang="en"
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
                  borderRadius: "8px",
                  border: "1px solid rgba(102, 126, 234, 0.2)",
                }}
                lang="en"
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
                  lang="en"
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
                      lang="en"
                    >
                      ✓ Auto-saved
                    </span>
                  )}
                </div>
                <button
                  type="submit"
                  className="btn btn-primary"
                  disabled={loading || !answer.trim()}
                  title="We value your privacy – see our policy"
                >
                  {loading ? (
                    <span
                      style={{
                        display: "flex",
                        alignItems: "center",
                        gap: "var(--spacing-sm)",
                      }}
                      lang="en"
                    >
                      <span className="spinner"></span>
                      Analyzing your writing…
                    </span>
                  ) : (
                    <span lang="en">Get Feedback →</span>
                  )}
                </button>
                <Link href="/" className="btn btn-secondary" lang="en">
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
                lang="en"
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
          <div className="error" role="alert" style={{ marginTop: "var(--spacing-md)" }}>
            <strong>⚠️ {error}</strong>
          </div>
        )}
      </div>
    </>
  );
}
