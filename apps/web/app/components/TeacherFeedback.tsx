"use client";

import { useEffect, useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import ReactMarkdown from "react-markdown";
import { getTeacherFeedback } from "@/app/lib/actions";

interface TeacherFeedbackProps {
  overall: number;
  dimensions: {
    TA: number;
    CC: number;
    Vocab: number;
    Grammar: number;
    Overall: number;
  };
  errorCount: number;
  aiFeedback?: {
    message: string;
    focusArea?: string;
    cluesMessage?: string;
    explanationMessage?: string;
  };
  submissionId?: string;
  answerId?: string;
  answerText?: string;
  questionText?: string;
  ltErrors?: any[];
  llmErrors?: any[];
  relevanceCheck?: {
    addressesQuestion: boolean;
    score: number;
    threshold: number;
  };
}

/**
 * TeacherFeedback - Provides AI-generated feedback from a helpful teacher
 * Supports interactive flow: initial feedback -> clues -> full explanation
 */
export function TeacherFeedback({
  overall,
  dimensions,
  errorCount,
  aiFeedback,
  submissionId,
  answerId,
  answerText,
  questionText,
  ltErrors,
  llmErrors,
  relevanceCheck,
}: TeacherFeedbackProps) {
  // Initialize from stored feedback if available
  const [feedbackMode, setFeedbackMode] = useState<"initial" | "explanation">("initial");
  const [explanation, setExplanation] = useState<string | null>(
    aiFeedback?.explanationMessage || null
  );
  const [initialMessage, setInitialMessage] = useState<string | null>(aiFeedback?.message || null);
  const [initialLoading, setInitialLoading] = useState(false);
  const [initialError, setInitialError] = useState<string | null>(null);
  const [hasRequestedClues, setHasRequestedClues] = useState(false);
  const [loading, setLoading] = useState(false);
  const [showTryAgain, setShowTryAgain] = useState(true);

  const getFeedbackText = () => {
    if (feedbackMode === "explanation" && explanation) {
      return explanation;
    }
    if (initialMessage) {
      return initialMessage;
    }
    if (aiFeedback?.message) {
      return aiFeedback.message;
    }
    if (initialLoading) {
      return "Preparing feedback...";
    }
    if (initialError) {
      return "Loading feedback...";
    }
    return "Preparing feedback...";
  };

  const feedbackText = getFeedbackText();

  // Ensure the short Groq encouragement is available even if the async pipeline
  // hasn’t stored it yet (common when submissions run in async mode).
  useEffect(() => {
    if (
      hasRequestedClues ||
      initialMessage ||
      aiFeedback?.message ||
      initialLoading ||
      !submissionId ||
      !answerId ||
      !answerText
    ) {
      return;
    }

    let cancelled = false;
    setInitialLoading(true);
    setInitialError(null);

    getTeacherFeedback(submissionId, answerId, "clues", answerText, questionText, {
      essayScores: {
        overall,
        dimensions,
      },
      ltErrors,
      llmErrors,
      relevanceCheck,
    })
      .then((data) => {
        if (cancelled) {
          return;
        }
        setInitialMessage(data.message || null);
        setHasRequestedClues(true);
      })
      .catch((error) => {
        if (cancelled) {
          return;
        }
        console.error("Error auto-fetching teacher clues:", error);
        const message = error instanceof Error ? error.message : "Unknown error";
        setInitialError(message);
      })
      .finally(() => {
        if (!cancelled) {
          setInitialLoading(false);
        }
      });

    return () => {
      cancelled = true;
    };
  }, [
    submissionId,
    answerId,
    answerText,
    aiFeedback?.message,
    initialMessage,
    initialLoading,
    hasRequestedClues,
  ]);

  const handleTryAgain = async () => {
    if (!submissionId || !answerId || !answerText) {
      console.error("Missing required data:", { submissionId, answerId, answerText: !!answerText });
      alert(
        "Cannot get feedback: Missing required information. Please ensure the essay was submitted correctly."
      );
      return;
    }

    setLoading(true);
    try {
      console.log("Requesting detailed teacher feedback", {
        submissionId,
        answerId,
        answerTextLength: answerText.length,
      });

      // Use Server Action instead of API route
      const data = await getTeacherFeedback(
        submissionId,
        answerId,
        "explanation",
        answerText,
        questionText,
        {
          essayScores: {
            overall,
            dimensions,
          },
          ltErrors,
          llmErrors,
          relevanceCheck,
        }
      );
      console.log("Received feedback data:", data);

      const explanationText = data.message || "Let me explain the issues in your essay...";
      // Small delay to allow smooth transition
      await new Promise((resolve) => setTimeout(resolve, 100));
      setExplanation(explanationText);
      setFeedbackMode("explanation");
      setShowTryAgain(false);
    } catch (error) {
      console.error("Error getting teacher feedback:", error);
      // Show error to user with more details
      const errorMessage = error instanceof Error ? error.message : "Unknown error";
      alert(
        `Failed to get feedback: ${errorMessage}\n\nPlease check the browser console for more details.`
      );
    } finally {
      setLoading(false);
    }
  };

  return (
    <div
      id="teacher-feedback-container"
      lang="en"
      translate="yes"
      style={{
        padding: "var(--spacing-lg)",
        backgroundColor: "rgba(139, 69, 19, 0.05)",
        border: "2px solid rgba(139, 69, 19, 0.2)",
        borderRadius: "var(--border-radius-lg)",
        marginBottom: "var(--spacing-lg)",
        position: "relative",
      }}
    >
      <div
        style={{
          display: "flex",
          alignItems: "center",
          gap: "var(--spacing-md)",
          marginBottom: "var(--spacing-md)",
        }}
      >
        <span style={{ fontSize: "32px" }}>👩‍🏫</span>
        <div lang="en">
          <h3
            style={{
              fontSize: "20px",
              fontWeight: 700,
              margin: 0,
              color: "var(--text-primary)",
            }}
          >
            Teacher's Feedback
          </h3>
        </div>
      </div>

      <div
        id="teacher-feedback-text"
        style={{
          fontSize: "16px",
          lineHeight: "1.6",
          color: "var(--text-primary)",
        }}
      >
        <AnimatePresence mode="wait">
          <motion.div
            key={feedbackMode}
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -10 }}
            transition={{ duration: 0.3, ease: [0.4, 0, 0.2, 1] }}
            style={{ marginBottom: "var(--spacing-md)" }}
          >
            {loading ? (
              <div
                style={{
                  display: "flex",
                  alignItems: "center",
                  gap: "var(--spacing-md)",
                  padding: "var(--spacing-lg)",
                  justifyContent: "center",
                }}
              >
                <span
                  className="spinner"
                  style={{
                    display: "inline-block",
                    width: "20px",
                    height: "20px",
                    border: "3px solid rgba(139, 69, 19, 0.2)",
                    borderTopColor: "rgba(139, 69, 19, 0.8)",
                    borderRadius: "50%",
                    animation: "spin 0.6s linear infinite",
                  }}
                />
                <span style={{ color: "var(--text-secondary)", fontSize: "14px" }}>
                  Generating detailed analysis...
                </span>
              </div>
            ) : (
              <ReactMarkdown
                components={{
                  p: ({ children }) => (
                    <p
                      style={{
                        marginBottom: "var(--spacing-md)",
                        fontSize: "16px",
                        lineHeight: "1.6",
                        color: "var(--text-primary)",
                      }}
                    >
                      {children}
                    </p>
                  ),
                  strong: ({ children }) => (
                    <strong style={{ fontWeight: 600, color: "var(--text-primary)" }}>
                      {children}
                    </strong>
                  ),
                  ul: ({ children }) => (
                    <ul
                      style={{
                        marginBottom: "var(--spacing-md)",
                        paddingLeft: "var(--spacing-lg)",
                        fontSize: "16px",
                        lineHeight: "1.6",
                        color: "var(--text-primary)",
                      }}
                    >
                      {children}
                    </ul>
                  ),
                  ol: ({ children }) => (
                    <ol
                      style={{
                        marginBottom: "var(--spacing-md)",
                        paddingLeft: "var(--spacing-lg)",
                        fontSize: "16px",
                        lineHeight: "1.6",
                        color: "var(--text-primary)",
                      }}
                    >
                      {children}
                    </ol>
                  ),
                  li: ({ children }) => (
                    <li
                      style={{
                        marginBottom: "var(--spacing-xs)",
                        lineHeight: "1.6",
                      }}
                    >
                      {children}
                    </li>
                  ),
                  h1: ({ children }) => (
                    <h1
                      style={{
                        fontSize: "20px",
                        fontWeight: 700,
                        marginBottom: "var(--spacing-md)",
                        marginTop: "var(--spacing-md)",
                        color: "var(--text-primary)",
                      }}
                    >
                      {children}
                    </h1>
                  ),
                  h2: ({ children }) => (
                    <h2
                      style={{
                        fontSize: "18px",
                        fontWeight: 600,
                        marginBottom: "var(--spacing-sm)",
                        marginTop: "var(--spacing-md)",
                        color: "var(--text-primary)",
                      }}
                    >
                      {children}
                    </h2>
                  ),
                  h3: ({ children }) => (
                    <h3
                      style={{
                        fontSize: "16px",
                        fontWeight: 600,
                        marginBottom: "var(--spacing-sm)",
                        marginTop: "var(--spacing-md)",
                        color: "var(--text-primary)",
                      }}
                    >
                      {children}
                    </h3>
                  ),
                }}
              >
                {feedbackText}
              </ReactMarkdown>
            )}
          </motion.div>
        </AnimatePresence>

        {feedbackMode === "initial" && aiFeedback?.focusArea && (
          <div
            lang="en"
            style={{
              padding: "var(--spacing-md)",
              backgroundColor: "rgba(102, 126, 234, 0.1)",
              borderLeft: "4px solid var(--primary-color)",
              borderRadius: "var(--spacing-xs)",
              fontSize: "14px",
              marginBottom: "var(--spacing-md)",
              lineHeight: "1.5",
            }}
          >
            💡 Focus on {aiFeedback.focusArea} next time.
          </div>
        )}

        <div
          style={{
            marginTop: "var(--spacing-lg)",
            display: "flex",
            gap: "var(--spacing-md)",
            flexWrap: "wrap",
          }}
        >
          {!submissionId || !answerId || !answerText ? (
            <p
              style={{
                fontSize: "14px",
                color: "var(--text-secondary)",
                fontStyle: "italic",
                lineHeight: "1.5",
              }}
            >
              Note: Interactive feedback requires submission data
            </p>
          ) : feedbackMode === "explanation" ? (
            <motion.button
              onClick={() => {
                setFeedbackMode("initial");
                setShowTryAgain(true);
              }}
              className="btn btn-secondary"
              lang="en"
              style={{
                fontSize: "14px",
                padding: "var(--spacing-sm) var(--spacing-md)",
                minHeight: "44px",
              }}
              whileHover={{ scale: 1.02 }}
              whileTap={{ scale: 0.98 }}
              transition={{ duration: 0.2 }}
            >
              ← Back to Summary
            </motion.button>
          ) : (
            <motion.button
              onClick={handleTryAgain}
              disabled={loading || initialLoading}
              className="btn btn-secondary"
              lang="en"
              style={{
                fontSize: "14px",
                padding: "var(--spacing-sm) var(--spacing-md)",
                minHeight: "44px",
                opacity: loading || initialLoading ? 0.6 : 1,
                cursor: loading || initialLoading ? "not-allowed" : "pointer",
              }}
              whileHover={!loading && !initialLoading ? { scale: 1.02 } : {}}
              whileTap={!loading && !initialLoading ? { scale: 0.98 } : {}}
              transition={{ duration: 0.2 }}
            >
              {loading || initialLoading ? (
                <motion.span
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  transition={{ duration: 0.2 }}
                  style={{ display: "flex", alignItems: "center", gap: "8px" }}
                >
                  <span
                    className="spinner"
                    style={{
                      display: "inline-block",
                      width: "14px",
                      height: "14px",
                      border: "2px solid rgba(0, 0, 0, 0.2)",
                      borderTopColor: "currentColor",
                      borderRadius: "50%",
                      animation: "spin 0.6s linear infinite",
                    }}
                  />
                  {loading ? "Generating detailed analysis..." : "Loading feedback..."}
                </motion.span>
              ) : (
                "View Detailed Analysis"
              )}
            </motion.button>
          )}
        </div>
      </div>
    </div>
  );
}
