"use client";

import { useEffect, useState, useMemo, useCallback } from "react";
import { motion, AnimatePresence } from "framer-motion";
import ReactMarkdown from "react-markdown";
import type { LanguageToolError, RelevanceCheck } from "@writeo/shared";
import { getTeacherFeedback } from "@/app/lib/actions";
import { useAIFeedbackStream } from "@/app/hooks/useAIFeedbackStream";

type FeedbackMode = "initial" | "explanation";

interface Dimensions {
  TA: number;
  CC: number;
  Vocab: number;
  Grammar: number;
  Overall: number;
}

interface AIFeedback {
  message: string;
  focusArea?: string;
  cluesMessage?: string;
  explanationMessage?: string;
}

interface TeacherFeedbackProps {
  overall: number;
  dimensions: Dimensions;
  errorCount: number;
  aiFeedback?: AIFeedback;
  submissionId?: string;
  answerId?: string;
  answerText?: string;
  questionText?: string;
  ltErrors?: LanguageToolError[];
  llmErrors?: LanguageToolError[];
  relevanceCheck?: RelevanceCheck;
}

const CONTAINER_STYLES = {
  padding: "var(--spacing-lg)",
  backgroundColor: "rgba(139, 69, 19, 0.05)",
  border: "2px solid rgba(139, 69, 19, 0.2)",
  borderRadius: "var(--border-radius-lg)",
  marginBottom: "var(--spacing-lg)",
  position: "relative" as const,
};

const TEXT_STYLES = {
  fontSize: "16px",
  lineHeight: "1.6",
  color: "var(--text-primary)",
};

const FOCUS_AREA_STYLES = {
  padding: "var(--spacing-md)",
  backgroundColor: "rgba(102, 126, 234, 0.1)",
  borderLeft: "4px solid var(--primary-color)",
  borderRadius: "var(--spacing-xs)",
  fontSize: "14px",
  marginBottom: "var(--spacing-md)",
  lineHeight: "1.5",
};

const BUTTON_STYLES = {
  fontSize: "14px",
  padding: "var(--spacing-sm) var(--spacing-md)",
  minHeight: "44px",
};

const SPINNER_STYLES = {
  display: "inline-block" as const,
  borderRadius: "50%",
  animation: "spin 0.6s linear infinite",
};

const LOADING_MESSAGES = {
  PREPARING: "Preparing feedback...",
  LOADING: "Loading feedback...",
  GENERATING: "Generating detailed analysis...",
} as const;

const ANIMATION_TRANSITION = { duration: 0.3, ease: [0.4, 0, 0.2, 1] as const };
const BUTTON_TRANSITION = { duration: 0.2 };

/**
 * Spinner component for loading states
 */
function Spinner({
  size = 20,
  color = "rgba(139, 69, 19, 0.8)",
}: {
  size?: number;
  color?: string;
}) {
  return (
    <span
      className="spinner"
      style={{
        ...SPINNER_STYLES,
        width: `${size}px`,
        height: `${size}px`,
        border: `${Math.max(2, size / 10)}px solid ${color.replace("0.8", "0.2")}`,
        borderTopColor: color,
      }}
    />
  );
}

/**
 * Loading indicator component
 */
function LoadingIndicator({ message }: { message: string }) {
  return (
    <div
      style={{
        display: "flex",
        alignItems: "center",
        gap: "var(--spacing-md)",
        padding: "var(--spacing-lg)",
        justifyContent: "center",
      }}
    >
      <Spinner size={20} />
      <span style={{ color: "var(--text-secondary)", fontSize: "14px" }}>{message}</span>
    </div>
  );
}

/**
 * ReactMarkdown custom components with consistent styling
 */
const markdownComponents: React.ComponentProps<typeof ReactMarkdown>["components"] = {
  p: ({ children }) => (
    <p style={{ ...TEXT_STYLES, marginBottom: "var(--spacing-md)" }}>{children}</p>
  ),
  strong: ({ children }) => (
    <strong style={{ fontWeight: 600, color: "var(--text-primary)" }}>{children}</strong>
  ),
  ul: ({ children }) => (
    <ul
      style={{
        ...TEXT_STYLES,
        marginBottom: "var(--spacing-md)",
        paddingLeft: "var(--spacing-lg)",
      }}
    >
      {children}
    </ul>
  ),
  ol: ({ children }) => (
    <ol
      style={{
        ...TEXT_STYLES,
        marginBottom: "var(--spacing-md)",
        paddingLeft: "var(--spacing-lg)",
      }}
    >
      {children}
    </ol>
  ),
  li: ({ children }) => (
    <li style={{ marginBottom: "var(--spacing-xs)", lineHeight: "1.6" }}>{children}</li>
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
};

/**
 * TeacherFeedback - Provides AI-generated feedback from a helpful teacher
 * Supports interactive flow: initial feedback -> clues -> full explanation
 */
export function TeacherFeedback({
  overall,
  dimensions,
  errorCount: _errorCount,
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
  const [feedbackMode, setFeedbackMode] = useState<FeedbackMode>("initial");
  const [explanation, setExplanation] = useState<string | null>(
    aiFeedback?.explanationMessage || null,
  );
  const [initialMessage, setInitialMessage] = useState<string | null>(aiFeedback?.message || null);
  const [initialLoading, setInitialLoading] = useState(false);
  const [initialError, setInitialError] = useState<string | null>(null);
  const [hasRequestedClues, setHasRequestedClues] = useState(false);
  const [loading, setLoading] = useState(false);

  // Use streaming hook for teacher feedback
  const {
    feedback: streamedFeedback,
    isStreaming,
    startStream,
    stopStream,
  } = useAIFeedbackStream();

  // Consolidate loading state
  const isLoading = loading || initialLoading || isStreaming;

  // Compute feedback text using useMemo
  const feedbackText = useMemo(() => {
    if (feedbackMode === "explanation" && streamedFeedback) {
      return streamedFeedback;
    }
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
      return LOADING_MESSAGES.PREPARING;
    }
    if (initialError) {
      return LOADING_MESSAGES.LOADING;
    }
    return LOADING_MESSAGES.PREPARING;
  }, [
    feedbackMode,
    streamedFeedback,
    explanation,
    initialMessage,
    aiFeedback?.message,
    initialLoading,
    initialError,
  ]);

  // Prepare assessment data
  const assessmentData = useMemo(
    () => ({
      essayScores: { overall, dimensions },
      ltErrors: ltErrors || [],
      llmErrors: llmErrors || [],
    }),
    [overall, dimensions, ltErrors, llmErrors],
  );

  // Ensure the short Groq encouragement is available even if the async pipeline
  // hasn't stored it yet (common when submissions run in async mode).
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
      essayScores: { overall, dimensions },
      ltErrors,
      llmErrors,
      relevanceCheck,
    })
      .then((data: { message: string; focusArea?: string }) => {
        if (!cancelled) {
          setInitialMessage(data.message || null);
          setHasRequestedClues(true);
        }
      })
      .catch((error: unknown) => {
        if (!cancelled) {
          console.error("Error auto-fetching teacher clues:", error);
          const message = error instanceof Error ? error.message : "Unknown error";
          setInitialError(message);
        }
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
    questionText,
    overall,
    dimensions,
    ltErrors,
    llmErrors,
    relevanceCheck,
    aiFeedback?.message,
    initialMessage,
    initialLoading,
    hasRequestedClues,
  ]);

  const handleTryAgain = useCallback(async () => {
    if (!submissionId || !answerId || !answerText) {
      console.error("Missing required data:", { submissionId, answerId, answerText: !!answerText });
      alert(
        "Cannot get feedback: Missing required information. Please ensure the essay was submitted correctly.",
      );
      return;
    }

    setLoading(true);
    setExplanation("");
    setFeedbackMode("explanation");

    try {
      console.log("Starting streaming teacher feedback", {
        submissionId,
        answerId,
        answerTextLength: answerText.length,
      });

      await startStream(submissionId, answerId, answerText, questionText, assessmentData);
    } catch (error) {
      console.error("Error starting teacher feedback stream:", error);
      const errorMessage = error instanceof Error ? error.message : "Unknown error";
      alert(
        `Failed to get feedback: ${errorMessage}\n\nPlease check the browser console for more details.`,
      );
      setLoading(false);
    }
  }, [submissionId, answerId, answerText, questionText, assessmentData, startStream]);

  const handleBackToSummary = useCallback(() => {
    setFeedbackMode("initial");
  }, []);

  // Update explanation when streamed feedback arrives
  useEffect(() => {
    if (streamedFeedback) {
      setExplanation(streamedFeedback);
      setLoading(false);
    }
  }, [streamedFeedback]);

  // Handle streaming completion
  useEffect(() => {
    if (!isStreaming && streamedFeedback) {
      setLoading(false);
    }
  }, [isStreaming, streamedFeedback]);

  // Cleanup on unmount
  useEffect(() => {
    return stopStream;
  }, [stopStream]);

  const hasRequiredData = Boolean(submissionId && answerId && answerText);
  const showLoading = isLoading && !streamedFeedback;

  return (
    <div id="teacher-feedback-container" lang="en" translate="yes" style={CONTAINER_STYLES}>
      <div
        style={{
          display: "flex",
          alignItems: "center",
          gap: "var(--spacing-md)",
          marginBottom: "var(--spacing-md)",
        }}
      >
        <span style={{ fontSize: "32px" }}>👩‍🏫</span>
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

      <div id="teacher-feedback-text" style={TEXT_STYLES}>
        <AnimatePresence mode="wait">
          <motion.div
            key={feedbackMode}
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -10 }}
            transition={ANIMATION_TRANSITION}
            style={{ marginBottom: "var(--spacing-md)" }}
          >
            {showLoading ? (
              <LoadingIndicator message={LOADING_MESSAGES.GENERATING} />
            ) : (
              <ReactMarkdown components={markdownComponents}>{feedbackText}</ReactMarkdown>
            )}
          </motion.div>
        </AnimatePresence>

        {feedbackMode === "initial" && aiFeedback?.focusArea && (
          <div style={FOCUS_AREA_STYLES}>💡 Focus on {aiFeedback.focusArea} next time.</div>
        )}

        <div
          style={{
            marginTop: "var(--spacing-lg)",
            display: "flex",
            gap: "var(--spacing-md)",
            flexWrap: "wrap",
          }}
        >
          {!hasRequiredData ? (
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
              onClick={handleBackToSummary}
              className="btn btn-secondary"
              style={BUTTON_STYLES}
              whileHover={{ scale: 1.02 }}
              whileTap={{ scale: 0.98 }}
              transition={BUTTON_TRANSITION}
            >
              ← Back to Summary
            </motion.button>
          ) : (
            <motion.button
              onClick={handleTryAgain}
              disabled={isLoading}
              className="btn btn-secondary"
              style={{
                ...BUTTON_STYLES,
                opacity: isLoading ? 0.6 : 1,
                cursor: isLoading ? "not-allowed" : "pointer",
              }}
              whileHover={!isLoading ? { scale: 1.02 } : {}}
              whileTap={!isLoading ? { scale: 0.98 } : {}}
              transition={BUTTON_TRANSITION}
            >
              {isLoading ? (
                <motion.span
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  transition={BUTTON_TRANSITION}
                  style={{ display: "flex", alignItems: "center", gap: "8px" }}
                >
                  <Spinner size={14} color="currentColor" />
                  {loading || isStreaming ? LOADING_MESSAGES.GENERATING : LOADING_MESSAGES.LOADING}
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
