"use client";

import { useState, useMemo, useCallback } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { countWords, MIN_ESSAY_WORDS, MAX_ESSAY_WORDS } from "@writeo/shared";
import { pluralize } from "@/app/lib/utils/text-utils";

interface EditableEssayProps {
  initialText: string;
  questionId?: string;
  questionText?: string;
  parentSubmissionId?: string;
  onSubmit: (editedText: string, parentSubmissionId?: string) => Promise<void>;
}

const SUCCESS_MESSAGE_DURATION = 3000;
const TEXTAREA_MIN_HEIGHT = "300px";
const REFLECTION_MIN_HEIGHT = "60px";

const BROWN_BORDER = "rgba(139, 69, 19, 0.2)";
const BROWN_BORDER_LIGHT = "rgba(139, 69, 19, 0.3)";
const BROWN_BG = "rgba(139, 69, 19, 0.1)";

type WordCountStatus = "too-short" | "valid" | "too-long";

function getWordCountStatus(wordCount: number): WordCountStatus {
  if (wordCount < MIN_ESSAY_WORDS) return "too-short";
  if (wordCount > MAX_ESSAY_WORDS) return "too-long";
  return "valid";
}

/**
 * EditableEssay - Allows users to edit their essay and resubmit
 */
export function EditableEssay({
  initialText,
  questionId: _questionId,
  questionText,
  parentSubmissionId,
  onSubmit,
}: EditableEssayProps) {
  const [showQuestion, setShowQuestion] = useState(false);
  const [editedText, setEditedText] = useState(initialText);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [reflection, setReflection] = useState("");
  const [showSuccess, setShowSuccess] = useState(false);

  const hasChanges = useMemo(() => {
    const trimmedEdited = editedText.trim();
    const trimmedInitial = initialText.trim();
    return trimmedEdited !== trimmedInitial && trimmedEdited.length > 0;
  }, [editedText, initialText]);

  const handleChange = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    setEditedText(e.target.value);
  };

  const wordCount = useMemo(() => countWords(editedText), [editedText]);
  const wordCountStatus = useMemo(() => getWordCountStatus(wordCount), [wordCount]);

  const validateSubmission = useCallback((): string | null => {
    if (!hasChanges) {
      return "Please make some changes to your essay before resubmitting. Fix the highlighted errors to improve your score.";
    }

    if (wordCountStatus === "too-short") {
      return `Your essay is too short. Please write at least ${MIN_ESSAY_WORDS} ${pluralize(MIN_ESSAY_WORDS, "word")} (currently ${wordCount} ${pluralize(wordCount, "word")}).`;
    }

    if (wordCountStatus === "too-long") {
      return `Your essay is too long. Please keep it under ${MAX_ESSAY_WORDS} ${pluralize(MAX_ESSAY_WORDS, "word")} (currently ${wordCount} ${pluralize(wordCount, "word")}).`;
    }

    return null;
  }, [hasChanges, wordCountStatus, wordCount]);

  const handleSubmit = useCallback(async () => {
    const validationError = validateSubmission();
    if (validationError) {
      alert(validationError);
      return;
    }

    setIsSubmitting(true);
    try {
      await onSubmit(editedText, parentSubmissionId);
      setShowSuccess(true);
      setTimeout(() => setShowSuccess(false), SUCCESS_MESSAGE_DURATION);
    } catch (error) {
      console.error("Error submitting edited essay:", error);
      alert(`Failed to submit: ${error instanceof Error ? error.message : "Unknown error"}`);
    } finally {
      setIsSubmitting(false);
    }
  }, [validateSubmission, editedText, parentSubmissionId, onSubmit]);

  const handleReset = useCallback(() => {
    setEditedText(initialText);
  }, [initialText]);

  return (
    <div
      lang="en"
      style={{
        marginTop: "var(--spacing-lg)",
        padding: "var(--spacing-lg)",
        backgroundColor: "var(--bg-secondary)",
        border: `2px solid ${BROWN_BORDER}`,
        borderRadius: "var(--border-radius-lg)",
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
        <span style={{ fontSize: "24px" }}>✏️</span>
        <h3
          style={{
            fontSize: "18px",
            fontWeight: 600,
            margin: 0,
            color: "var(--text-primary)",
          }}
        >
          Improve Your Writing
        </h3>
      </div>

      <p
        style={{
          fontSize: "16px",
          color: "var(--text-secondary)",
          marginBottom: "var(--spacing-md)",
          lineHeight: "1.5",
        }}
      >
        Make changes based on the feedback, then submit another draft to see your progress. Try
        correcting the highlighted areas and click "Submit Improved Draft" to check again!
      </p>

      {/* Show question text if available */}
      {questionText && (
        <div
          style={{
            marginBottom: "var(--spacing-md)",
            padding: "var(--spacing-sm) var(--spacing-md)",
            backgroundColor: BROWN_BG,
            border: `1px solid ${BROWN_BORDER}`,
            borderRadius: "var(--border-radius)",
          }}
        >
          <button
            onClick={() => setShowQuestion(!showQuestion)}
            style={{
              background: "none",
              border: "none",
              padding: 0,
              fontSize: "14px",
              fontWeight: 600,
              color: "var(--primary-color)",
              cursor: "pointer",
              display: "flex",
              alignItems: "center",
              gap: "var(--spacing-xs)",
            }}
          >
            {showQuestion ? "▼" : "▶"} {showQuestion ? "Hide" : "Show"} Question
          </button>
          {showQuestion && (
            <p
              style={{
                marginTop: "var(--spacing-sm)",
                fontSize: "14px",
                lineHeight: "1.6",
                color: "var(--text-primary)",
              }}
            >
              {questionText}
            </p>
          )}
        </div>
      )}

      {/* Success message */}
      <AnimatePresence>
        {showSuccess && (
          <motion.div
            initial={{ opacity: 0, y: -10, scale: 0.95 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: -10, scale: 0.95 }}
            transition={{ duration: 0.3, ease: [0.4, 0, 0.2, 1] }}
            style={{
              marginBottom: "var(--spacing-md)",
              padding: "var(--spacing-md)",
              backgroundColor: "rgba(16, 185, 129, 0.1)",
              border: "1px solid rgba(16, 185, 129, 0.3)",
              borderRadius: "var(--border-radius)",
              color: "var(--secondary-accent)",
              fontWeight: 600,
              textAlign: "center",
            }}
          >
            ✅ Draft submitted successfully! Analyzing your improvements...
          </motion.div>
        )}
      </AnimatePresence>

      <textarea
        value={editedText}
        onChange={handleChange}
        disabled={isSubmitting}
        translate="no"
        lang="en"
        style={{
          width: "100%",
          minHeight: TEXTAREA_MIN_HEIGHT,
          padding: "var(--spacing-md)",
          fontSize: "16px",
          lineHeight: "1.5",
          fontFamily: "inherit",
          border: `1px solid ${BROWN_BORDER_LIGHT}`,
          borderRadius: "var(--border-radius)",
          backgroundColor: "var(--bg-primary)",
          color: "var(--text-primary)",
          resize: "vertical",
          marginBottom: "var(--spacing-sm)",
        }}
      />

      {/* Reflection Prompt */}
      {hasChanges && (
        <div
          style={{
            marginBottom: "var(--spacing-md)",
            padding: "var(--spacing-md)",
            backgroundColor: "var(--primary-bg-light)",
            borderRadius: "var(--border-radius)",
          }}
        >
          <p style={{ marginBottom: "var(--spacing-sm)", fontSize: "14px", fontWeight: 600 }}>
            💭 Reflection (optional)
          </p>
          <p
            style={{
              marginBottom: "var(--spacing-sm)",
              fontSize: "14px",
              color: "var(--text-secondary)",
              lineHeight: "1.5",
            }}
          >
            What did you change this time? (Optional - helps you reflect on your improvements)
          </p>
          <textarea
            value={reflection}
            onChange={(e) => setReflection(e.target.value)}
            placeholder="E.g., I fixed the subject-verb agreement errors and added more examples..."
            style={{
              width: "100%",
              minHeight: REFLECTION_MIN_HEIGHT,
              padding: "var(--spacing-sm)",
              fontSize: "14px",
              fontFamily: "inherit",
              border: `1px solid ${BROWN_BORDER_LIGHT}`,
              borderRadius: "var(--border-radius)",
              backgroundColor: "var(--bg-primary)",
              color: "var(--text-primary)",
              resize: "vertical",
              lineHeight: "1.5",
            }}
            lang="en"
          />
        </div>
      )}

      {/* Word count display */}
      <div
        style={{
          marginBottom: "var(--spacing-md)",
          fontSize: "14px",
          color: "var(--text-secondary)",
          display: "flex",
          gap: "var(--spacing-md)",
          alignItems: "center",
        }}
      >
        <span>
          {wordCount} {pluralize(wordCount, "word")}
        </span>
        {wordCountStatus === "too-short" && (
          <span style={{ color: "var(--error-color)", fontWeight: 600 }}>
            (Need at least {MIN_ESSAY_WORDS} {pluralize(MIN_ESSAY_WORDS, "word")})
          </span>
        )}
        {wordCountStatus === "valid" && <span style={{ color: "var(--secondary-accent)" }}>✓</span>}
        {wordCountStatus === "too-long" && (
          <span style={{ color: "var(--error-color)", fontWeight: 600 }}>
            (Too long - maximum {MAX_ESSAY_WORDS} {pluralize(MAX_ESSAY_WORDS, "word")})
          </span>
        )}
      </div>

      <div
        style={{
          display: "flex",
          gap: "var(--spacing-md)",
          alignItems: "center",
          flexWrap: "wrap",
        }}
      >
        <motion.button
          onClick={handleSubmit}
          disabled={isSubmitting || !hasChanges}
          className="btn btn-primary"
          style={{
            fontSize: "14px",
            padding: "var(--spacing-sm) var(--spacing-lg)",
            opacity: isSubmitting || !hasChanges ? 0.6 : 1,
            cursor: isSubmitting || !hasChanges ? "not-allowed" : "pointer",
          }}
          whileHover={!isSubmitting && hasChanges ? { scale: 1.02 } : {}}
          whileTap={!isSubmitting && hasChanges ? { scale: 0.98 } : {}}
          transition={{ duration: 0.2 }}
        >
          {isSubmitting ? (
            <motion.span
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ duration: 0.2 }}
              style={{ display: "flex", alignItems: "center", gap: "8px" }}
            >
              <span
                style={{
                  display: "inline-block",
                  width: "14px",
                  height: "14px",
                  border: "2px solid rgba(255, 255, 255, 0.3)",
                  borderTopColor: "white",
                  borderRadius: "50%",
                  animation: "spin 0.6s linear infinite",
                }}
              />
              Submitting your improved draft…
            </motion.span>
          ) : (
            "Submit Improved Draft"
          )}
        </motion.button>

        {hasChanges && (
          <>
            <button
              onClick={handleReset}
              disabled={isSubmitting}
              className="btn btn-secondary"
              style={{
                fontSize: "14px",
                padding: "var(--spacing-sm) var(--spacing-lg)",
              }}
            >
              Reset Changes
            </button>
            <span
              style={{
                fontSize: "14px",
                color: "var(--text-secondary)",
                fontStyle: "italic",
              }}
            >
              You have unsaved changes
            </span>
          </>
        )}
      </div>
    </div>
  );
}
