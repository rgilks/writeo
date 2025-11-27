"use client";

import { useState, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { countWords, MIN_ESSAY_WORDS, MAX_ESSAY_WORDS } from "@writeo/shared";

interface EditableEssayProps {
  initialText: string;
  questionId?: string;
  questionText?: string; // The question/prompt text
  parentSubmissionId?: string; // For draft tracking
  onSubmit: (editedText: string, parentSubmissionId?: string) => Promise<void>;
}

/**
 * EditableEssay - Allows users to edit their essay and resubmit
 */
export function EditableEssay({
  initialText,
  questionId,
  questionText,
  parentSubmissionId,
  onSubmit,
}: EditableEssayProps) {
  const [showQuestion, setShowQuestion] = useState(false);
  const [editedText, setEditedText] = useState(initialText);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [hasChanges, setHasChanges] = useState(false);
  const [reflection, setReflection] = useState("");
  const [showReflection, setShowReflection] = useState(false);
  const [showSuccess, setShowSuccess] = useState(false);

  // Sync hasChanges state whenever editedText or initialText changes
  useEffect(() => {
    const trimmedEdited = editedText.trim();
    const trimmedInitial = initialText.trim();
    setHasChanges(trimmedEdited !== trimmedInitial && trimmedEdited.length > 0);
  }, [editedText, initialText]);

  const handleChange = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    const newText = e.target.value;
    setEditedText(newText);
    // Also update immediately for better responsiveness
    const trimmedNew = newText.trim();
    const trimmedInitial = initialText.trim();
    setHasChanges(trimmedNew !== trimmedInitial && trimmedNew.length > 0);
  };

  // Fallback handler for browser automation that may not trigger onChange
  const handleInput = (e: React.FormEvent<HTMLTextAreaElement>) => {
    const newText = (e.target as HTMLTextAreaElement).value;
    setEditedText(newText);
  };

  // Calculate word count
  const wordCount = countWords(editedText);
  const MIN_WORDS = MIN_ESSAY_WORDS;
  const MAX_WORDS = MAX_ESSAY_WORDS; // Soft cap - warn but allow

  const handleSubmit = async () => {
    if (!hasChanges) {
      alert(
        "Please make some changes to your essay before resubmitting. Fix the highlighted errors to improve your score.",
      );
      return;
    }

    // Validate word count
    if (wordCount < MIN_WORDS) {
      alert(
        `Your essay is too short. Please write at least ${MIN_WORDS} words (currently ${wordCount} words).`,
      );
      return;
    }

    if (wordCount > MAX_WORDS) {
      alert(
        `Your essay is too long. Please keep it under ${MAX_WORDS} words (currently ${wordCount} words).`,
      );
      return;
    }

    setIsSubmitting(true);
    try {
      await onSubmit(editedText, parentSubmissionId);
      setHasChanges(false);
      setShowSuccess(true);
      // Hide success message after 3 seconds
      setTimeout(() => setShowSuccess(false), 3000);
    } catch (error) {
      console.error("Error submitting edited essay:", error);
      alert(`Failed to submit: ${error instanceof Error ? error.message : "Unknown error"}`);
    } finally {
      setIsSubmitting(false);
    }
  };

  const handleReset = () => {
    setEditedText(initialText);
    setHasChanges(false);
  };

  return (
    <div
      lang="en"
      style={{
        marginTop: "var(--spacing-lg)",
        padding: "var(--spacing-lg)",
        backgroundColor: "var(--bg-secondary)",
        border: "2px solid rgba(139, 69, 19, 0.2)",
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
          lang="en"
        >
          Improve Your Writing
        </h3>
      </div>

      <p
        lang="en"
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
          lang="en"
          style={{
            marginBottom: "var(--spacing-md)",
            padding: "var(--spacing-sm) var(--spacing-md)",
            backgroundColor: "rgba(139, 69, 19, 0.1)",
            border: "1px solid rgba(139, 69, 19, 0.2)",
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
            lang="en"
          >
            {showQuestion ? "▼" : "▶"} {showQuestion ? "Hide" : "Show"} Question
          </button>
          {showQuestion && (
            <p
              lang="en"
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
            lang="en"
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
        onInput={handleInput}
        disabled={isSubmitting}
        translate="no"
        lang="en"
        style={{
          width: "100%",
          minHeight: "300px",
          padding: "var(--spacing-md)",
          fontSize: "16px",
          lineHeight: "1.5",
          fontFamily: "inherit",
          border: "1px solid rgba(139, 69, 19, 0.3)",
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
          lang="en"
          style={{
            marginBottom: "var(--spacing-md)",
            padding: "var(--spacing-md)",
            backgroundColor: "rgba(102, 126, 234, 0.1)",
            borderRadius: "var(--border-radius)",
          }}
        >
          <p
            style={{ marginBottom: "var(--spacing-sm)", fontSize: "14px", fontWeight: 600 }}
            lang="en"
          >
            💭 Reflection (optional)
          </p>
          <p
            style={{
              marginBottom: "var(--spacing-sm)",
              fontSize: "14px",
              color: "var(--text-secondary)",
              lineHeight: "1.5",
            }}
            lang="en"
          >
            What did you change this time? (Optional - helps you reflect on your improvements)
          </p>
          <textarea
            value={reflection}
            onChange={(e) => setReflection(e.target.value)}
            placeholder="E.g., I fixed the subject-verb agreement errors and added more examples..."
            style={{
              width: "100%",
              minHeight: "60px",
              padding: "var(--spacing-sm)",
              fontSize: "14px",
              fontFamily: "inherit",
              border: "1px solid rgba(139, 69, 19, 0.3)",
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
          lang="en"
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
          <button
            onClick={handleReset}
            disabled={isSubmitting}
            className="btn btn-secondary"
            lang="en"
            style={{
              fontSize: "14px",
              padding: "var(--spacing-sm) var(--spacing-lg)",
            }}
          >
            Reset Changes
          </button>
        )}

        {hasChanges && (
          <span
            lang="en"
            style={{
              fontSize: "14px",
              color: "var(--text-secondary)",
              fontStyle: "italic",
            }}
          >
            You have unsaved changes
          </span>
        )}
      </div>
    </div>
  );
}
