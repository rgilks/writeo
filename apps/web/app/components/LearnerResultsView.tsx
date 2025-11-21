"use client";

import { useState, useEffect } from "react";
import { useRouter } from "next/navigation";
import Link from "next/link";
import { TeacherFeedback } from "./TeacherFeedback";
import { HeatMapText } from "./HeatMapText";
import { EditableEssay } from "./EditableEssay";
import { ProgressChart } from "./ProgressChart";
import { CEFRInfo, CEFRBadge } from "./CEFRInfo";
import { CelebrationMessage } from "./CelebrationMessage";
import { submitEssay } from "@/app/lib/actions";
import { useDraftStore } from "@/app/lib/stores/draft-store";
import { usePreferencesStore } from "@/app/lib/stores/preferences-store";
import { extractErrorIds, getTopErrorTypes } from "@/app/lib/utils/progress";
import type { AssessmentResults, LanguageToolError } from "@writeo/shared";

// Component for clickable error type with explanation
function ErrorTypeItem({
  type,
  count,
  examples,
}: {
  type: string;
  count: number;
  examples: LanguageToolError[];
}) {
  const [isExpanded, setIsExpanded] = useState(false);

  const getErrorExplanation = (errorType: string): string => {
    const explanations: Record<string, string> = {
      "Subject-verb agreement":
        "The subject and verb must agree in number (singular/plural). Example: 'He go' should be 'He goes'.",
      "Verb tense":
        "Use consistent verb tenses. Check if actions happened in the past, present, or future.",
      "Article use":
        "Use 'a' before consonant sounds, 'an' before vowel sounds, and 'the' for specific things.",
      Preposition:
        "Prepositions show relationships (in, on, at, with, etc.). Choose the correct one for the context.",
      Spelling:
        "Check spelling carefully. Common mistakes include homophones (words that sound the same but are spelled differently).",
      Punctuation: "Use punctuation marks correctly: periods, commas, question marks, etc.",
      "Word order": "English follows a specific word order: Subject-Verb-Object.",
      "Grammar error": "A grammatical mistake that affects clarity or correctness.",
    };
    return (
      explanations[errorType] ||
      `This type of error appears ${count} ${count === 1 ? "time" : "times"} in your essay.`
    );
  };

  return (
    <li style={{ marginBottom: "var(--spacing-sm)", fontSize: "16px" }} lang="en">
      <div
        style={{
          display: "flex",
          alignItems: "center",
          gap: "var(--spacing-sm)",
          cursor: "pointer",
        }}
        onClick={() => setIsExpanded(!isExpanded)}
        lang="en"
      >
        <span style={{ fontSize: "12px", color: "var(--text-secondary)" }}>
          {isExpanded ? "▼" : "▶"}
        </span>
        <strong>{type}</strong>
        <span style={{ color: "var(--text-secondary)" }}>
          ({count} {count === 1 ? "time" : "times"})
        </span>
      </div>
      {isExpanded && (
        <div
          style={{
            marginTop: "var(--spacing-xs)",
            marginLeft: "var(--spacing-md)",
            padding: "var(--spacing-sm)",
            backgroundColor: "rgba(102, 126, 234, 0.1)",
            borderRadius: "var(--border-radius)",
            fontSize: "14px",
            lineHeight: "1.5",
          }}
          lang="en"
        >
          <p style={{ marginBottom: "var(--spacing-xs)", fontWeight: 600 }} lang="en">
            {getErrorExplanation(type)}
          </p>
          {examples.length > 0 && (
            <div lang="en">
              <p style={{ marginBottom: "4px", fontSize: "12px", fontWeight: 600 }} lang="en">
                Examples from your essay:
              </p>
              <ul
                style={{ margin: 0, paddingLeft: "var(--spacing-md)", fontSize: "13px" }}
                lang="en"
              >
                {examples.map((err, idx) => (
                  <li key={idx} lang="en">
                    {err.message}
                    {err.suggestions && err.suggestions.length > 0 && (
                      <span style={{ color: "var(--text-secondary)" }}>
                        {" "}
                        → Try: {err.suggestions[0]}
                      </span>
                    )}
                  </li>
                ))}
              </ul>
            </div>
          )}
        </div>
      )}
    </li>
  );
}

interface LearnerResultsViewProps {
  data: AssessmentResults;
  answerText: string;
  processingTime?: number | null;
}

const getScoreColor = (score: number): string => {
  if (score >= 7.5) return "#10b981"; // green
  if (score >= 6.5) return "#3b82f6"; // blue
  if (score >= 5.5) return "#f59e0b"; // amber
  return "#ef4444"; // red
};

const getScoreLabel = (score: number): string => {
  if (score >= 7.5) return "Excellent";
  if (score >= 6.5) return "Good";
  if (score >= 5.5) return "Fair";
  return "Needs Improvement";
};

// CEFR level descriptors
const getCEFRDescriptor = (level: string): string => {
  const descriptors: Record<string, string> = {
    A2: "Can write simple connected text on familiar topics.",
    B1: "Can write simple connected text on topics which are familiar or of personal interest.",
    B2: "Can write clear, detailed text on a wide range of subjects.",
    C1: "Can write clear, well-structured text on complex subjects.",
    C2: "Can write clear, smoothly flowing text in an appropriate style.",
  };
  return descriptors[level] || "Writing proficiency level.";
};

// Map score to CEFR level
const mapScoreToCEFR = (score: number): string => {
  if (score >= 8.5) return "C2";
  if (score >= 7.0) return "C1";
  if (score >= 5.5) return "B2";
  if (score >= 4.0) return "B1";
  return "A2";
};

// Get CEFR level thresholds
const getCEFRThresholds = (): Record<string, { min: number; max: number }> => {
  return {
    A2: { min: 0, max: 4.0 },
    B1: { min: 4.0, max: 5.5 },
    B2: { min: 5.5, max: 7.0 },
    C1: { min: 7.0, max: 8.5 },
    C2: { min: 8.5, max: 9.0 },
  };
};

// Calculate progress toward next CEFR level
const calculateCEFRProgress = (
  score: number
): { current: string; next: string; progress: number; scoreToNext: number } => {
  const thresholds = getCEFRThresholds();
  const current = mapScoreToCEFR(score);
  const cefrLevels = ["A2", "B1", "B2", "C1", "C2"];
  const currentIndex = cefrLevels.indexOf(current);
  const nextIndex = currentIndex < cefrLevels.length - 1 ? currentIndex + 1 : currentIndex;
  const next = cefrLevels[nextIndex];

  if (current === "C2") {
    // Already at max level
    return { current, next: current, progress: 100, scoreToNext: 0 };
  }

  const currentThreshold = thresholds[current];
  const nextThreshold = thresholds[next];

  // Calculate progress within current level
  const range = nextThreshold.min - currentThreshold.min;
  const position = score - currentThreshold.min;
  const progress = Math.min(100, Math.max(0, (position / range) * 100));
  const scoreToNext = nextThreshold.min - score;

  return { current, next, progress, scoreToNext };
};

// Calculate confidence level (low/medium/high) based on score range
const getConfidenceLevel = (score: number): "low" | "medium" | "high" => {
  // Scores near band boundaries are less confident
  const distanceFromBoundary = Math.min(score % 0.5, 0.5 - (score % 0.5));

  if (distanceFromBoundary < 0.1) return "low"; // Very close to boundary
  if (distanceFromBoundary < 0.2) return "medium"; // Somewhat close to boundary
  return "high"; // Well within a band
};

const getConfidenceBadgeColor = (confidence: "low" | "medium" | "high"): string => {
  if (confidence === "high") return "var(--secondary-accent)"; // green
  if (confidence === "medium") return "var(--warm-accent)"; // amber
  return "var(--error-color)"; // red
};

const getCEFRLabel = (level: string): string => {
  const labels: Record<string, string> = {
    A2: "Elementary",
    B1: "Intermediate",
    B2: "Upper Intermediate",
    C1: "Advanced",
    C2: "Proficient",
  };
  return labels[level] || level;
};

export function LearnerResultsView({ data, answerText, processingTime }: LearnerResultsViewProps) {
  const router = useRouter();
  const [isResubmitting, setIsResubmitting] = useState(false);
  const [isFeedbackRevealed, setIsFeedbackRevealed] = useState(false);

  // Use selectors to get only the functions we need (functions are stable, so this is fine)
  // But we could also use individual selectors if we wanted to be more explicit
  const addDraft = useDraftStore((state) => state.addDraft);
  const getDraftHistory = useDraftStore((state) => state.getDraftHistory);
  const trackFixedErrors = useDraftStore((state) => state.trackFixedErrors);

  const parts = data.results?.parts || [];
  const firstPart = parts[0];
  // Get assessor-results from the first answer
  const firstAnswer = firstPart?.answers?.[0];
  const assessorResults = firstAnswer?.["assessor-results"] || [];

  // Find essay assessor
  const essayAssessor = assessorResults.find((a: any) => a.id === "T-AES-ESSAY");
  const overall = essayAssessor?.overall || 0;
  const rawDimensions = essayAssessor?.dimensions || {};
  const dimensions = {
    TA: rawDimensions.TA ?? 0,
    CC: rawDimensions.CC ?? 0,
    Vocab: rawDimensions.Vocab ?? 0,
    Grammar: rawDimensions.Grammar ?? 0,
    Overall: rawDimensions.Overall ?? 0,
  };

  // Find the weakest dimension for highlighting
  const lowestDim = Object.entries(dimensions)
    .filter(([k]) => k !== "Overall")
    .sort(([, a], [, b]) => (a ?? 0) - (b ?? 0))[0];

  // Find LanguageTool assessor
  const ltAssessor = assessorResults.find((a: any) => a.id === "T-GEC-LT");
  const ltErrors: LanguageToolError[] = Array.isArray(ltAssessor?.errors) ? ltAssessor.errors : [];

  // Find LLM assessor (separate from LanguageTool)
  const llmAssessor = assessorResults.find((a: any) => a.id === "T-GEC-LLM");
  const llmErrors: LanguageToolError[] = Array.isArray(llmAssessor?.errors)
    ? llmAssessor.errors
    : [];

  // Combine errors from both assessors for display
  const grammarErrors: LanguageToolError[] = [...ltErrors, ...llmErrors];

  // Find Teacher feedback assessor
  const teacherAssessor = assessorResults.find((a: any) => a.id === "T-TEACHER-FEEDBACK");
  const teacherFeedback = teacherAssessor?.meta
    ? {
        message: teacherAssessor.meta.message as string,
        focusArea: teacherAssessor.meta.focusArea as string | undefined,
        cluesMessage: (teacherAssessor.meta as any).cluesMessage as string | undefined,
        explanationMessage: (teacherAssessor.meta as any).explanationMessage as string | undefined,
      }
    : undefined;

  // Extract submission ID from URL
  const submissionId =
    typeof window !== "undefined"
      ? window.location.pathname.split("/results/")[1]?.split("/")[0]
      : undefined;

  // Get draft tracking info from metadata
  const draftNumber = (data.meta?.draftNumber as number) || 1;
  const parentSubmissionId = data.meta?.parentSubmissionId as string | undefined;
  const draftHistory =
    (data.meta?.draftHistory as Array<{
      draftNumber: number;
      timestamp: string;
      wordCount: number;
      errorCount: number;
      overallScore?: number;
    }>) || [];

  // Calculate draft comparison if we have previous drafts
  const previousDraft = draftHistory.length > 1 ? draftHistory[draftHistory.length - 2] : null;
  const currentDraft = draftHistory.length > 0 ? draftHistory[draftHistory.length - 1] : null;

  const wordCountDiff =
    previousDraft && currentDraft ? currentDraft.wordCount - previousDraft.wordCount : null;
  const errorCountDiff =
    previousDraft && currentDraft ? currentDraft.errorCount - previousDraft.errorCount : null;
  const scoreDiff =
    previousDraft && currentDraft && previousDraft.overallScore && currentDraft.overallScore
      ? currentDraft.overallScore - previousDraft.overallScore
      : null;

  // Calculate pattern insights: most common error types
  const topErrorTypes = getTopErrorTypes(grammarErrors, 3);

  // Get answer ID and text from metadata
  const answerTexts = data.meta?.answerTexts as Record<string, string> | undefined;
  const answerId = answerTexts ? Object.keys(answerTexts)[0] : undefined;

  // Get question text from metadata
  const questionTexts = data.meta?.questionTexts as Record<string, string> | undefined;
  const questionText = questionTexts && answerId ? questionTexts[answerId] : "";

  // Ensure we have answerText - use from prop or metadata
  const finalAnswerText = answerText || (answerTexts && answerId ? answerTexts[answerId] : "");

  // Store draft in Zustand store
  useEffect(() => {
    if (submissionId && overall > 0) {
      try {
        const cefrLevel = mapScoreToCEFR(overall);
        const wordCount = finalAnswerText.split(/\s+/).filter((w) => w.length > 0).length;

        const errorIds = extractErrorIds(grammarErrors, finalAnswerText);

        // Create draft data object
        const draftData = {
          draftNumber,
          submissionId,
          timestamp: new Date().toISOString(),
          wordCount,
          errorCount: grammarErrors.length,
          overallScore: overall,
          cefrLevel,
          errorIds: [...errorIds], // Ensure it's a fresh array
        };

        const newAchievements = addDraft(draftData, parentSubmissionId);

        // Show achievement notifications if any were unlocked
        if (newAchievements.length > 0) {
          // Could add toast notifications here in the future
          console.log("New achievements unlocked:", newAchievements);
        }

        // Track fixed errors if we have a previous draft
        if (parentSubmissionId) {
          const previousDrafts = getDraftHistory(parentSubmissionId);
          if (previousDrafts.length > 0) {
            const previousDraft = previousDrafts[previousDrafts.length - 1];
            // Ensure errorIds are arrays (defensive copy)
            const previousErrorIds = Array.isArray(previousDraft.errorIds)
              ? [...previousDraft.errorIds]
              : [];
            const currentErrorIds = [...errorIds];
            trackFixedErrors(submissionId, previousErrorIds, currentErrorIds);
          }
        }
      } catch (error) {
        console.error("Error storing draft:", error);
      }
    }
    // Note: Zustand store functions (addDraft, getDraftHistory, trackFixedErrors) are stable
    // and don't change between renders, so they're safe to exclude from deps
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [
    submissionId,
    overall,
    draftNumber,
    grammarErrors.length,
    finalAnswerText,
    parentSubmissionId,
    addDraft,
    getDraftHistory,
    trackFixedErrors,
  ]);

  // Get stored draft history and merge with metadata, deduplicating by draftNumber
  const storedDraftHistory = submissionId ? getDraftHistory(submissionId) : [];

  // Also try to get draft history using parentSubmissionId if available
  const parentDraftHistory = parentSubmissionId ? getDraftHistory(parentSubmissionId) : [];
  
  // Combine both histories, preferring stored over parent
  const allStoredDrafts = [...storedDraftHistory, ...parentDraftHistory.filter(
    (d) => !storedDraftHistory.some((sd) => sd.submissionId === d.submissionId)
  )];

  // Create a map to deduplicate by draftNumber (prefer stored over metadata)
  const draftMap = new Map<number, (typeof allStoredDrafts)[0]>();

  // First, add all stored drafts (keyed by draftNumber)
  // Prefer drafts with valid submissionIds
  allStoredDrafts.forEach((draft) => {
    if (draft.draftNumber) {
      const existing = draftMap.get(draft.draftNumber);
      // Prefer draft with valid submissionId, or newer timestamp
      if (!existing) {
        draftMap.set(draft.draftNumber, draft);
      } else if (
        (!existing.submissionId && draft.submissionId) ||
        (existing.timestamp < draft.timestamp && draft.submissionId)
      ) {
        draftMap.set(draft.draftNumber, draft);
      }
    }
  });

  // Then add metadata drafts only if not already present (by draftNumber)
  draftHistory.forEach((d, idx) => {
    const draftNum = d.draftNumber || idx + 1;
    if (!draftMap.has(draftNum)) {
      // Use current submissionId only for the current draft, otherwise use a placeholder
      // The current draft should match the current submissionId
      const isCurrentDraft = draftNum === draftNumber;
      draftMap.set(draftNum, {
        draftNumber: draftNum,
        submissionId: isCurrentDraft ? submissionId || "" : "",
        timestamp: d.timestamp,
        wordCount: d.wordCount,
        errorCount: d.errorCount,
        overallScore: d.overallScore,
        cefrLevel: d.overallScore ? mapScoreToCEFR(d.overallScore) : undefined,
        errorIds: [],
      });
    }
  });

  // Sort by draft number and ensure current draft is included
  let displayDraftHistory = Array.from(draftMap.values()).sort(
    (a, b) => a.draftNumber - b.draftNumber
  );

  // Ensure current draft is included if not already present
  const hasCurrentDraft = displayDraftHistory.some((d) => d.draftNumber === draftNumber);
  if (!hasCurrentDraft && submissionId && overall > 0) {
    const wordCount = finalAnswerText.split(/\s+/).filter((w) => w.length > 0).length;
    displayDraftHistory.push({
      draftNumber,
      submissionId,
      timestamp: new Date().toISOString(),
      wordCount,
      errorCount: grammarErrors.length,
      overallScore: overall,
      cefrLevel: mapScoreToCEFR(overall),
      errorIds: [],
    });
    // Re-sort after adding
    displayDraftHistory.sort((a, b) => a.draftNumber - b.draftNumber);
  }
  
  // Final deduplication: remove any duplicates by draftNumber, keeping the one with the best submissionId
  const finalDraftMap = new Map<number, typeof displayDraftHistory[0]>();
  displayDraftHistory.forEach((draft) => {
    const existing = finalDraftMap.get(draft.draftNumber);
    if (!existing || (!existing.submissionId && draft.submissionId)) {
      finalDraftMap.set(draft.draftNumber, draft);
    }
  });
  displayDraftHistory = Array.from(finalDraftMap.values()).sort(
    (a, b) => a.draftNumber - b.draftNumber
  );

  // Handle resubmission of edited essay (with draft tracking)
  const handleResubmit = async (editedText: string, parentSubmissionId?: string) => {
    if (!questionText) {
      throw new Error("Cannot resubmit: Question text not available");
    }
    if (!submissionId) {
      throw new Error("Cannot resubmit: Submission ID not available");
    }

    setIsResubmitting(true);
    try {
      // Use current submission ID as parent if not provided
      const parentId = parentSubmissionId || submissionId;

      // Check if we're in local mode (not saving to server)
      const storeResults = usePreferencesStore.getState().storeResults;

      // Get parent results from localStorage if in local mode
      let parentResults: any = undefined;
      if (!storeResults && typeof window !== "undefined" && parentId) {
        const storedParentResults = localStorage.getItem(`results_${parentId}`);
        if (storedParentResults) {
          try {
            parentResults = JSON.parse(storedParentResults);
          } catch (e) {
            console.warn("[handleResubmit] Failed to parse parent results from localStorage:", e);
          }
        }
      }

      const { submissionId: newSubmissionId, results } = await submitEssay(
        questionText,
        editedText,
        parentId,
        storeResults,
        parentResults
      );
      if (!newSubmissionId || !results) {
        throw new Error("No submission ID or results returned");
      }

      // Store results immediately in sessionStorage and localStorage for immediate access
      // This is critical for server storage mode - results may not be immediately available on server
      if (typeof window !== "undefined") {
        // Store in sessionStorage for immediate display (cleaned up after page load)
        sessionStorage.setItem(`results_${newSubmissionId}`, JSON.stringify(results));
        // Also store in localStorage for persistence
        localStorage.setItem(`results_${newSubmissionId}`, JSON.stringify(results));

        // Store parent relationship for draft tracking (in URL param and localStorage)
        const parentToUse = parentId || submissionId;
        if (parentToUse) {
          localStorage.setItem(`draft_parent_${newSubmissionId}`, parentToUse);
        }
      }

      // Redirect to new results page with parent param
      const parentToUse = parentId || submissionId;
      if (parentToUse) {
        router.push(`/results/${newSubmissionId}?parent=${parentToUse}`);
      } else {
        router.push(`/results/${newSubmissionId}`);
      }
    } catch (error) {
      setIsResubmitting(false);
      throw error;
    }
  };

  return (
    <div className="container" lang="en">
      <div style={{ marginBottom: "var(--spacing-xl)" }} lang="en">
        <h1 className="page-title">Your Writing Feedback</h1>
        <p className="page-subtitle">
          Review your results and see where you can improve. Try another draft to practice more.
        </p>
      </div>

      {/* Celebration Message - Show when learner improves */}
      <CelebrationMessage
        scoreDiff={scoreDiff}
        errorCountDiff={errorCountDiff}
        draftNumber={draftNumber}
      />

      {/* Teacher Feedback */}
      <TeacherFeedback
        overall={overall}
        dimensions={dimensions}
        errorCount={grammarErrors.length}
        aiFeedback={teacherFeedback}
        submissionId={submissionId}
        answerId={answerId}
        answerText={finalAnswerText}
        questionText={questionText}
        ltErrors={ltErrors}
        llmErrors={llmErrors}
        relevanceCheck={(() => {
          const relevanceAssessor = assessorResults.find((a: any) => a.id === "T-RELEVANCE-CHECK");
          if (relevanceAssessor?.meta) {
            return {
              addressesQuestion: Boolean(relevanceAssessor.meta.addressesQuestion ?? false),
              score: Number(relevanceAssessor.meta.similarityScore ?? 0),
              threshold: Number(relevanceAssessor.meta.threshold ?? 0.5),
            };
          }
          return undefined;
        })()}
      />

      {/* Overall Score & Dimensions - Clean, Learner-Focused */}
      <div className="card results-card">
        <div className="overall-score-section">
          <div className="overall-score-main">
            <div className="overall-score-value" style={{ color: getScoreColor(overall) }}>
              {overall.toFixed(1)}
            </div>
            <div className="overall-score-label" lang="en">
              <div className="score-label-main">Your Writing Level</div>
              <div
                className="score-label-sub"
                style={{
                  marginBottom: "var(--spacing-md)",
                  fontSize: "28px",
                  fontWeight: 700,
                  color: "var(--text-primary)",
                  marginTop: "var(--spacing-xs)",
                }}
              >
                {getScoreLabel(overall)}
              </div>

              {/* Dimension Scores - Learner-Friendly Labels */}
              <div
                style={{
                  marginTop: "var(--spacing-md)",
                  padding: "var(--spacing-sm) var(--spacing-md)",
                  backgroundColor: "var(--bg-secondary)",
                  borderRadius: "var(--border-radius)",
                }}
                lang="en"
              >
                <h3
                  style={{
                    fontSize: "16px",
                    fontWeight: 600,
                    marginBottom: "var(--spacing-sm)",
                    color: "var(--text-primary)",
                  }}
                  lang="en"
                >
                  How You Did
                </h3>
                <div className="dimensions-grid-responsive" lang="en">
                  {[
                    {
                      key: "TA",
                      label: "Answering the Question",
                      score: dimensions.TA,
                    },
                    { key: "CC", label: "Organization", score: dimensions.CC },
                    {
                      key: "Vocab",
                      label: "Vocabulary",
                      score: dimensions.Vocab,
                    },
                    {
                      key: "Grammar",
                      label: "Grammar",
                      score: dimensions.Grammar,
                    },
                  ].map(({ key, label, score }) => {
                    const isWeakest = lowestDim && lowestDim[0] === key;
                    const scoreLabel = getScoreLabel(score);
                    const scoreColor = getScoreColor(score);

                    return (
                      <div
                        key={key}
                        style={{
                          padding: "10px 8px",
                          backgroundColor: isWeakest
                            ? "rgba(239, 68, 68, 0.08)"
                            : "var(--bg-primary)",
                          border: isWeakest
                            ? "2px solid rgba(239, 68, 68, 0.4)"
                            : "1px solid rgba(0, 0, 0, 0.08)",
                          borderRadius: "var(--border-radius)",
                          transition: "all 0.2s ease",
                          position: "relative",
                        }}
                        lang="en"
                      >
                        <div
                          style={{
                            fontSize: "28px",
                            fontWeight: 700,
                            color: scoreColor,
                            marginBottom: "4px",
                            lineHeight: "1",
                          }}
                          lang="en"
                        >
                          {score.toFixed(1)}
                        </div>
                        <div
                          style={{
                            fontSize: "10px",
                            fontWeight: 600,
                            color: scoreColor,
                            marginBottom: "4px",
                            textTransform: "uppercase",
                            letterSpacing: "0.5px",
                          }}
                          lang="en"
                        >
                          {scoreLabel}
                        </div>
                        <div
                          style={{
                            fontSize: "12px",
                            color: "var(--text-primary)",
                            lineHeight: "1.3",
                            fontWeight: 500,
                            marginBottom: "6px",
                          }}
                          lang="en"
                        >
                          {label}
                        </div>
                        {/* Visual score indicator bar */}
                        <div
                          style={{
                            height: "3px",
                            backgroundColor: "rgba(0, 0, 0, 0.1)",
                            borderRadius: "2px",
                            overflow: "hidden",
                          }}
                          lang="en"
                        >
                          <div
                            style={{
                              height: "100%",
                              width: `${(score / 9) * 100}%`,
                              backgroundColor: scoreColor,
                              borderRadius: "2px",
                              transition: "width 0.3s ease",
                            }}
                          />
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>

              {/* Simple Level Indicator */}
              <div
                style={{
                  marginTop: "var(--spacing-md)",
                  padding: "var(--spacing-sm) var(--spacing-md)",
                  backgroundColor: "rgba(59, 130, 246, 0.1)",
                  borderRadius: "var(--border-radius)",
                  display: "flex",
                  alignItems: "center",
                  gap: "var(--spacing-sm)",
                  flexWrap: "wrap",
                }}
                lang="en"
              >
                <CEFRBadge level={mapScoreToCEFR(overall)} showLabel={true} />
                <span
                  style={{
                    fontSize: "14px",
                    color: "var(--text-secondary)",
                    flex: "1 1 auto",
                    minWidth: "200px",
                  }}
                  lang="en"
                >
                  {getCEFRDescriptor(mapScoreToCEFR(overall))}
                </span>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Pattern Insights - Simplified */}
      {topErrorTypes && topErrorTypes.length > 0 && (
        <div className="card" lang="en">
          <h2
            style={{
              fontSize: "20px",
              marginBottom: "var(--spacing-md)",
              fontWeight: 600,
            }}
            lang="en"
          >
            Common Areas to Improve
          </h2>
          <p
            style={{
              marginBottom: "var(--spacing-md)",
              fontSize: "16px",
              color: "var(--text-secondary)",
            }}
            lang="en"
          >
            You made these types of errors most often:
          </p>
          <ul style={{ margin: 0, paddingLeft: "var(--spacing-md)" }} lang="en">
            {getTopErrorTypes(grammarErrors, 3).map(({ type, count }) => {
              // Find example errors of this type for explanation
              const exampleErrors = grammarErrors
                .filter((err) => (err.errorType || err.category) === type)
                .slice(0, 2);

              return (
                <ErrorTypeItem key={type} type={type} count={count} examples={exampleErrors} />
              );
            })}
          </ul>
        </div>
      )}

      {/* Grammar & Language Feedback with Heat Map */}
      {grammarErrors && grammarErrors.length > 0 && finalAnswerText && (
        <div className="card notranslate" translate="no" lang="en">
          <h2
            style={{
              fontSize: "20px",
              marginBottom: "var(--spacing-md)",
              fontWeight: 600,
            }}
            lang="en"
          >
            Your Writing with Feedback
          </h2>
          {isFeedbackRevealed && (
            <p
              style={{
                marginBottom: "var(--spacing-md)",
                fontSize: "14px",
                color: "var(--text-secondary)",
              }}
              lang="en"
            >
              Click on highlighted text to see suggestions for improvement.
            </p>
          )}
          <HeatMapText
            text={finalAnswerText}
            errors={grammarErrors}
            onReveal={() => setIsFeedbackRevealed(true)}
          />
        </div>
      )}

      {/* Editable Essay Component */}
      {finalAnswerText && (
        <div>
          {questionText ? (
            <EditableEssay
              initialText={finalAnswerText}
              questionId={answerId}
              questionText={questionText}
              parentSubmissionId={submissionId}
              onSubmit={handleResubmit}
            />
          ) : (
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
              <p
                lang="en"
                style={{
                  fontSize: "14px",
                  color: "var(--text-secondary)",
                  fontStyle: "italic",
                }}
              >
                Note: Question text is not available for editing. You can still view your essay and
                feedback above.
              </p>
            </div>
          )}
        </div>
      )}

      {/* Draft History & Comparison */}
      {displayDraftHistory.length > 1 && (
        <div className="card" lang="en" style={{ padding: "var(--spacing-md)" }}>
          <>
            <h2
              style={{
                fontSize: "16px",
                marginBottom: "var(--spacing-sm)",
                fontWeight: 600,
              }}
              lang="en"
            >
              Draft History
            </h2>
            <div
              style={{
                display: "flex",
                flexWrap: "wrap",
                gap: "var(--spacing-sm)",
                marginBottom: "var(--spacing-md)",
              }}
              lang="en"
            >
              {displayDraftHistory.map((draft) => {
                // Find the root parent submission ID (draft 1's submissionId)
                // This should be the parentSubmissionId when viewing draft 2+
                const rootDraft = displayDraftHistory.find((d) => d.draftNumber === 1);
                
                // For Draft 1, use parentSubmissionId if available (it's the root), otherwise try to find it
                // For Draft 2+, the parentSubmissionId IS Draft 1's submissionId
                let draftSubmissionId = draft.submissionId;
                
                // If this is Draft 1 and we don't have a submissionId, try parentSubmissionId
                // (which would be the case if we're viewing from Draft 2's perspective)
                if (draft.draftNumber === 1 && (!draftSubmissionId || draftSubmissionId.length === 0)) {
                  // When viewing Draft 2, parentSubmissionId IS Draft 1's submissionId
                  if (parentSubmissionId && draftNumber > 1) {
                    draftSubmissionId = parentSubmissionId;
                  } else {
                    // Try to find it from stored history
                    const storedHistory = submissionId ? getDraftHistory(submissionId) : [];
                    const storedDraft = storedHistory.find((d) => d.draftNumber === 1);
                    draftSubmissionId = storedDraft?.submissionId || "";
                  }
                }
                
                // For other drafts, try to find submissionId from store if missing
                if ((!draftSubmissionId || draftSubmissionId.length === 0) && draft.draftNumber !== 1) {
                  const storedHistory = submissionId ? getDraftHistory(submissionId) : [];
                  const storedDraft = storedHistory.find((d) => d.draftNumber === draft.draftNumber);
                  draftSubmissionId = storedDraft?.submissionId || "";
                }
                
                const hasValidSubmissionId = draftSubmissionId && draftSubmissionId.length > 0;
                const isFirstDraft = draft.draftNumber === 1;
                const rootSubmissionId = rootDraft?.submissionId || parentSubmissionId || draftSubmissionId;
                
                // Build navigation URL
                // For Draft 1, just go to its results page
                // For Draft 2+, include parent param pointing to Draft 1
                const navigateUrl = hasValidSubmissionId
                  ? isFirstDraft
                    ? `/results/${draftSubmissionId}`
                    : rootSubmissionId && rootSubmissionId !== draftSubmissionId
                      ? `/results/${draftSubmissionId}?parent=${rootSubmissionId}`
                      : `/results/${draftSubmissionId}`
                  : "#";

                return (
                  <div
                    key={`draft-${draft.draftNumber}-${draftSubmissionId || draft.timestamp}`}
                    style={{
                      padding: "var(--spacing-sm) var(--spacing-md)",
                      backgroundColor:
                        draft.draftNumber === draftNumber
                          ? "var(--primary-color)"
                          : "var(--bg-secondary)",
                      color: draft.draftNumber === draftNumber ? "white" : "var(--text-primary)",
                      borderRadius: "var(--border-radius)",
                      fontSize: "14px",
                      fontWeight: draft.draftNumber === draftNumber ? 600 : 500,
                      cursor: hasValidSubmissionId && draft.draftNumber !== draftNumber ? "pointer" : "default",
                      transition: "all 0.2s ease",
                      opacity: hasValidSubmissionId ? 1 : 0.6,
                      textAlign: "center",
                      minWidth: "100px",
                      border:
                        draft.draftNumber === draftNumber
                          ? "2px solid var(--primary-color)"
                          : "1px solid var(--border-color)",
                    }}
                    onClick={() => {
                      // Don't navigate if this is the current draft (would just reload)
                      if (hasValidSubmissionId && draft.draftNumber !== draftNumber) {
                        router.push(navigateUrl);
                      }
                    }}
                    onMouseEnter={(e) => {
                      if (draft.draftNumber !== draftNumber && hasValidSubmissionId) {
                        e.currentTarget.style.backgroundColor = "var(--bg-primary)";
                        e.currentTarget.style.transform = "scale(1.05)";
                        e.currentTarget.style.boxShadow = "var(--shadow-sm)";
                      }
                    }}
                    onMouseLeave={(e) => {
                      if (draft.draftNumber !== draftNumber) {
                        e.currentTarget.style.backgroundColor = "var(--bg-secondary)";
                        e.currentTarget.style.transform = "scale(1)";
                        e.currentTarget.style.boxShadow = "none";
                      }
                    }}
                    lang="en"
                  >
                    <div style={{ fontWeight: 600, marginBottom: "2px" }}>
                      Draft {draft.draftNumber}
                    </div>
                    {draft.overallScore && (
                      <div style={{ fontSize: "12px", opacity: 0.9 }}>
                        {draft.overallScore.toFixed(1)}
                      </div>
                    )}
                  </div>
                );
              })}
            </div>
            {displayDraftHistory.length > 1 && (
              <ProgressChart draftHistory={displayDraftHistory} type="score" />
            )}
          </>

          {/* Draft Comparison Table */}
          {displayDraftHistory.length > 1 && (
            <div
              style={{
                marginTop: "var(--spacing-md)",
                padding: "var(--spacing-md)",
                backgroundColor: "var(--bg-primary)",
                borderRadius: "var(--border-radius)",
                border: "1px solid var(--border-color)",
              }}
              lang="en"
            >
              <h3
                style={{
                  fontSize: "16px",
                  fontWeight: 600,
                  marginBottom: "var(--spacing-md)",
                  color: "var(--text-primary)",
                }}
                lang="en"
              >
                Draft Comparison
              </h3>
              <div
                style={{
                  overflowX: "auto",
                }}
              >
                <table
                  style={{
                    width: "100%",
                    borderCollapse: "collapse",
                    fontSize: "14px",
                  }}
                  lang="en"
                >
                  <thead>
                    <tr
                      style={{
                        borderBottom: "2px solid var(--border-color)",
                      }}
                    >
                      <th
                        style={{
                          textAlign: "left",
                          padding: "var(--spacing-sm)",
                          fontWeight: 600,
                          color: "var(--text-primary)",
                        }}
                        lang="en"
                      >
                        Draft
                      </th>
                      <th
                        style={{
                          textAlign: "right",
                          padding: "var(--spacing-sm)",
                          fontWeight: 600,
                          color: "var(--text-primary)",
                        }}
                        lang="en"
                      >
                        Score
                      </th>
                      <th
                        style={{
                          textAlign: "right",
                          padding: "var(--spacing-sm)",
                          fontWeight: 600,
                          color: "var(--text-primary)",
                        }}
                        lang="en"
                      >
                        Words
                      </th>
                      <th
                        style={{
                          textAlign: "right",
                          padding: "var(--spacing-sm)",
                          fontWeight: 600,
                          color: "var(--text-primary)",
                        }}
                        lang="en"
                      >
                        Errors
                      </th>
                      <th
                        style={{
                          textAlign: "left",
                          padding: "var(--spacing-sm)",
                          fontWeight: 600,
                          color: "var(--text-primary)",
                        }}
                        lang="en"
                      >
                        Change
                      </th>
                    </tr>
                  </thead>
                  <tbody>
                    {displayDraftHistory.map((draft, index) => {
                      const prevDraft = index > 0 ? displayDraftHistory[index - 1] : null;
                      const scoreChange =
                        prevDraft && draft.overallScore && prevDraft.overallScore
                          ? draft.overallScore - prevDraft.overallScore
                          : null;
                      const wordChange =
                        prevDraft && draft.wordCount && prevDraft.wordCount
                          ? draft.wordCount - prevDraft.wordCount
                          : null;
                      const errorChange =
                        prevDraft && draft.errorCount !== undefined && prevDraft.errorCount !== undefined
                          ? draft.errorCount - prevDraft.errorCount
                          : null;
                      const isCurrent = draft.draftNumber === draftNumber;

                      return (
                        <tr
                          key={`draft-row-${draft.draftNumber}-${draft.submissionId || draft.timestamp}`}
                          style={{
                            borderBottom: "1px solid var(--border-color)",
                            backgroundColor: isCurrent ? "rgba(59, 130, 246, 0.05)" : "transparent",
                          }}
                        >
                          <td
                            style={{
                              padding: "var(--spacing-sm)",
                              fontWeight: isCurrent ? 600 : 500,
                              color: isCurrent ? "var(--primary-color)" : "var(--text-primary)",
                            }}
                            lang="en"
                          >
                            Draft {draft.draftNumber}
                            {isCurrent && (
                              <span
                                style={{
                                  marginLeft: "var(--spacing-xs)",
                                  fontSize: "12px",
                                  color: "var(--text-secondary)",
                                }}
                                lang="en"
                              >
                                (current)
                              </span>
                            )}
                          </td>
                          <td
                            style={{
                              textAlign: "right",
                              padding: "var(--spacing-sm)",
                              fontWeight: 600,
                              color: draft.overallScore
                                ? getScoreColor(draft.overallScore)
                                : "var(--text-secondary)",
                            }}
                            lang="en"
                          >
                            {draft.overallScore ? draft.overallScore.toFixed(1) : "-"}
                          </td>
                          <td
                            style={{
                              textAlign: "right",
                              padding: "var(--spacing-sm)",
                              color: "var(--text-primary)",
                            }}
                            lang="en"
                          >
                            {draft.wordCount || "-"}
                          </td>
                          <td
                            style={{
                              textAlign: "right",
                              padding: "var(--spacing-sm)",
                              color: "var(--text-primary)",
                            }}
                            lang="en"
                          >
                            {draft.errorCount !== undefined ? draft.errorCount : "-"}
                          </td>
                          <td
                            style={{
                              padding: "var(--spacing-sm)",
                              color: "var(--text-secondary)",
                              fontSize: "12px",
                            }}
                            lang="en"
                          >
                            {index === 0 ? (
                              <span style={{ fontStyle: "italic" }}>Baseline</span>
                            ) : (
                              <div style={{ display: "flex", flexDirection: "column", gap: "2px" }}>
                                {scoreChange !== null && scoreChange !== 0 && (
                                  <span
                                    style={{
                                      color:
                                        scoreChange > 0
                                          ? "var(--secondary-accent)"
                                          : "var(--error-color)",
                                    }}
                                  >
                                    {scoreChange > 0 ? "+" : ""}
                                    {scoreChange.toFixed(1)} score
                                  </span>
                                )}
                                {wordChange !== null && wordChange !== 0 && (
                                  <span>
                                    {wordChange > 0 ? "+" : ""}
                                    {wordChange} words
                                  </span>
                                )}
                                {errorChange !== null && errorChange !== 0 && (
                                  <span
                                    style={{
                                      color:
                                        errorChange < 0
                                          ? "var(--secondary-accent)"
                                          : errorChange > 0
                                            ? "var(--error-color)"
                                            : "inherit",
                                    }}
                                  >
                                    {errorChange > 0 ? "+" : ""}
                                    {errorChange} errors
                                  </span>
                                )}
                                {scoreChange === 0 &&
                                  wordChange === 0 &&
                                  errorChange === 0 && <span>No change</span>}
                              </div>
                            )}
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            </div>
          )}
        </div>
      )}

      {/* Action Buttons */}
      <div
        style={{
          marginTop: "var(--spacing-lg)",
          display: "flex",
          gap: "var(--spacing-md)",
          flexWrap: "wrap",
        }}
      >
        <Link href="/" className="btn btn-primary" lang="en">
          ← Back to Tasks
        </Link>
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
      </div>
    </div>
  );
}
