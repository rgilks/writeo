"use client";

import { useState, useMemo, useCallback } from "react";
import { TeacherFeedback } from "./TeacherFeedback";
import { CelebrationMessage } from "./CelebrationMessage";
import type { AssessmentResults } from "@writeo/shared";
import { ScoreDisplay } from "./learner-results/ScoreDisplay";
import { DimensionScores } from "./learner-results/DimensionScores";
import { ErrorTypeList } from "./learner-results/ErrorTypeList";
import { DraftHistorySection } from "./learner-results/DraftHistorySection";
import { HeatMapSection } from "./learner-results/HeatMapSection";
import { EditableEssaySection } from "./learner-results/EditableEssaySection";
import { FooterSection } from "./learner-results/FooterSection";
import { useDataExtraction } from "./learner-results/useDataExtraction";
import { useDraftHistory } from "./learner-results/useDraftHistory";
import { useDraftStorage } from "./learner-results/useDraftStorage";
import { useResubmit } from "./learner-results/useResubmit";
import { useDraftStore } from "@/app/lib/stores/draft-store";

interface LearnerResultsViewProps {
  data: AssessmentResults;
  answerText: string;
  submissionId?: string;
  onDraftSwitch?: (submissionId: string, parentId?: string) => boolean;
}

export function LearnerResultsView({
  data,
  answerText,
  submissionId: propSubmissionId,
  onDraftSwitch,
}: LearnerResultsViewProps) {
  const [isFeedbackRevealed, setIsFeedbackRevealed] = useState(false);
  const getDraftHistory = useDraftStore((state) => state.getDraftHistory);

  const {
    overall,
    dimensions,
    lowestDim,
    grammarErrors,
    ltErrors,
    llmErrors,
    teacherFeedback,
    submissionId,
    draftNumber,
    parentSubmissionId,
    scoreDiff,
    errorCountDiff,
    answerId,
    questionText,
    relevanceCheck,
  } = useDataExtraction(data, propSubmissionId);

  const finalAnswerText = useMemo(() => {
    if (answerText) return answerText;
    const answerTexts = data.meta?.answerTexts as Record<string, string> | undefined;
    return answerTexts && answerId ? answerTexts[answerId] : "";
  }, [answerText, data.meta, answerId]);

  useDraftStorage(
    submissionId,
    overall,
    draftNumber,
    grammarErrors,
    finalAnswerText,
    parentSubmissionId,
  );

  const { displayDraftHistory } = useDraftHistory(
    data,
    submissionId,
    overall,
    grammarErrors,
    finalAnswerText,
    parentSubmissionId,
  );

  const { handleResubmit } = useResubmit();

  const handleResubmitWrapper = useCallback(
    async (editedText: string) => {
      if (!submissionId) {
        throw new Error("Cannot resubmit: Missing submission ID");
      }
      await handleResubmit(editedText, questionText || "", submissionId);
    },
    [submissionId, questionText, handleResubmit],
  );

  const handleReveal = useCallback(() => {
    setIsFeedbackRevealed(true);
  }, []);

  return (
    <div className="container" lang="en">
      <div style={{ marginBottom: "var(--spacing-xl)" }}>
        <h1 className="page-title">Your Writing Feedback</h1>
        <p className="page-subtitle">
          Review your results and see where you can improve. Try another draft to practice more.
        </p>
      </div>

      <CelebrationMessage
        scoreDiff={scoreDiff}
        errorCountDiff={errorCountDiff}
        draftNumber={draftNumber}
      />

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
        relevanceCheck={relevanceCheck}
      />

      <div className="card results-card">
        <ScoreDisplay overall={overall} />
        <DimensionScores
          dimensions={dimensions}
          lowestDim={lowestDim}
          questionText={questionText}
        />
      </div>

      <ErrorTypeList grammarErrors={grammarErrors} />

      <HeatMapSection
        grammarErrors={grammarErrors}
        finalAnswerText={finalAnswerText}
        isFeedbackRevealed={isFeedbackRevealed}
        onReveal={handleReveal}
      />

      <EditableEssaySection
        finalAnswerText={finalAnswerText}
        questionText={questionText}
        answerId={answerId}
        submissionId={submissionId}
        onSubmit={handleResubmitWrapper}
      />

      <DraftHistorySection
        displayDraftHistory={displayDraftHistory}
        draftNumber={draftNumber}
        submissionId={submissionId}
        parentSubmissionId={parentSubmissionId}
        getDraftHistory={getDraftHistory}
        onDraftSwitch={onDraftSwitch}
      />

      <FooterSection />
    </div>
  );
}
