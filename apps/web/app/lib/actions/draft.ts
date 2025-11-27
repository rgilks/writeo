/**
 * Draft tracking server actions
 */

"use server";

import { getSubmissionResults } from "./submission";

export async function getDraftInfo(
  parentSubmissionId: string,
  storeResults: boolean = false,
  parentResults?: any,
): Promise<{
  draftNumber: number;
  parentSubmissionId: string;
  draftHistory: Array<{
    draftNumber: number;
    timestamp: string;
    wordCount: number;
    errorCount: number;
    overallScore?: number;
  }>;
}> {
  try {
    let parentResultsData = parentResults;

    if (!parentResultsData) {
      if (storeResults) {
        parentResultsData = await getSubmissionResults(parentSubmissionId);
      } else {
        // In local mode, we should have parentResults passed from the client
        // If not available, use defaults but still maintain the draft chain
        return {
          draftNumber: 2,
          parentSubmissionId: parentSubmissionId, // Use parent as root for draft chain
          draftHistory: [],
        };
      }
    }

    const parentDraftNumber = (parentResultsData.meta?.draftNumber as number) || 1;
    // Use the root submission ID from parent, or the parent's ID if this is draft 2
    const rootSubmissionId =
      (parentResultsData.meta?.parentSubmissionId as string) || parentSubmissionId;
    const parentHistory = (parentResultsData.meta?.draftHistory as any[]) || [];

    // Extract overall score from the assessment results if not in meta
    let overallScore = parentResultsData.meta?.overallScore as number | undefined;
    if (overallScore === undefined) {
      // Try to get score from assessor results
      try {
        const parts = parentResultsData.results?.parts || [];
        const firstPart = parts[0];
        const firstAnswer = firstPart?.answers?.[0];
        const assessorResults = firstAnswer?.["assessor-results"] || [];
        const essayAssessor = assessorResults.find(
          (r: any) => r.assessor === "essay-assessor" || r.assessor === "modal-essay-assessor",
        );
        overallScore = essayAssessor?.result?.overall;
      } catch {
        // Ignore errors
      }
    }

    // Extract error count from assessor results if not in meta
    let errorCount = (parentResultsData.meta?.errorCount as number) || 0;
    if (errorCount === 0) {
      try {
        const parts = parentResultsData.results?.parts || [];
        const firstPart = parts[0];
        const firstAnswer = firstPart?.answers?.[0];
        const assessorResults = firstAnswer?.["assessor-results"] || [];
        const ltAssessor = assessorResults.find(
          (r: any) => r.assessor === "languagetool-assessor" || r.assessor === "modal-lt-assessor",
        );
        const llmAssessor = assessorResults.find((r: any) => r.assessor === "llm-error-assessor");
        errorCount =
          (ltAssessor?.result?.errors?.length || 0) + (llmAssessor?.result?.errors?.length || 0);
      } catch {
        // Ignore errors
      }
    }

    // Check if the parent's entry already exists in the history
    const parentAlreadyInHistory = parentHistory.some(
      (h: any) => h.draftNumber === parentDraftNumber,
    );

    const draftHistory = parentAlreadyInHistory
      ? [...parentHistory]
      : [
          ...parentHistory,
          {
            draftNumber: parentDraftNumber,
            timestamp: (parentResultsData.meta?.timestamp as string) || new Date().toISOString(),
            wordCount: (parentResultsData.meta?.wordCount as number) || 0,
            errorCount,
            overallScore,
          },
        ];

    return {
      draftNumber: parentDraftNumber + 1,
      parentSubmissionId: rootSubmissionId,
      draftHistory,
    };
  } catch {
    return {
      draftNumber: 2,
      parentSubmissionId,
      draftHistory: [],
    };
  }
}

export async function getSubmissionResultsWithDraftTracking(
  submissionId: string,
  parentSubmissionId?: string,
  storeResults: boolean = false,
  parentResults?: any,
): Promise<any> {
  const results = await getSubmissionResults(submissionId);

  if (parentSubmissionId && results.status === "success") {
    try {
      const draftInfo = await getDraftInfo(parentSubmissionId, storeResults, parentResults);

      if (!results.meta) {
        results.meta = {};
      }
      results.meta.draftNumber = draftInfo.draftNumber;
      results.meta.parentSubmissionId = draftInfo.parentSubmissionId;
      results.meta.draftHistory = draftInfo.draftHistory;

      if (results.meta.wordCount !== undefined && results.meta.errorCount !== undefined) {
        const currentDraft = {
          draftNumber: draftInfo.draftNumber,
          timestamp: results.meta.timestamp || new Date().toISOString(),
          wordCount: results.meta.wordCount as number,
          errorCount: results.meta.errorCount as number,
          overallScore: results.meta.overallScore as number | undefined,
        };
        results.meta.draftHistory = [...draftInfo.draftHistory, currentDraft];
      }
    } catch (error) {
      console.warn("Failed to get draft info:", error);
    }
  } else if (results.status === "success" && results.meta) {
    results.meta.draftNumber = 1;
    results.meta.draftHistory = [
      {
        draftNumber: 1,
        timestamp: results.meta.timestamp || new Date().toISOString(),
        wordCount: results.meta.wordCount || 0,
        errorCount: results.meta.errorCount || 0,
        overallScore: results.meta.overallScore,
      },
    ];
  }

  return results;
}
