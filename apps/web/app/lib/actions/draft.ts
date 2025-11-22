/**
 * Draft tracking server actions
 */

"use server";

import { getSubmissionResults } from "./submission";

export async function getDraftInfo(
  parentSubmissionId: string,
  storeResults: boolean = false,
  parentResults?: any
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
        throw new Error("Parent results not available in local mode");
      }
    }

    const parentDraftNumber = (parentResultsData.meta?.draftNumber as number) || 1;
    const rootSubmissionId =
      (parentResultsData.meta?.parentSubmissionId as string) || parentSubmissionId;
    const parentHistory = (parentResultsData.meta?.draftHistory as any[]) || [];

    const draftHistory = [
      ...parentHistory,
      {
        draftNumber: parentDraftNumber,
        timestamp: (parentResultsData.meta?.timestamp as string) || new Date().toISOString(),
        wordCount: (parentResultsData.meta?.wordCount as number) || 0,
        errorCount: (parentResultsData.meta?.errorCount as number) || 0,
        overallScore: parentResultsData.meta?.overallScore as number | undefined,
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
  parentResults?: any
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
