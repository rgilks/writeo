"use server";

import type { AssessmentResults } from "@writeo/shared";
import {
  getEssayAssessorResult,
  getLanguageToolAssessorResult,
  getLLMAssessorResult,
} from "@writeo/shared";
import { getSubmissionResults } from "./submission";

interface DraftHistoryEntry {
  draftNumber: number;
  timestamp: string;
  wordCount: number;
  errorCount: number;
  overallScore?: number;
}

interface DraftInfo {
  draftNumber: number;
  parentSubmissionId: string;
  draftHistory: DraftHistoryEntry[];
}

function getFirstAssessorResults(results: AssessmentResults) {
  const parts = results.results?.parts;
  if (!parts || parts.length === 0) {
    return [];
  }

  const firstPart = parts[0];
  const firstAnswer = firstPart?.answers?.[0];
  return firstAnswer?.assessorResults || [];
}

function extractOverallScore(results: AssessmentResults): number | undefined {
  const metaScore = results.meta?.overallScore;
  if (typeof metaScore === "number") {
    return metaScore;
  }

  const assessorResults = getFirstAssessorResults(results);
  const essayAssessor = getEssayAssessorResult(assessorResults);
  return essayAssessor?.overall;
}

function extractErrorCount(results: AssessmentResults): number {
  const metaErrorCount = results.meta?.errorCount;
  if (typeof metaErrorCount === "number" && metaErrorCount > 0) {
    return metaErrorCount;
  }

  const assessorResults = getFirstAssessorResults(results);
  const ltAssessor = getLanguageToolAssessorResult(assessorResults);
  const llmAssessor = getLLMAssessorResult(assessorResults);

  const ltErrorCount = ltAssessor?.errors?.length || 0;
  const llmErrorCount = llmAssessor?.errors?.length || 0;
  return ltErrorCount + llmErrorCount;
}

export async function getDraftInfo(
  parentSubmissionId: string,
  storeResults: boolean = false,
  parentResults?: AssessmentResults,
): Promise<DraftInfo> {
  try {
    let parentResultsData: AssessmentResults | undefined = parentResults;

    if (!parentResultsData) {
      if (storeResults) {
        const fetched = await getSubmissionResults(parentSubmissionId);
        if (
          typeof fetched === "object" &&
          fetched !== null &&
          "status" in fetched &&
          "template" in fetched
        ) {
          parentResultsData = fetched as AssessmentResults;
        }
      }
      if (!parentResultsData) {
        return {
          draftNumber: 2,
          parentSubmissionId,
          draftHistory: [],
        };
      }
    }

    const parentDraftNumber =
      (typeof parentResultsData.meta?.draftNumber === "number"
        ? parentResultsData.meta.draftNumber
        : 1) || 1;

    const rootSubmissionId =
      (typeof parentResultsData.meta?.parentSubmissionId === "string"
        ? parentResultsData.meta.parentSubmissionId
        : parentSubmissionId) || parentSubmissionId;

    const parentHistory = (
      Array.isArray(parentResultsData.meta?.draftHistory) ? parentResultsData.meta.draftHistory : []
    ) as DraftHistoryEntry[];

    const overallScore = extractOverallScore(parentResultsData);
    const errorCount = extractErrorCount(parentResultsData);

    const parentAlreadyInHistory = parentHistory.some(
      (entry) => entry.draftNumber === parentDraftNumber,
    );

    const draftHistory: DraftHistoryEntry[] = parentAlreadyInHistory
      ? [...parentHistory]
      : [
          ...parentHistory,
          {
            draftNumber: parentDraftNumber,
            timestamp:
              (typeof parentResultsData.meta?.timestamp === "string"
                ? parentResultsData.meta.timestamp
                : new Date().toISOString()) || new Date().toISOString(),
            wordCount:
              (typeof parentResultsData.meta?.wordCount === "number"
                ? parentResultsData.meta.wordCount
                : 0) || 0,
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
  parentResults?: AssessmentResults,
): Promise<AssessmentResults> {
  const fetched = await getSubmissionResults(submissionId);

  if (
    typeof fetched !== "object" ||
    fetched === null ||
    !("status" in fetched) ||
    !("template" in fetched)
  ) {
    throw new Error("Invalid submission results format");
  }

  const results = fetched as AssessmentResults;

  if (results.status !== "success") {
    return results;
  }

  if (!results.meta) {
    results.meta = {};
  }

  if (parentSubmissionId) {
    try {
      const draftInfo = await getDraftInfo(parentSubmissionId, storeResults, parentResults);

      results.meta.draftNumber = draftInfo.draftNumber;
      results.meta.parentSubmissionId = draftInfo.parentSubmissionId;
      results.meta.draftHistory = draftInfo.draftHistory;

      const wordCount = results.meta.wordCount;
      const errorCount = results.meta.errorCount;
      if (typeof wordCount === "number" && typeof errorCount === "number") {
        const currentDraft: DraftHistoryEntry = {
          draftNumber: draftInfo.draftNumber,
          timestamp:
            (typeof results.meta.timestamp === "string"
              ? results.meta.timestamp
              : new Date().toISOString()) || new Date().toISOString(),
          wordCount,
          errorCount,
          overallScore:
            typeof results.meta.overallScore === "number" ? results.meta.overallScore : undefined,
        };
        results.meta.draftHistory = [...draftInfo.draftHistory, currentDraft];
      }
    } catch (error) {
      console.warn("Failed to get draft info:", error);
    }
  } else {
    results.meta.draftNumber = 1;
    results.meta.draftHistory = [
      {
        draftNumber: 1,
        timestamp:
          (typeof results.meta.timestamp === "string"
            ? results.meta.timestamp
            : new Date().toISOString()) || new Date().toISOString(),
        wordCount: (typeof results.meta.wordCount === "number" ? results.meta.wordCount : 0) || 0,
        errorCount:
          (typeof results.meta.errorCount === "number" ? results.meta.errorCount : 0) || 0,
        overallScore:
          typeof results.meta.overallScore === "number" ? results.meta.overallScore : undefined,
      },
    ];
  }

  return results;
}
