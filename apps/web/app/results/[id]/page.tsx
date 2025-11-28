"use client";

import { useEffect, useState, useCallback } from "react";
import { useParams, useRouter } from "next/navigation";
import Link from "next/link";
import { getSubmissionResults } from "@/app/lib/actions";
import { usePreferencesStore } from "@/app/lib/stores/preferences-store";
import { useDraftStore } from "@/app/lib/stores/draft-store";
import { LearnerResultsView } from "@/app/components/LearnerResultsView";
import { DeveloperResultsView } from "@/app/components/DeveloperResultsView";
import { ModeSwitcher } from "@/app/components/ModeSwitcher";
import { ErrorBoundary } from "@/app/components/ErrorBoundary";
import type { AssessmentResults } from "@writeo/shared";

import { getErrorMessage, DEFAULT_ERROR_MESSAGES } from "@/app/lib/utils/error-messages";

function getFriendlyErrorMessage(error: unknown): string {
  if (error instanceof Error) {
    return getErrorMessage(error, "results");
  }
  return DEFAULT_ERROR_MESSAGES.results;
}

function extractAnswerText(data: AssessmentResults): string {
  const answerTexts = data.meta?.answerTexts as Record<string, string> | undefined;
  if (answerTexts) {
    const firstAnswerId = Object.keys(answerTexts)[0];
    return answerTexts[firstAnswerId] || "";
  }
  return "";
}

export default function ResultsPage() {
  const params = useParams();
  const router = useRouter();
  const submissionId = params.id as string;
  const mode = usePreferencesStore((state) => state.viewMode);
  const getResult = useDraftStore((state) => state.getResult);
  const setResult = useDraftStore((state) => state.setResult);

  const [data, setData] = useState<AssessmentResults | null>(() => {
    // Try to get from store on initial render (Zustand handles hydration automatically)
    if (typeof window !== "undefined") {
      return getResult(submissionId) || null;
    }
    return null;
  });
  const [status, setStatus] = useState<"pending" | "success" | "error">(
    data ? "success" : "pending",
  );
  const [error, setError] = useState<string | null>(null);
  const [answerText, setAnswerText] = useState<string>("");

  // Fetch results if we don't have them
  useEffect(() => {
    // If we already have data, don't fetch
    if (data) {
      return;
    }

    let cancelled = false;

    async function fetchResults() {
      try {
        setStatus("pending");

        // Check store first (Zustand persist handles hydration automatically)
        const storedResults = getResult(submissionId);
        if (storedResults) {
          if (!cancelled) {
            setData(storedResults);
            setStatus("success");
          }
          return;
        }

        // Check if we're in local mode
        const storeResults = usePreferencesStore.getState().storeResults;

        // In local-only mode, results should be in store
        if (!storeResults) {
          throw new Error(
            "Results not found in local storage. In local-only mode, results are only stored in your browser.",
          );
        }

        // Server storage mode - fetch from server
        const fetched = await getSubmissionResults(submissionId);
        if (!cancelled) {
          if (
            typeof fetched === "object" &&
            fetched !== null &&
            "status" in fetched &&
            "template" in fetched
          ) {
            const results = fetched as AssessmentResults;
            setData(results);
            setStatus("success");
            setResult(submissionId, results);
          } else {
            throw new Error("Invalid results format");
          }
        }
      } catch (err) {
        if (!cancelled) {
          setError(getFriendlyErrorMessage(err));
          setStatus("error");
        }
      }
    }

    if (submissionId) {
      fetchResults();
    }

    return () => {
      cancelled = true;
    };
    // Only run when submissionId changes, not when data changes
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [submissionId]);

  // Extract answer text when results arrive
  useEffect(() => {
    if (data && status === "success") {
      setAnswerText(extractAnswerText(data));
    }
  }, [data, status]);

  // Function to switch drafts without page reload
  const switchDraft = useCallback(
    (targetSubmissionId: string) => {
      const storedResults = getResult(targetSubmissionId);
      if (storedResults) {
        setData(storedResults);
        setStatus("success");
        setError(null);
        setAnswerText(extractAnswerText(storedResults));
        router.replace(`/results/${targetSubmissionId}`);
        return true;
      }
      return false;
    },
    [getResult, router],
  );

  return (
    <>
      <header className="header">
        <div className="header-content">
          <div className="logo-group">
            <Link href="/" className="logo">
              Writeo
            </Link>
          </div>
          <nav className="header-actions" aria-label="Results actions">
            <ModeSwitcher />
            {status !== "pending" && (
              <Link href="/" className="nav-back-link">
                <span aria-hidden="true">←</span> Back to Home
              </Link>
            )}
          </nav>
        </div>
      </header>
      <div className="container" style={{ minHeight: "calc(100vh - 200px)" }}>
        {status === "pending" && (
          <div
            style={{
              textAlign: "center",
              padding: "var(--spacing-3xl) var(--spacing-lg)",
              minHeight: "400px",
              display: "flex",
              flexDirection: "column",
              justifyContent: "center",
            }}
          >
            <h2 style={{ marginBottom: "var(--spacing-md)" }}>Loading Results...</h2>
            <p style={{ color: "var(--text-secondary)" }}>Fetching your essay results...</p>
          </div>
        )}

        {status === "error" && (
          <div
            className="card"
            style={{
              maxWidth: "600px",
              margin: "var(--spacing-3xl) auto",
              textAlign: "center",
              minHeight: "400px",
              display: "flex",
              flexDirection: "column",
              justifyContent: "center",
            }}
          >
            <div style={{ fontSize: "48px", marginBottom: "var(--spacing-md)" }}>📝</div>
            <h2 style={{ marginBottom: "var(--spacing-sm)" }}>Results Not Available</h2>
            <p
              style={{
                color: "var(--text-secondary)",
                marginBottom: "var(--spacing-lg)",
                lineHeight: "1.6",
              }}
            >
              {error || DEFAULT_ERROR_MESSAGES.results}
            </p>
            <div style={{ marginTop: "var(--spacing-lg)" }}>
              <Link href="/" className="btn btn-primary">
                Try Another Essay →
              </Link>
            </div>
          </div>
        )}

        {status === "success" && data && (
          <div style={mode === "developer" ? {} : { animation: "fadeIn 0.3s ease-in" }}>
            {mode === "developer" ? (
              <DeveloperResultsView data={data} answerText={answerText} />
            ) : (
              <ErrorBoundary>
                <LearnerResultsView
                  data={data}
                  answerText={answerText}
                  submissionId={submissionId}
                  onDraftSwitch={switchDraft}
                />
              </ErrorBoundary>
            )}
          </div>
        )}
      </div>
    </>
  );
}
