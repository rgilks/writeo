"use client";

import { useEffect, useState } from "react";
import { useParams, useRouter } from "next/navigation";
import Link from "next/link";
import { getSubmissionResults, getSubmissionResultsWithDraftTracking } from "@/app/lib/actions";
import { usePreferencesStore } from "@/app/lib/stores/preferences-store";
import { useDraftStore } from "@/app/lib/stores/draft-store";
import { LearnerResultsView } from "@/app/components/LearnerResultsView";
import { DeveloperResultsView } from "@/app/components/DeveloperResultsView";
import { ModeSwitcher } from "@/app/components/ModeSwitcher";
import { ErrorBoundary } from "@/app/components/ErrorBoundary";
import type { AssessmentResults } from "@writeo/shared";

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
  const [processingTime, setProcessingTime] = useState<number | null>(null);
  const [submissionStartTime] = useState<number>(Date.now());

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
        const results = await getSubmissionResults(submissionId);
        if (!cancelled) {
          setData(results);
          setStatus("success");
          setResult(submissionId, results);
        }
      } catch (err) {
        if (!cancelled) {
          const errorMessage = err instanceof Error ? err.message : "Failed to fetch results";
          const friendlyMessage =
            errorMessage.includes("Server Component") ||
            errorMessage.includes("omitted in production") ||
            errorMessage.includes("not found")
              ? "We couldn't load your results. This might happen if the submission ID is incorrect or the results are no longer available."
              : errorMessage;
          setError(friendlyMessage);
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

  // Extract answer text and calculate processing time when results arrive
  useEffect(() => {
    if (data && status === "success") {
      const answerTexts = data.meta?.answerTexts as Record<string, string> | undefined;
      if (answerTexts) {
        const firstAnswerId = Object.keys(answerTexts)[0];
        setAnswerText(answerTexts[firstAnswerId] || "");
      }

      const endTime = Date.now();
      const elapsed = (endTime - submissionStartTime) / 1000;
      setProcessingTime(elapsed);
    }
  }, [data, status, submissionStartTime]);

  // Function to switch drafts without page reload
  const switchDraft = (targetSubmissionId: string) => {
    const storedResults = getResult(targetSubmissionId);
    if (storedResults) {
      setData(storedResults);
      setStatus("success");
      setError(null);
      router.replace(`/results/${targetSubmissionId}`);

      const answerTexts = storedResults.meta?.answerTexts as Record<string, string> | undefined;
      if (answerTexts) {
        const firstAnswerId = Object.keys(answerTexts)[0];
        setAnswerText(answerTexts[firstAnswerId] || "");
      }
      return true;
    }
    return false;
  };

  return (
    <>
      <header className="header">
        <div className="header-content">
          <Link href="/" className="logo">
            Writeo
          </Link>
          {status === "pending" ? (
            <ModeSwitcher />
          ) : (
            <div style={{ display: "flex", gap: "12px", alignItems: "center" }}>
              <ModeSwitcher />
              <Link href="/" className="btn btn-secondary">
                ← Back to Tasks
              </Link>
            </div>
          )}
        </div>
      </header>
      <div className="container" style={{ minHeight: "calc(100vh - 200px)" }}>
        {status === "pending" && (
          <div
            style={{
              textAlign: "center",
              padding: "48px 24px",
              minHeight: "400px",
              display: "flex",
              flexDirection: "column",
              justifyContent: "center",
            }}
          >
            <h2 style={{ marginBottom: "16px" }}>Loading Results...</h2>
            <p style={{ color: "var(--text-secondary)" }}>Fetching your essay results...</p>
          </div>
        )}

        {status === "error" && (
          <div
            className="card"
            style={{
              maxWidth: "600px",
              margin: "48px auto",
              padding: "var(--spacing-xl)",
              textAlign: "center",
              minHeight: "400px",
              display: "flex",
              flexDirection: "column",
              justifyContent: "center",
            }}
          >
            <div
              style={{
                fontSize: "48px",
                marginBottom: "var(--spacing-md)",
              }}
            >
              📝
            </div>
            <h2 style={{ marginBottom: "var(--spacing-sm)", color: "var(--text-primary)" }}>
              Results Not Available
            </h2>
            <p
              style={{
                color: "var(--text-secondary)",
                marginBottom: "var(--spacing-lg)",
                lineHeight: "1.6",
              }}
            >
              {error ||
                "We couldn't load your results. This might happen if the submission ID is incorrect or the results are no longer available."}
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
                  processingTime={processingTime}
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
