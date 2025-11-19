"use client";

import { useEffect, useState } from "react";
import { useParams, useSearchParams, useRouter } from "next/navigation";
import Link from "next/link";
import { getSubmissionResults, getSubmissionResultsWithDraftTracking } from "@/app/lib/actions";
import { useMode } from "@/app/lib/mode-context";
import { LearnerResultsView } from "@/app/components/LearnerResultsView";
import { DeveloperResultsView } from "@/app/components/DeveloperResultsView";
import { ModeSwitcher } from "@/app/components/ModeSwitcher";
import { ErrorBoundary } from "@/app/components/ErrorBoundary";
import type { AssessmentResults } from "@writeo/shared";

export default function ResultsPage() {
  const params = useParams();
  const searchParams = useSearchParams();
  const router = useRouter();
  const submissionId = params.id as string;
  const { mode } = useMode();
  const [submissionStartTime] = useState<number>(Date.now());
  const [processingTime, setProcessingTime] = useState<number | null>(null);
  const [answerText, setAnswerText] = useState<string>("");

  // Get parent submission ID from URL params or localStorage
  const parentId =
    searchParams?.get("parent") ||
    (typeof window !== "undefined" ? localStorage.getItem(`draft_parent_${submissionId}`) : null);

  // Check if results were passed via router state (from write page)
  const [initialResults] = useState<AssessmentResults | null>(() => {
    // Try to get results from sessionStorage (set by write page)
    if (typeof window !== "undefined") {
      const stored = sessionStorage.getItem(`results_${submissionId}`);
      if (stored) {
        try {
          const parsed = JSON.parse(stored);
          sessionStorage.removeItem(`results_${submissionId}`); // Clean up
          return parsed;
        } catch {
          return null;
        }
      }
    }
    return null;
  });

  // Fetch results
  const [data, setData] = useState<AssessmentResults | null>(initialResults);
  const [status, setStatus] = useState<"pending" | "success" | "error">(
    initialResults ? "success" : "pending"
  );
  const [error, setError] = useState<string | null>(null);
  const [previousData, setPreviousData] = useState<AssessmentResults | null>(initialResults);

  useEffect(() => {
    // Update previousData when data changes (for draft switching)
    if (data) {
      setPreviousData(data);
    }
  }, [data]);

  useEffect(() => {
    // If we already have results from write page, skip fetching
    if (initialResults) {
      return;
    }

    let cancelled = false;

    async function fetchResults() {
      try {
        setStatus("pending");
        // Use draft tracking if parent ID is provided
        const results = parentId
          ? await getSubmissionResultsWithDraftTracking(submissionId, parentId)
          : await getSubmissionResults(submissionId);
        if (!cancelled) {
          setData(results);
          setStatus("success");
        }
      } catch (err) {
        if (!cancelled) {
          const errorMessage = err instanceof Error ? err.message : "Failed to fetch results";
          // Replace technical Server Component errors with friendly messages
          const friendlyMessage =
            errorMessage.includes("Server Component") ||
            errorMessage.includes("omitted in production") ||
            errorMessage.includes("Server Components")
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
  }, [submissionId, parentId, initialResults]);

  // Extract answer text and calculate processing time when results arrive
  useEffect(() => {
    if (data && status === "success") {
      // Get answer text from metadata
      const answerTexts = data.meta?.answerTexts as Record<string, string> | undefined;
      if (answerTexts) {
        const firstAnswerId = Object.keys(answerTexts)[0];
        setAnswerText(answerTexts[firstAnswerId] || "");
      }

      // Calculate processing time
      const endTime = Date.now();
      const elapsed = (endTime - submissionStartTime) / 1000; // Convert to seconds
      setProcessingTime(elapsed);
    }
  }, [data, status, submissionStartTime]);

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
        {status === "pending" && !previousData && (
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

        {status === "pending" && previousData && (
          <div style={{ opacity: 0.6, pointerEvents: "none", transition: "opacity 0.2s ease" }}>
            {mode === "developer" ? (
              <DeveloperResultsView data={previousData} answerText={answerText} />
            ) : (
              <ErrorBoundary>
                <LearnerResultsView
                  data={previousData}
                  answerText={answerText}
                  processingTime={processingTime}
                />
              </ErrorBoundary>
            )}
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
              {error &&
              (error.includes("Server Component") || error.includes("omitted in production"))
                ? "We couldn't load your results. This might happen if the submission ID is incorrect or the results are no longer available."
                : error ||
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
                  processingTime={processingTime}
                />
              </ErrorBoundary>
            )}
          </div>
        )}
      </div>
    </>
  );
}
