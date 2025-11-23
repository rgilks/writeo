"use client";

import { useEffect, useState } from "react";
import { useParams, useSearchParams, useRouter } from "next/navigation";
import Link from "next/link";
import { getSubmissionResults, getSubmissionResultsWithDraftTracking } from "@/app/lib/actions";
import { usePreferencesStore } from "@/app/lib/stores/preferences-store";
import { useResultsStore } from "@/app/lib/stores/results-store";
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
  const mode = usePreferencesStore((state) => state.viewMode);
  const getResult = useResultsStore((state) => state.getResult);
  const getParentSubmissionId = useResultsStore((state) => state.getParentSubmissionId);
  const setResult = useResultsStore((state) => state.setResult);
  const [submissionStartTime] = useState<number>(Date.now());
  const [processingTime, setProcessingTime] = useState<number | null>(null);
  const [answerText, setAnswerText] = useState<string>("");

  // Get parent submission ID from results.meta (single source of truth)
  // URL param ?parent= is no longer needed since parentSubmissionId is in results.meta
  const [parentId, setParentId] = useState<string | null>(() => {
    // Try to get from initial results if available
    if (initialResults?.meta?.parentSubmissionId) {
      return initialResults.meta.parentSubmissionId as string;
    }
    // Fallback to results store
    return getParentSubmissionId(submissionId);
  });

  // Check if results were passed via router state (from write page) or stored locally
  const [initialResults] = useState<AssessmentResults | null>(() => {
    // Try to get results from sessionStorage first (immediate display from write page)
    if (typeof window !== "undefined") {
      const sessionStored = sessionStorage.getItem(`results_${submissionId}`);
      if (sessionStored) {
        try {
          const parsed = JSON.parse(sessionStored);
          sessionStorage.removeItem(`results_${submissionId}`); // Clean up
          return parsed;
        } catch {
          // Fall through to store check
        }
      }
      // Then check results store (persistent storage)
      return getResult(submissionId);
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
      // Update parentId from results.meta (single source of truth)
      const metaParentId = data.meta?.parentSubmissionId as string | undefined;
      if (metaParentId) {
        setParentId(metaParentId);
      }
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
        // Check if we're in local mode (not saving to server)
        const storeResults = usePreferencesStore.getState().storeResults;

        // Get parent results from results store if in local mode
        let parentResults: any = undefined;
        if (!storeResults && parentId) {
          parentResults = getResult(parentId);
        }

        // Try to fetch from server (only works if user opted in to server storage)
        // But first check if we have results in store or sessionStorage (from draft creation)
        const storedResults = getResult(submissionId);
        if (storedResults) {
          if (!cancelled) {
            // Update parentId from stored results if available
            const storedParentId = storedResults.meta?.parentSubmissionId as string | undefined;
            if (storedParentId && !parentId) {
              setParentId(storedParentId);
            }

            // Use stored results immediately, but still try to fetch from server in background
            setData(storedResults);
            setStatus("success");
            // Ensure results are stored in localStorage for tests and persistence
            if (typeof window !== "undefined") {
              try {
                localStorage.setItem(`results_${submissionId}`, JSON.stringify(storedResults));
              } catch (error) {
                console.warn("Failed to store results in localStorage:", error);
              }
            }

            // Try to fetch from server to get latest version (but don't wait)
            // This ensures we have the latest data if available
            getSubmissionResults(submissionId)
              .then((serverResults) => {
                if (!cancelled && serverResults) {
                  // Update with server results if available
                  setData(serverResults);
                  setResult(submissionId, serverResults);
                  // Also store in localStorage
                  if (typeof window !== "undefined") {
                    try {
                      localStorage.setItem(
                        `results_${submissionId}`,
                        JSON.stringify(serverResults)
                      );
                    } catch (error) {
                      console.warn("Failed to store results in localStorage:", error);
                    }
                  }
                }
              })
              .catch(() => {
                // Server fetch failed, but we already have local results, so that's OK
              });

            return;
          }
        }

        // Get parentId from stored results if not already set
        const effectiveParentId =
          parentId || (storedResults?.meta?.parentSubmissionId as string | undefined);
        const effectiveParentResults =
          effectiveParentId && !storeResults
            ? getResult(effectiveParentId) || parentResults
            : parentResults;

        try {
          const results = effectiveParentId
            ? await getSubmissionResultsWithDraftTracking(
                submissionId,
                effectiveParentId,
                storeResults,
                effectiveParentResults
              )
            : await getSubmissionResults(submissionId);
          if (!cancelled) {
            // Ensure questionTexts are preserved if they exist in initialResults
            const resultsToStore = { ...results };
            if (initialResults?.meta?.questionTexts && !resultsToStore.meta?.questionTexts) {
              if (!resultsToStore.meta) {
                resultsToStore.meta = {};
              }
              resultsToStore.meta.questionTexts = initialResults.meta.questionTexts;
            }
            setData(resultsToStore);
            setStatus("success");
            // Save to results store for future access
            setResult(submissionId, resultsToStore);
            // Also store in localStorage for tests and persistence
            if (typeof window !== "undefined") {
              try {
                localStorage.setItem(`results_${submissionId}`, JSON.stringify(resultsToStore));
              } catch (error) {
                console.warn("Failed to store results in localStorage:", error);
              }
            }
          }
        } catch (serverError) {
          // If server fetch fails (especially 404), try results store as fallback
          const fallbackResults = getResult(submissionId);
          if (fallbackResults) {
            if (!cancelled) {
              console.log(`[ResultsPage] Using results store data for submission ${submissionId}`);
              setData(fallbackResults);
              setStatus("success");
              return;
            }
          }
          // Check sessionStorage as well (temporary storage)
          if (typeof window !== "undefined") {
            const sessionStored = sessionStorage.getItem(`results_${submissionId}`);
            if (sessionStored) {
              try {
                const parsed = JSON.parse(sessionStored);
                if (!cancelled) {
                  console.log(
                    `[ResultsPage] Using sessionStorage data for submission ${submissionId}`
                  );
                  setData(parsed);
                  setStatus("success");
                  return;
                }
              } catch {
                // Fall through to error handling
              }
            }
          }
          // If both fail, show error
          throw serverError;
        }
      } catch (err) {
        if (!cancelled) {
          const errorMessage = err instanceof Error ? err.message : "Failed to fetch results";
          // Replace technical Server Component errors with friendly messages
          const friendlyMessage =
            errorMessage.includes("Server Component") ||
            errorMessage.includes("omitted in production") ||
            errorMessage.includes("Server Components") ||
            errorMessage.includes("not found")
              ? "We couldn't load your results. This might happen if the submission ID is incorrect, the results weren't saved on the server, or they're no longer available. Results are stored in your browser by default."
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

  // Function to switch drafts without page reload
  const switchDraft = (targetSubmissionId: string, rootSubmissionId?: string) => {
    // Check if results are in results store
    const storedResults = getResult(targetSubmissionId);
    if (storedResults) {
      setData(storedResults);
      setStatus("success");
      setError(null);

      // Update URL without reload
      // No need for ?parent= param since parentSubmissionId is in results.meta
      router.replace(`/results/${targetSubmissionId}`);

      // Update answer text
      const answerTexts = storedResults.meta?.answerTexts as Record<string, string> | undefined;
      if (answerTexts) {
        const firstAnswerId = Object.keys(answerTexts)[0];
        setAnswerText(answerTexts[firstAnswerId] || "");
      }
      return true; // Successfully switched
    }
    return false; // Results not found, will need to navigate
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
