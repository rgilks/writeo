"use client";

import { useEffect, useState } from "react";
import { useParams, useSearchParams, useRouter } from "next/navigation";
import Link from "next/link";
import { getSubmissionResults, getSubmissionResultsWithDraftTracking } from "@/app/lib/actions";
import { usePreferencesStore } from "@/app/lib/stores/preferences-store";
import { useDraftStore } from "@/app/lib/stores/draft-store";
import { LearnerResultsView } from "@/app/components/LearnerResultsView";
import { DeveloperResultsView } from "@/app/components/DeveloperResultsView";
import { ModeSwitcher } from "@/app/components/ModeSwitcher";
import { ErrorBoundary } from "@/app/components/ErrorBoundary";
import type { AssessmentResults } from "@writeo/shared";

// Helper to wait for Zustand persist rehydration
function waitForRehydration(): Promise<void> {
  return new Promise((resolve) => {
    // Check if store has already been hydrated
    if (useDraftStore.persist.hasHydrated()) {
      resolve();
      return;
    }

    // If not hydrated yet, wait for the onFinishHydration callback
    const unsubscribe = useDraftStore.persist.onFinishHydration(() => {
      unsubscribe();
      resolve();
    });

    // Fallback timeout in case hydration takes too long or fails silently
    setTimeout(() => {
      unsubscribe();
      resolve();
    }, 2000);
  });
}

export default function ResultsPage() {
  const params = useParams();
  const searchParams = useSearchParams();
  const router = useRouter();
  const submissionId = params.id as string;
  const mode = usePreferencesStore((state) => state.viewMode);
  const getResult = useDraftStore((state) => state.getResult);
  const getParentSubmissionId = useDraftStore((state) => state.getParentSubmissionId);
  const setResult = useDraftStore((state) => state.setResult);
  const [submissionStartTime] = useState<number>(Date.now());
  const [processingTime, setProcessingTime] = useState<number | null>(null);
  const [answerText, setAnswerText] = useState<string>("");

  // Track hydration state - results may not be available until store is hydrated
  const [isHydrated, setIsHydrated] = useState(() => useDraftStore.persist.hasHydrated());

  // Check if results are available in draft store after hydration
  // Don't read from store during SSR or before hydration
  const getInitialResults = (): AssessmentResults | null => {
    if (typeof window === "undefined" || !isHydrated) return null;
    return getResult(submissionId);
  };

  // Get parent submission ID from results.meta (single source of truth)
  const getInitialParentId = (): string | null => {
    if (typeof window === "undefined" || !isHydrated) return null;
    const results = getResult(submissionId);
    if (results?.meta?.parentSubmissionId) {
      return results.meta.parentSubmissionId as string;
    }
    return getParentSubmissionId(submissionId);
  };

  const [parentId, setParentId] = useState<string | null>(getInitialParentId);

  // Fetch results - initialized as pending, will check store after hydration
  const [data, setData] = useState<AssessmentResults | null>(getInitialResults);
  const [status, setStatus] = useState<"pending" | "success" | "error">(
    getInitialResults() ? "success" : "pending"
  );
  const [error, setError] = useState<string | null>(null);
  const [previousData, setPreviousData] = useState<AssessmentResults | null>(getInitialResults());

  // Listen for hydration completion
  useEffect(() => {
    if (useDraftStore.persist.hasHydrated()) {
      setIsHydrated(true);
      return;
    }

    const unsubscribe = useDraftStore.persist.onFinishHydration(() => {
      setIsHydrated(true);
    });

    return () => {
      unsubscribe();
    };
  }, []);

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
    // Wait for hydration before checking store
    if (!isHydrated) {
      return;
    }

    // Check if we already have results in the store (from write page or previous visit)
    const storedResults = getResult(submissionId);
    if (storedResults && status === "pending") {
      setData(storedResults);
      setStatus("success");
      const storedParentId = storedResults.meta?.parentSubmissionId as string | undefined;
      if (storedParentId && !parentId) {
        setParentId(storedParentId);
      }
      return;
    }

    // If we already have results, no need to fetch
    if (data && status === "success") {
      return;
    }

    let cancelled = false;

    async function fetchResults() {
      try {
        setStatus("pending");

        // Wait for Zustand persist to rehydrate from localStorage
        // This ensures we can read results that were just stored
        await waitForRehydration();

        // Check if we're in local mode (not saving to server)
        const storeResults = usePreferencesStore.getState().storeResults;

        // Get parent results from results store if in local mode
        let parentResults: any = undefined;
        if (!storeResults && parentId) {
          parentResults = getResult(parentId);
        }

        // Try to fetch from server (only works if user opted in to server storage)
        // But first check if we have results in store or sessionStorage (from draft creation)
        const currentStoredResults = getResult(submissionId);
        if (currentStoredResults) {
          if (!cancelled) {
            // Update parentId from stored results if available
            const storedParentId = currentStoredResults.meta?.parentSubmissionId as
              | string
              | undefined;
            if (storedParentId && !parentId) {
              setParentId(storedParentId);
            }

            // Use stored results immediately
            setData(currentStoredResults);
            setStatus("success");

            // Only try to fetch from server if user opted in to server storage
            // In local-only mode, skip server fetch to avoid unnecessary errors
            if (storeResults) {
              // Try to fetch from server to get latest version (but don't wait)
              // This ensures we have the latest data if available
              getSubmissionResults(submissionId)
                .then((serverResults) => {
                  if (!cancelled && serverResults) {
                    // Update with server results if available
                    setData(serverResults);
                    // Zustand persist handles localStorage automatically
                    setResult(submissionId, serverResults);
                  }
                })
                .catch(() => {
                  // Server fetch failed, but we already have local results, so that's OK
                });
            }

            return;
          }
        }

        // Get parentId from stored results if not already set
        const effectiveParentId =
          parentId || (currentStoredResults?.meta?.parentSubmissionId as string | undefined);
        const effectiveParentResults =
          effectiveParentId && !storeResults
            ? getResult(effectiveParentId) || parentResults
            : parentResults;

        // In local-only mode, don't try to fetch from server
        if (!storeResults) {
          // Results should already be in Zustand store (persisted to localStorage)
          const fallbackResults = getResult(submissionId);

          if (fallbackResults) {
            if (!cancelled) {
              const storedParentId = fallbackResults.meta?.parentSubmissionId as string | undefined;
              if (storedParentId && !parentId) {
                setParentId(storedParentId);
              }
              setData(fallbackResults);
              setStatus("success");
              return;
            }
          } else {
            throw new Error(
              "Results not found in local storage. In local-only mode, results are only stored in your browser. If you cleared your browser data, the results are no longer available."
            );
          }
        }

        // Server storage mode - try to fetch from server
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
            setData(results);
            setStatus("success");
            // Save to draft store (Zustand persist handles localStorage automatically)
            setResult(submissionId, results);
          }
        } catch (serverError) {
          // If server fetch fails (especially 404), try results store as fallback
          const fallbackResults = getResult(submissionId);
          if (fallbackResults) {
            if (!cancelled) {
              setData(fallbackResults);
              setStatus("success");
              return;
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

    if (submissionId && !data) {
      fetchResults();
    }

    return () => {
      cancelled = true;
    };
  }, [submissionId, parentId, isHydrated, data, status]);

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
