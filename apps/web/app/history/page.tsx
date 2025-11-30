"use client";

import { useMemo } from "react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { useDraftStore } from "@/app/lib/stores/draft-store";
import { useStoreHydration } from "@/app/hooks/useStoreHydration";
import { getEssayAssessorResult } from "@writeo/shared";
import { mapScoreToCEFR } from "@writeo/shared";
import type { AssessmentResults } from "@writeo/shared";
import { getScoreColor } from "@/app/components/learner-results/utils";

interface GroupedItem {
  dateLabel: string;
  items: HistoryItem[];
}

interface HistoryItem {
  id: string;
  type: "content-draft" | "submission";
  timestamp: number;
  // Content draft fields
  content?: string;
  summary?: string;
  wordCount?: number;
  // Submission fields
  submissionId?: string;
  overallScore?: number;
  cefrLevel?: string;
  dimensions?: {
    TA?: number;
    CC?: number;
    Vocab?: number;
    Grammar?: number;
  };
  errorCount?: number;
  draftNumber?: number;
  questionText?: string;
}

function formatDate(timestamp: number): string {
  const date = new Date(timestamp);
  const today = new Date();
  const yesterday = new Date(today);
  yesterday.setDate(yesterday.getDate() - 1);
  const thisWeek = new Date(today);
  thisWeek.setDate(thisWeek.getDate() - 7);

  const dateStr = date.toISOString().split("T")[0];
  const todayStr = today.toISOString().split("T")[0];
  const yesterdayStr = yesterday.toISOString().split("T")[0];
  const thisWeekStr = thisWeek.toISOString().split("T")[0];

  if (dateStr === todayStr) return "Today";
  if (dateStr === yesterdayStr) return "Yesterday";
  if (dateStr >= thisWeekStr) return "This Week";
  return date.toLocaleDateString("en-US", { month: "short", day: "numeric", year: "numeric" });
}

function extractSubmissionData(
  submissionId: string,
  result: AssessmentResults,
): Omit<HistoryItem, "id" | "type" | "timestamp"> {
  const parts = result.results?.parts ?? [];
  const [firstPart] = parts;
  const [firstAnswer] = firstPart?.answers ?? [];
  const assessorResults = firstAnswer?.assessorResults ?? [];

  const essayAssessor = getEssayAssessorResult(assessorResults);
  const overall = essayAssessor?.overall;
  const dimensions = essayAssessor?.dimensions;

  const meta = result.meta ?? {};
  const errorCount = typeof meta.errorCount === "number" ? meta.errorCount : 0;
  const draftNumber = typeof meta.draftNumber === "number" ? meta.draftNumber : 1;
  const questionTexts = meta.questionTexts as Record<string, string> | undefined;
  const answerTexts = meta.answerTexts as Record<string, string> | undefined;
  const [answerId] = answerTexts ? Object.keys(answerTexts) : [];
  const questionText = (answerId && questionTexts?.[answerId]) || "";

  return {
    submissionId,
    overallScore: overall,
    cefrLevel: overall ? mapScoreToCEFR(overall) : undefined,
    dimensions: dimensions
      ? {
          TA: dimensions.TA,
          CC: dimensions.CC,
          Vocab: dimensions.Vocab,
          Grammar: dimensions.Grammar,
        }
      : undefined,
    errorCount,
    draftNumber,
    questionText,
  };
}

export default function HistoryPage() {
  const { contentDrafts, results } = useDraftStore();
  const isHydrated = useStoreHydration(useDraftStore);

  const historyItems = useMemo(() => {
    if (!isHydrated) return [];

    const items: HistoryItem[] = [];

    // Add content drafts
    contentDrafts.forEach((draft) => {
      items.push({
        id: draft.id,
        type: "content-draft",
        timestamp: draft.lastModified,
        content: draft.content,
        summary: draft.summary,
        wordCount: draft.wordCount,
      });
    });

    // Add submitted drafts
    Object.entries(results).forEach(([submissionId, storedResult]) => {
      try {
        const result = storedResult.results;
        if (result && result.status === "success") {
          const meta = result.meta ?? {};
          const timestamp =
            typeof meta.timestamp === "string"
              ? new Date(meta.timestamp).getTime()
              : storedResult.timestamp || Date.now();

          const submissionData = extractSubmissionData(submissionId, result);
          items.push({
            id: submissionId,
            type: "submission",
            timestamp,
            ...submissionData,
          });
        }
      } catch (error) {
        // Skip invalid results
        console.warn("Error processing submission result:", submissionId, error);
      }
    });

    // Sort by timestamp (most recent first)
    items.sort((a, b) => b.timestamp - a.timestamp);

    return items;
  }, [isHydrated, contentDrafts, results]);

  const groupedItems = useMemo(() => {
    const groups: Record<string, HistoryItem[]> = {};

    historyItems.forEach((item) => {
      const dateLabel = formatDate(item.timestamp);
      if (!groups[dateLabel]) {
        groups[dateLabel] = [];
      }
      groups[dateLabel].push(item);
    });

    const orderedGroups: GroupedItem[] = [];
    const order = ["Today", "Yesterday", "This Week"];

    // Add ordered groups first
    order.forEach((label) => {
      if (groups[label]) {
        orderedGroups.push({ dateLabel: label, items: groups[label] });
        delete groups[label];
      }
    });

    // Add remaining groups (older dates) sorted by date
    const remaining = Object.entries(groups)
      .map(([dateLabel, items]) => ({ dateLabel, items }))
      .sort((a, b) => {
        // Parse dates for sorting - use the first item's timestamp for accurate sorting
        const timestampA = a.items[0]?.timestamp || 0;
        const timestampB = b.items[0]?.timestamp || 0;
        return timestampB - timestampA;
      });

    orderedGroups.push(...remaining);

    return orderedGroups;
  }, [historyItems]);

  if (!isHydrated) {
    return (
      <>
        <header className="header">
          <div className="header-content">
            <div className="logo-group">
              <Link href="/" className="logo">
                Writeo
              </Link>
            </div>
            <nav className="header-actions" aria-label="Primary navigation">
              <Link href="/history" className="nav-history-link nav-history-link--active">
                <span aria-hidden="true">📜</span>
                <span>History</span>
              </Link>
              <Link href="/" className="nav-back-link">
                <span aria-hidden="true">←</span> Back to Home
              </Link>
            </nav>
          </div>
        </header>
        <div className="container">
          <div style={{ padding: "var(--spacing-xl)" }}>Loading...</div>
        </div>
      </>
    );
  }

  return (
    <>
      <header className="header">
        <div className="header-content">
          <div className="logo-group">
            <Link href="/" className="logo">
              Writeo
            </Link>
          </div>
          <nav className="header-actions" aria-label="Primary navigation">
            <Link href="/history" className="nav-history-link nav-history-link--active">
              <span aria-hidden="true">📜</span>
              <span>History</span>
            </Link>
            <Link href="/" className="nav-back-link">
              <span aria-hidden="true">←</span> Back to Home
            </Link>
          </nav>
        </div>
      </header>

      <div className="container" data-testid="history-page">
        <div style={{ marginBottom: "var(--spacing-xl)" }}>
          <h1 className="page-title" data-testid="history-page-title">
            History
          </h1>
          <p className="page-subtitle">
            View and access your drafts and submissions. Continue editing unsaved drafts or review
            your past work.
          </p>
        </div>

        {historyItems.length === 0 ? (
          <div
            className="card"
            style={{ textAlign: "center", padding: "var(--spacing-3xl)" }}
            data-testid="history-empty-state"
          >
            <div style={{ fontSize: "48px", marginBottom: "var(--spacing-md)" }}>📝</div>
            <h2 style={{ fontSize: "24px", fontWeight: 600, marginBottom: "var(--spacing-sm)" }}>
              No History Yet
            </h2>
            <p style={{ color: "var(--text-secondary)", marginBottom: "var(--spacing-lg)" }}>
              Start writing to see your drafts and submissions here.
            </p>
            <Link href="/" className="btn btn-primary">
              Start Writing
            </Link>
          </div>
        ) : (
          <div
            style={{ display: "flex", flexDirection: "column", gap: "var(--spacing-xl)" }}
            data-testid="history-items-container"
          >
            {groupedItems.map((group) => (
              <div
                key={group.dateLabel}
                data-testid={`history-group-${group.dateLabel.toLowerCase().replace(/\s+/g, "-")}`}
              >
                <h2
                  style={{
                    fontSize: "18px",
                    fontWeight: 600,
                    color: "var(--text-secondary)",
                    marginBottom: "var(--spacing-md)",
                    textTransform: "uppercase",
                    letterSpacing: "0.5px",
                  }}
                >
                  {group.dateLabel}
                </h2>
                <div
                  style={{
                    display: "grid",
                    gridTemplateColumns: "repeat(auto-fill, minmax(320px, 1fr))",
                    gap: "var(--spacing-md)",
                  }}
                  data-testid="history-cards-grid"
                >
                  {group.items.map((item) =>
                    item.type === "content-draft" ? (
                      <ContentDraftCard key={item.id} item={item} />
                    ) : (
                      <SubmissionCard key={item.id} item={item} />
                    ),
                  )}
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </>
  );
}

function ContentDraftCard({ item }: { item: HistoryItem }) {
  const { loadContentDraft } = useDraftStore();
  const router = useRouter();

  const handleContinue = () => {
    loadContentDraft(item.id);
    router.push("/write/custom");
  };

  return (
    <div
      className="card"
      style={{ display: "flex", flexDirection: "column", gap: "var(--spacing-md)" }}
      data-testid="content-draft-card"
    >
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start" }}>
        <div style={{ flex: 1 }}>
          <div
            style={{
              display: "inline-block",
              padding: "var(--spacing-xs) var(--spacing-sm)",
              backgroundColor: "var(--bg-tertiary)",
              borderRadius: "var(--border-radius)",
              fontSize: "12px",
              fontWeight: 600,
              color: "var(--text-secondary)",
              marginBottom: "var(--spacing-sm)",
            }}
          >
            Draft
          </div>
          <h3
            style={{
              fontSize: "16px",
              fontWeight: 600,
              marginBottom: "var(--spacing-xs)",
              color: "var(--text-primary)",
            }}
          >
            {item.summary || "Untitled Draft"}
          </h3>
        </div>
      </div>

      <div style={{ color: "var(--text-secondary)", fontSize: "14px" }}>
        <div style={{ marginBottom: "var(--spacing-xs)" }}>{item.wordCount || 0} words</div>
        <div style={{ fontSize: "12px" }}>
          {new Date(item.timestamp).toLocaleString("en-US", {
            month: "short",
            day: "numeric",
            hour: "numeric",
            minute: "2-digit",
          })}
        </div>
      </div>

      <button
        onClick={handleContinue}
        className="btn btn-secondary"
        style={{ marginTop: "auto" }}
        data-testid="continue-editing-button"
      >
        Continue Editing
      </button>
    </div>
  );
}

function SubmissionCard({ item }: { item: HistoryItem }) {
  const hasScore = typeof item.overallScore === "number" && item.overallScore > 0;
  const scoreColor =
    hasScore && item.overallScore !== undefined
      ? getScoreColor(item.overallScore)
      : "var(--text-secondary)";

  return (
    <Link
      href={`/results/${item.submissionId}`}
      style={{ textDecoration: "none", color: "inherit" }}
      data-testid="submission-card-link"
    >
      <div
        className="card"
        style={{ display: "flex", flexDirection: "column", gap: "var(--spacing-md)" }}
        data-testid="submission-card"
      >
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start" }}>
          <div style={{ flex: 1 }}>
            <div
              style={{
                display: "flex",
                gap: "var(--spacing-xs)",
                alignItems: "center",
                marginBottom: "var(--spacing-sm)",
                flexWrap: "wrap",
              }}
            >
              <div
                style={{
                  display: "inline-block",
                  padding: "var(--spacing-xs) var(--spacing-sm)",
                  backgroundColor: "var(--primary-bg-light)",
                  borderRadius: "var(--border-radius)",
                  fontSize: "12px",
                  fontWeight: 600,
                  color: "var(--primary-color)",
                }}
              >
                Submission
              </div>
              {item.draftNumber && item.draftNumber > 1 && (
                <div
                  style={{
                    display: "inline-block",
                    padding: "var(--spacing-xs) var(--spacing-sm)",
                    backgroundColor: "var(--bg-tertiary)",
                    borderRadius: "var(--border-radius)",
                    fontSize: "12px",
                    fontWeight: 600,
                    color: "var(--text-secondary)",
                  }}
                >
                  Draft #{item.draftNumber}
                </div>
              )}
            </div>
            {item.questionText && (
              <h3
                style={{
                  fontSize: "16px",
                  fontWeight: 600,
                  marginBottom: "var(--spacing-xs)",
                  color: "var(--text-primary)",
                  lineHeight: "1.4",
                }}
              >
                {item.questionText.length > 80
                  ? `${item.questionText.slice(0, 80)}...`
                  : item.questionText}
              </h3>
            )}
          </div>
        </div>

        {hasScore && item.overallScore !== undefined && (
          <div
            style={{
              display: "flex",
              alignItems: "center",
              gap: "var(--spacing-md)",
              padding: "var(--spacing-md)",
              backgroundColor: "var(--bg-secondary)",
              borderRadius: "var(--border-radius)",
            }}
          >
            <div
              style={{
                fontSize: "36px",
                fontWeight: 800,
                color: scoreColor,
                lineHeight: 1,
              }}
            >
              {item.overallScore.toFixed(1)}
            </div>
            <div style={{ flex: 1 }}>
              {item.cefrLevel && (
                <div
                  style={{
                    display: "inline-block",
                    padding: "var(--spacing-xs) var(--spacing-sm)",
                    backgroundColor: "var(--success-bg-light)",
                    borderRadius: "var(--border-radius)",
                    fontSize: "14px",
                    fontWeight: 600,
                    color: "var(--success-text)",
                    marginBottom: "var(--spacing-xs)",
                  }}
                >
                  {item.cefrLevel}
                </div>
              )}
              {item.dimensions && (
                <div
                  style={{
                    display: "flex",
                    gap: "var(--spacing-sm)",
                    flexWrap: "wrap",
                    marginTop: "var(--spacing-xs)",
                  }}
                >
                  {item.dimensions.TA !== undefined && (
                    <div style={{ fontSize: "12px", color: "var(--text-secondary)" }}>
                      TA: {item.dimensions.TA.toFixed(1)}
                    </div>
                  )}
                  {item.dimensions.CC !== undefined && (
                    <div style={{ fontSize: "12px", color: "var(--text-secondary)" }}>
                      CC: {item.dimensions.CC.toFixed(1)}
                    </div>
                  )}
                  {item.dimensions.Vocab !== undefined && (
                    <div style={{ fontSize: "12px", color: "var(--text-secondary)" }}>
                      Vocab: {item.dimensions.Vocab.toFixed(1)}
                    </div>
                  )}
                  {item.dimensions.Grammar !== undefined && (
                    <div style={{ fontSize: "12px", color: "var(--text-secondary)" }}>
                      Grammar: {item.dimensions.Grammar.toFixed(1)}
                    </div>
                  )}
                </div>
              )}
            </div>
          </div>
        )}

        <div
          style={{
            display: "flex",
            justifyContent: "space-between",
            color: "var(--text-secondary)",
            fontSize: "14px",
          }}
        >
          <div>
            {item.errorCount !== undefined && (
              <span style={{ marginRight: "var(--spacing-md)" }}>
                {item.errorCount} {item.errorCount === 1 ? "error" : "errors"}
              </span>
            )}
          </div>
          <div style={{ fontSize: "12px" }}>
            {new Date(item.timestamp).toLocaleString("en-US", {
              month: "short",
              day: "numeric",
              hour: "numeric",
              minute: "2-digit",
            })}
          </div>
        </div>

        <div style={{ marginTop: "auto" }}>
          <div
            className="btn btn-primary"
            style={{ width: "100%" }}
            data-testid="view-results-button"
          >
            View Results
          </div>
        </div>
      </div>
    </Link>
  );
}
