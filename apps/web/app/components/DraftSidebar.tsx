"use client";

import React from "react";
import { useDraftStore, type DraftContent } from "@/app/lib/stores/draft-store";

/**
 * DraftSidebar - Collapsible sidebar for navigating draft history
 * Displays drafts in chronological order (most recent first) with metadata
 */
export function DraftSidebar() {
  const drafts = useDraftStore((state) => state.contentDrafts);
  const activeDraftId = useDraftStore((state) => state.activeDraftId);
  const loadDraft = useDraftStore((state) => state.loadContentDraft);
  const createNewDraft = useDraftStore((state) => state.createNewContentDraft);
  const deleteDraft = useDraftStore((state) => state.deleteContentDraft);

  // Format timestamp for display
  const formatTime = (timestamp: number): string => {
    const date = new Date(timestamp);
    const now = new Date();
    const diffMs = now.getTime() - date.getTime();
    const diffMins = Math.floor(diffMs / 60000);
    const diffHours = Math.floor(diffMs / 3600000);
    const diffDays = Math.floor(diffMs / 86400000);

    if (diffMins < 1) return "Just now";
    if (diffMins < 60) return `${diffMins}m ago`;
    if (diffHours < 24) return `${diffHours}h ago`;
    if (diffDays < 7) return `${diffDays}d ago`;
    return date.toLocaleDateString();
  };

  const handleDraftClick = (id: string) => {
    loadDraft(id);
  };

  const handleDeleteClick = (e: React.MouseEvent, id: string) => {
    e.stopPropagation(); // Prevent loading the draft when clicking delete
    if (confirm("Are you sure you want to delete this draft?")) {
      deleteDraft(id);
    }
  };

  return (
    <aside
      className="draft-sidebar"
      style={{
        width: "280px",
        background: "#f4f4f5",
        padding: "20px",
        display: "flex",
        flexDirection: "column",
        borderRight: "1px solid #e4e4e7",
        height: "100vh",
        overflow: "hidden",
        flexShrink: 0,
      }}
      aria-label="Draft history"
    >
      <header
        style={{
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
          marginBottom: "20px",
        }}
      >
        <h2 style={{ margin: 0, fontSize: "1.2rem", fontWeight: 600 }}>Drafts</h2>
        <button
          onClick={createNewDraft}
          style={{
            padding: "6px 12px",
            background: "#2563eb",
            color: "white",
            border: "none",
            borderRadius: "4px",
            cursor: "pointer",
            fontSize: "0.875rem",
            fontWeight: 500,
          }}
          aria-label="Create new draft"
        >
          + New
        </button>
      </header>

      <div
        style={{
          flex: 1,
          overflowY: "auto",
          overflowX: "hidden",
        }}
      >
        {drafts.length === 0 && (
          <p
            style={{
              color: "#71717a",
              fontSize: "0.875rem",
              textAlign: "center",
              marginTop: "20px",
            }}
          >
            No drafts yet. Start writing to create your first draft.
          </p>
        )}

        {drafts.map((draft) => (
          <div
            key={draft.id}
            onClick={() => handleDraftClick(draft.id)}
            role="button"
            tabIndex={0}
            onKeyDown={(e) => {
              if (e.key === "Enter" || e.key === " ") {
                e.preventDefault();
                handleDraftClick(draft.id);
              }
            }}
            aria-label={`Draft from ${formatTime(draft.lastModified)}, ${draft.wordCount} words`}
            style={{
              padding: "12px",
              marginBottom: "8px",
              borderRadius: "6px",
              background: activeDraftId === draft.id ? "#dbeafe" : "white",
              border: activeDraftId === draft.id ? "1px solid #2563eb" : "1px solid #e4e4e7",
              cursor: "pointer",
              transition: "all 0.2s ease",
              position: "relative",
            }}
          >
            {/* Delete button */}
            <button
              onClick={(e) => handleDeleteClick(e, draft.id)}
              aria-label={`Delete draft from ${formatTime(draft.lastModified)}`}
              style={{
                position: "absolute",
                top: "8px",
                right: "8px",
                background: "transparent",
                border: "none",
                cursor: "pointer",
                fontSize: "1.2rem",
                lineHeight: 1,
                padding: "4px",
                color: "#71717a",
                opacity: 0.6,
                transition: "opacity 0.2s",
              }}
              onMouseEnter={(e) => {
                e.currentTarget.style.opacity = "1";
                e.currentTarget.style.color = "#dc2626";
              }}
              onMouseLeave={(e) => {
                e.currentTarget.style.opacity = "0.6";
                e.currentTarget.style.color = "#71717a";
              }}
            >
              ×
            </button>

            {/* Time */}
            <div
              style={{
                fontWeight: 600,
                fontSize: "0.9rem",
                marginBottom: "4px",
                paddingRight: "24px", // Space for delete button
              }}
            >
              {formatTime(draft.lastModified)}
            </div>

            {/* Summary */}
            <div
              style={{
                fontSize: "0.85rem",
                color: "#52525b",
                marginBottom: "6px",
                lineHeight: "1.4",
              }}
            >
              {draft.summary}
            </div>

            {/* Metadata */}
            <div
              style={{
                fontSize: "0.75rem",
                color: "#71717a",
                display: "flex",
                justifyContent: "space-between",
                alignItems: "center",
              }}
            >
              <span>{new Date(draft.lastModified).toLocaleDateString()}</span>
              <span>
                {draft.wordCount} {draft.wordCount === 1 ? "word" : "words"}
              </span>
            </div>
          </div>
        ))}
      </div>
    </aside>
  );
}
