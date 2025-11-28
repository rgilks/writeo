"use client";

import { useState, useMemo } from "react";
import Link from "next/link";
import type { AssessmentResults, LanguageToolError, AssessmentPart } from "@writeo/shared";

interface DeveloperResultsViewProps {
  data: AssessmentResults;
  answerText: string;
}

interface CopyButtonProps {
  text: string;
  label?: string;
}

const COPY_RESET_DELAY = 2000;

const STATUS_COLORS = {
  success: { bg: "#d1fae5", text: "#065f46" },
  error: { bg: "#fef2f2", text: "#991b1b" },
  default: { bg: "var(--bg-secondary)", text: "var(--text-primary)" },
} as const;

const ERROR_COLORS = {
  bg: "#fef2f2",
  border: "#fecaca",
  text: "#991b1b",
} as const;

const SUCCESS_COLOR = "#10b981";
const ERROR_COLOR = "#dc2626";

const CODE_BLOCK_STYLES = {
  padding: "12px",
  backgroundColor: "var(--bg-secondary)",
  borderRadius: "6px",
  fontSize: "13px",
  overflow: "auto",
  margin: 0,
} as const;

const SECTION_HEADER_STYLES = {
  display: "flex",
  justifyContent: "space-between",
  alignItems: "center",
  marginBottom: "16px",
} as const;

function CopyButton({ text, label }: CopyButtonProps) {
  const [copied, setCopied] = useState(false);

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(text);
      setCopied(true);
      setTimeout(() => setCopied(false), COPY_RESET_DELAY);
    } catch (err) {
      console.error("Failed to copy:", err);
    }
  };

  return (
    <button
      onClick={handleCopy}
      style={{
        padding: "6px 12px",
        fontSize: "12px",
        backgroundColor: copied ? SUCCESS_COLOR : "var(--bg-secondary)",
        color: copied ? "white" : "var(--text-primary)",
        border: "1px solid var(--border-color)",
        borderRadius: "4px",
        cursor: "pointer",
        fontWeight: 500,
        transition: "all 0.2s",
      }}
      title="Copy to clipboard"
    >
      {copied ? "✓ Copied" : label || "Copy"}
    </button>
  );
}

interface StatusBadgeProps {
  status: string;
  label?: string;
}

function StatusBadge({ status, label }: StatusBadgeProps) {
  const colors = STATUS_COLORS[status as keyof typeof STATUS_COLORS] || STATUS_COLORS.default;

  return (
    <code
      style={{
        padding: "8px 12px",
        backgroundColor: colors.bg,
        color: colors.text,
        borderRadius: "6px",
        fontSize: "14px",
        display: "inline-block",
        fontWeight: 600,
      }}
    >
      {label ? `${label}: ${status}` : status}
    </code>
  );
}

interface CodeBlockProps {
  content: string | object;
  maxHeight?: string;
  fontSize?: string;
}

function CodeBlock({ content, maxHeight, fontSize = "13px" }: CodeBlockProps) {
  const jsonString = typeof content === "string" ? content : JSON.stringify(content, null, 2);

  return (
    <pre
      style={{
        ...CODE_BLOCK_STYLES,
        fontSize,
        ...(maxHeight && { maxHeight }),
      }}
    >
      {jsonString}
    </pre>
  );
}

interface SectionHeaderProps {
  title: string;
  copyText?: string;
  copyLabel?: string;
}

function SectionHeader({ title, copyText, copyLabel }: SectionHeaderProps) {
  return (
    <div style={SECTION_HEADER_STYLES}>
      <h3 style={{ fontSize: "18px", fontWeight: 600, margin: 0 }}>{title}</h3>
      {copyText && <CopyButton text={copyText} label={copyLabel} />}
    </div>
  );
}

interface StatItemProps {
  label: string;
  value: string | number;
  color?: string;
  monospace?: boolean;
}

function StatItem({ label, value, color, monospace }: StatItemProps) {
  return (
    <div>
      <div style={{ fontSize: "12px", color: "var(--text-secondary)", marginBottom: "4px" }}>
        {label}
      </div>
      <div
        style={{
          fontSize: "18px",
          fontWeight: 600,
          ...(color && { color }),
          ...(monospace && { fontFamily: "monospace", fontSize: "14px" }),
        }}
      >
        {value}
      </div>
    </div>
  );
}

interface ErrorDisplayProps {
  error: LanguageToolError;
}

function ErrorDisplay({ error }: ErrorDisplayProps) {
  return (
    <div
      style={{
        padding: "8px",
        marginBottom: "4px",
        backgroundColor: "var(--bg-secondary)",
        borderRadius: "4px",
        fontSize: "12px",
      }}
    >
      <div>
        <strong>
          [{error.start}-{error.end}]
        </strong>{" "}
        {error.category} ({error.severity})
      </div>
      <div style={{ color: "var(--text-secondary)", marginTop: "4px" }}>{error.message}</div>
      {error.suggestions && error.suggestions.length > 0 && (
        <div style={{ marginTop: "4px", color: SUCCESS_COLOR }}>
          Suggestions: {error.suggestions.join(", ")}
        </div>
      )}
    </div>
  );
}

interface AssessorDisplayProps {
  assessor: Record<string, unknown>;
  assessorIndex: number;
}

function AssessorDisplay({ assessor, assessorIndex: _assessorIndex }: AssessorDisplayProps) {
  const assessorId = assessor.id as string;
  const assessorName = (assessor.name as string) || assessorId;
  const assessorType = assessor.type as string;
  const overall = assessor.overall as number | undefined;
  const label = assessor.label as string | undefined;
  const dimensions = assessor.dimensions;
  const errors = assessor.errors as LanguageToolError[] | undefined;
  const meta = assessor.meta;

  return (
    <div
      style={{
        padding: "20px",
        backgroundColor: "var(--bg-secondary)",
        borderRadius: "8px",
        border: "1px solid var(--border-color)",
        marginBottom: "12px",
      }}
    >
      <div style={{ marginBottom: "16px" }}>
        <div
          style={{
            display: "flex",
            gap: "12px",
            alignItems: "center",
            marginBottom: "8px",
          }}
        >
          <h3 style={{ fontSize: "18px", fontWeight: 600, margin: 0 }}>{assessorName}</h3>
          <code
            style={{
              padding: "4px 8px",
              backgroundColor: "white",
              borderRadius: "4px",
              fontSize: "11px",
              fontWeight: 600,
            }}
          >
            {assessorId}
          </code>
          <code
            style={{
              padding: "4px 8px",
              backgroundColor: "white",
              borderRadius: "4px",
              fontSize: "11px",
            }}
          >
            Type: {assessorType}
          </code>
        </div>
      </div>

      {overall !== undefined && (
        <div style={{ marginBottom: "12px" }}>
          <strong>Overall Score:</strong> {overall}
        </div>
      )}

      {label && (
        <div style={{ marginBottom: "12px" }}>
          <strong>CEFR Label:</strong> {label}
        </div>
      )}

      {dimensions != null && (
        <div style={{ marginBottom: "12px" }}>
          <strong>Dimensions:</strong>
          <CodeBlock content={dimensions as object} />
        </div>
      )}

      {errors && Array.isArray(errors) && errors.length > 0 && (
        <div style={{ marginBottom: "12px" }}>
          <strong>Errors ({errors.length}):</strong>
          <div
            style={{
              marginTop: "8px",
              maxHeight: "300px",
              overflowY: "auto",
              padding: "8px",
              backgroundColor: "white",
              borderRadius: "4px",
            }}
          >
            {errors.map((error, errorIndex) => (
              <ErrorDisplay key={errorIndex} error={error} />
            ))}
          </div>
        </div>
      )}

      {meta != null && (
        <div>
          <strong>Metadata:</strong>
          <CodeBlock content={meta as object} maxHeight="200px" />
        </div>
      )}

      {/* Full Assessor Object */}
      <details style={{ marginTop: "12px" }}>
        <summary
          style={{
            cursor: "pointer",
            fontSize: "13px",
            fontWeight: 500,
            color: "var(--text-secondary)",
          }}
        >
          View Full JSON
        </summary>
        <div style={{ position: "relative", marginTop: "8px" }}>
          <div style={{ position: "absolute", top: "8px", right: "8px", zIndex: 1 }}>
            <CopyButton text={JSON.stringify(assessor, null, 2)} />
          </div>
          <CodeBlock content={assessor} maxHeight="400px" fontSize="11px" />
        </div>
      </details>
    </div>
  );
}

interface AnswerWithAssessors {
  id?: string;
  "assessor-results"?: unknown[];
}

export function DeveloperResultsView({ data, answerText }: DeveloperResultsViewProps) {
  const parts: AssessmentPart[] = data.results?.parts || [];

  // Extract quick stats from metadata
  const meta = data.meta || {};
  const wordCount = meta.wordCount as number | undefined;
  const errorCount = meta.errorCount as number | undefined;
  const overallScore = meta.overallScore as number | undefined;
  const timestamp = meta.timestamp as string | undefined;

  const totalAssessors = useMemo(() => {
    return parts.reduce((acc, part) => {
      return (
        acc +
        (part.answers || []).reduce((sum, answer) => {
          const answerWithAssessors = answer as AnswerWithAssessors;
          return sum + (answerWithAssessors["assessor-results"] || []).length;
        }, 0)
      );
    }, 0);
  }, [parts]);

  const hasQuickStats = useMemo(() => {
    return (
      wordCount !== undefined ||
      errorCount !== undefined ||
      overallScore !== undefined ||
      timestamp ||
      totalAssessors > 0
    );
  }, [wordCount, errorCount, overallScore, timestamp, totalAssessors]);

  return (
    <div className="container">
      <div style={{ marginBottom: "32px" }}>
        <h1 className="page-title">Developer Results View</h1>
        <p className="page-subtitle">Complete assessment data for evaluation and analysis</p>
      </div>

      {/* Quick Stats */}
      {hasQuickStats && (
        <div className="card" style={{ marginBottom: "24px" }}>
          <h3 style={{ fontSize: "18px", fontWeight: 600, marginBottom: "16px" }}>Quick Stats</h3>
          <div
            style={{
              display: "grid",
              gridTemplateColumns: "repeat(auto-fit, minmax(200px, 1fr))",
              gap: "16px",
            }}
          >
            {timestamp && <StatItem label="Timestamp" value={timestamp} monospace />}
            {wordCount !== undefined && <StatItem label="Word Count" value={wordCount} />}
            {errorCount !== undefined && (
              <StatItem
                label="Errors"
                value={errorCount}
                color={errorCount > 0 ? ERROR_COLOR : SUCCESS_COLOR}
              />
            )}
            {overallScore !== undefined && (
              <StatItem label="Overall Score" value={overallScore.toFixed(1)} />
            )}
            {totalAssessors > 0 && <StatItem label="Total Assessors" value={totalAssessors} />}
            <StatItem label="Parts" value={parts.length} />
          </div>
        </div>
      )}

      {/* Raw Data Section */}
      <div className="card">
        <div style={SECTION_HEADER_STYLES}>
          <h2 style={{ fontSize: "24px", fontWeight: 600, margin: 0 }}>Raw Assessment Data</h2>
          <CopyButton text={JSON.stringify(data, null, 2)} label="Copy All JSON" />
        </div>

        <div style={{ marginBottom: "24px" }}>
          <SectionHeader title="Status" />
          <StatusBadge status={data.status} />
        </div>

        <div style={{ marginBottom: "24px" }}>
          <SectionHeader title="Template" copyText={JSON.stringify(data.template, null, 2)} />
          <CodeBlock content={data.template} />
        </div>

        {data.meta && (
          <div style={{ marginBottom: "24px" }}>
            <SectionHeader title="Metadata" copyText={JSON.stringify(data.meta, null, 2)} />
            <CodeBlock content={data.meta} maxHeight="300px" />
          </div>
        )}

        {data.error_message && (
          <div style={{ marginBottom: "24px" }}>
            <SectionHeader title="Error Message" copyText={data.error_message} />
            <div
              style={{
                padding: "12px",
                backgroundColor: ERROR_COLORS.bg,
                border: `1px solid ${ERROR_COLORS.border}`,
                borderRadius: "6px",
                color: ERROR_COLORS.text,
                fontFamily: "monospace",
                fontSize: "13px",
              }}
            >
              {data.error_message}
            </div>
          </div>
        )}
      </div>

      {/* Assessor Results */}
      {parts.map((part, partIndex) => (
        <div key={partIndex} className="card">
          <div
            style={{
              display: "flex",
              justifyContent: "space-between",
              alignItems: "center",
              marginBottom: "16px",
            }}
          >
            <h2 style={{ fontSize: "24px", fontWeight: 600, margin: 0 }}>
              Part {part.part} - Assessor Results
            </h2>
            <CopyButton text={JSON.stringify(part, null, 2)} />
          </div>

          <div style={{ marginBottom: "16px" }}>
            <StatusBadge status={part.status} label="Status" />
          </div>

          <div style={{ display: "flex", flexDirection: "column", gap: "20px" }}>
            {(part.answers || []).map((answer: AnswerWithAssessors, answerIndex: number) => {
              const assessorResults = (answer["assessor-results"] || []) as Record<
                string,
                unknown
              >[];
              const answerId = answer.id;

              return (
                <div key={answerIndex} style={{ marginBottom: "24px" }}>
                  {answerId && (
                    <div
                      style={{
                        marginBottom: "12px",
                        paddingBottom: "8px",
                        borderBottom: "1px solid var(--border-color)",
                      }}
                    >
                      <strong style={{ fontSize: "14px", color: "var(--text-secondary)" }}>
                        Answer ID: {answerId}
                      </strong>
                    </div>
                  )}
                  {assessorResults.map((assessor, assessorIndex) => (
                    <AssessorDisplay
                      key={assessorIndex}
                      assessor={assessor}
                      assessorIndex={assessorIndex}
                    />
                  ))}
                </div>
              );
            })}
          </div>
        </div>
      ))}

      {/* Answer Text */}
      {answerText && (
        <div className="card">
          <div style={SECTION_HEADER_STYLES}>
            <h2 style={{ fontSize: "24px", fontWeight: 600, margin: 0 }}>Answer Text</h2>
            <CopyButton text={answerText} />
          </div>
          <pre
            style={{
              ...CODE_BLOCK_STYLES,
              padding: "16px",
              fontSize: "14px",
              whiteSpace: "pre-wrap",
              wordWrap: "break-word",
              maxHeight: "500px",
            }}
          >
            {answerText}
          </pre>
        </div>
      )}

      {/* Full JSON Dump */}
      <div className="card">
        <details>
          <summary
            style={{
              cursor: "pointer",
              fontSize: "18px",
              fontWeight: 600,
              marginBottom: "12px",
              display: "flex",
              justifyContent: "space-between",
              alignItems: "center",
            }}
          >
            <span>Complete JSON Response</span>
            <CopyButton text={JSON.stringify(data, null, 2)} />
          </summary>
          <CodeBlock content={data} maxHeight="600px" fontSize="12px" />
        </details>
      </div>

      {/* Action Buttons */}
      <div
        style={{
          marginTop: "24px",
          display: "flex",
          gap: "12px",
          flexWrap: "wrap",
        }}
      >
        <Link href="/" className="btn btn-primary">
          Try Another Essay →
        </Link>
        <Link href="/" className="btn btn-secondary">
          ← Back to Tasks
        </Link>
      </div>
    </div>
  );
}
