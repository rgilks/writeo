"use client";

import { useState } from "react";
import Link from "next/link";
import type { AssessmentResults, LanguageToolError, AssessmentPart } from "@writeo/shared";

interface DeveloperResultsViewProps {
  data: AssessmentResults;
  answerText: string;
}

function CopyButton({ text, label }: { text: string; label?: string }) {
  const [copied, setCopied] = useState(false);

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(text);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
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
        backgroundColor: copied ? "#10b981" : "var(--bg-secondary)",
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

export function DeveloperResultsView({ data, answerText }: DeveloperResultsViewProps) {
  const parts: AssessmentPart[] = data.results?.parts || [];

  // Extract quick stats from metadata
  const meta = data.meta || {};
  const wordCount = meta.wordCount as number | undefined;
  const errorCount = meta.errorCount as number | undefined;
  const overallScore = meta.overallScore as number | undefined;
  const timestamp = meta.timestamp as string | undefined;
  const totalAssessors = parts.reduce((acc, part) => {
    return (
      acc +
      (part.answers || []).reduce((sum, answer) => {
        return sum + ((answer as any)["assessor-results"] || []).length;
      }, 0)
    );
  }, 0);

  return (
    <div className="container">
      <div style={{ marginBottom: "32px" }}>
        <h1 className="page-title">Developer Results View</h1>
        <p className="page-subtitle">Complete assessment data for evaluation and analysis</p>
      </div>

      {/* Quick Stats */}
      {(wordCount !== undefined ||
        errorCount !== undefined ||
        overallScore !== undefined ||
        timestamp ||
        totalAssessors > 0) && (
        <div className="card" style={{ marginBottom: "24px" }}>
          <h3 style={{ fontSize: "18px", fontWeight: 600, marginBottom: "16px" }}>Quick Stats</h3>
          <div
            style={{
              display: "grid",
              gridTemplateColumns: "repeat(auto-fit, minmax(200px, 1fr))",
              gap: "16px",
            }}
          >
            {timestamp && (
              <div>
                <div
                  style={{ fontSize: "12px", color: "var(--text-secondary)", marginBottom: "4px" }}
                >
                  Timestamp
                </div>
                <div style={{ fontSize: "14px", fontFamily: "monospace" }}>{timestamp}</div>
              </div>
            )}
            {wordCount !== undefined && (
              <div>
                <div
                  style={{ fontSize: "12px", color: "var(--text-secondary)", marginBottom: "4px" }}
                >
                  Word Count
                </div>
                <div style={{ fontSize: "18px", fontWeight: 600 }}>{wordCount}</div>
              </div>
            )}
            {errorCount !== undefined && (
              <div>
                <div
                  style={{ fontSize: "12px", color: "var(--text-secondary)", marginBottom: "4px" }}
                >
                  Errors
                </div>
                <div
                  style={{
                    fontSize: "18px",
                    fontWeight: 600,
                    color: errorCount > 0 ? "#dc2626" : "#10b981",
                  }}
                >
                  {errorCount}
                </div>
              </div>
            )}
            {overallScore !== undefined && (
              <div>
                <div
                  style={{ fontSize: "12px", color: "var(--text-secondary)", marginBottom: "4px" }}
                >
                  Overall Score
                </div>
                <div style={{ fontSize: "18px", fontWeight: 600 }}>{overallScore.toFixed(1)}</div>
              </div>
            )}
            {totalAssessors > 0 && (
              <div>
                <div
                  style={{ fontSize: "12px", color: "var(--text-secondary)", marginBottom: "4px" }}
                >
                  Total Assessors
                </div>
                <div style={{ fontSize: "18px", fontWeight: 600 }}>{totalAssessors}</div>
              </div>
            )}
            <div>
              <div
                style={{ fontSize: "12px", color: "var(--text-secondary)", marginBottom: "4px" }}
              >
                Parts
              </div>
              <div style={{ fontSize: "18px", fontWeight: 600 }}>{parts.length}</div>
            </div>
          </div>
        </div>
      )}

      {/* Raw Data Section */}
      <div className="card">
        <div
          style={{
            display: "flex",
            justifyContent: "space-between",
            alignItems: "center",
            marginBottom: "16px",
          }}
        >
          <h2 style={{ fontSize: "24px", fontWeight: 600, margin: 0 }}>Raw Assessment Data</h2>
          <CopyButton text={JSON.stringify(data, null, 2)} label="Copy All JSON" />
        </div>

        <div style={{ marginBottom: "24px" }}>
          <div
            style={{
              display: "flex",
              justifyContent: "space-between",
              alignItems: "center",
              marginBottom: "8px",
            }}
          >
            <h3 style={{ fontSize: "18px", fontWeight: 600, margin: 0 }}>Status</h3>
          </div>
          <code
            style={{
              padding: "8px 12px",
              backgroundColor:
                data.status === "success"
                  ? "#d1fae5"
                  : data.status === "error"
                    ? "#fef2f2"
                    : "var(--bg-secondary)",
              color:
                data.status === "success"
                  ? "#065f46"
                  : data.status === "error"
                    ? "#991b1b"
                    : "var(--text-primary)",
              borderRadius: "6px",
              fontSize: "14px",
              display: "inline-block",
              fontWeight: 600,
            }}
          >
            {data.status}
          </code>
        </div>

        <div style={{ marginBottom: "24px" }}>
          <div
            style={{
              display: "flex",
              justifyContent: "space-between",
              alignItems: "center",
              marginBottom: "8px",
            }}
          >
            <h3 style={{ fontSize: "18px", fontWeight: 600, margin: 0 }}>Template</h3>
            <CopyButton text={JSON.stringify(data.template, null, 2)} />
          </div>
          <pre
            style={{
              padding: "12px",
              backgroundColor: "var(--bg-secondary)",
              borderRadius: "6px",
              fontSize: "13px",
              overflow: "auto",
              margin: 0,
            }}
          >
            {JSON.stringify(data.template, null, 2)}
          </pre>
        </div>

        {data.meta && (
          <div style={{ marginBottom: "24px" }}>
            <div
              style={{
                display: "flex",
                justifyContent: "space-between",
                alignItems: "center",
                marginBottom: "8px",
              }}
            >
              <h3 style={{ fontSize: "18px", fontWeight: 600, margin: 0 }}>Metadata</h3>
              <CopyButton text={JSON.stringify(data.meta, null, 2)} />
            </div>
            <pre
              style={{
                padding: "12px",
                backgroundColor: "var(--bg-secondary)",
                borderRadius: "6px",
                fontSize: "13px",
                overflow: "auto",
                maxHeight: "300px",
                margin: 0,
              }}
            >
              {JSON.stringify(data.meta, null, 2)}
            </pre>
          </div>
        )}

        {data.error_message && (
          <div style={{ marginBottom: "24px" }}>
            <div
              style={{
                display: "flex",
                justifyContent: "space-between",
                alignItems: "center",
                marginBottom: "8px",
              }}
            >
              <h3 style={{ fontSize: "18px", fontWeight: 600, color: "#dc2626", margin: 0 }}>
                Error Message
              </h3>
              <CopyButton text={data.error_message} />
            </div>
            <div
              style={{
                padding: "12px",
                backgroundColor: "#fef2f2",
                border: "1px solid #fecaca",
                borderRadius: "6px",
                color: "#991b1b",
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
            <code
              style={{
                padding: "6px 10px",
                backgroundColor: part.status === "success" ? "#d1fae5" : "#fef2f2",
                color: part.status === "success" ? "#065f46" : "#991b1b",
                borderRadius: "4px",
                fontSize: "12px",
                fontWeight: 600,
              }}
            >
              Status: {part.status}
            </code>
          </div>

          <div style={{ display: "flex", flexDirection: "column", gap: "20px" }}>
            {/* answers array with assessor-results nested under each answer */}
            {(part.answers || []).map((answer: any, answerIndex: number) => {
              const assessorResults = answer["assessor-results"] || [];
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
                  {assessorResults.map((assessor: any, assessorIndex: number) => (
                    <div
                      key={assessorIndex}
                      style={{
                        padding: "20px",
                        backgroundColor: "var(--bg-secondary)",
                        borderRadius: "8px",
                        border: "1px solid var(--border-color)",
                        marginBottom: answerId ? "12px" : "0",
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
                          <h3 style={{ fontSize: "18px", fontWeight: 600, margin: 0 }}>
                            {assessor.name || assessor.id}
                          </h3>
                          <code
                            style={{
                              padding: "4px 8px",
                              backgroundColor: "white",
                              borderRadius: "4px",
                              fontSize: "11px",
                              fontWeight: 600,
                            }}
                          >
                            {assessor.id}
                          </code>
                          <code
                            style={{
                              padding: "4px 8px",
                              backgroundColor: "white",
                              borderRadius: "4px",
                              fontSize: "11px",
                            }}
                          >
                            Type: {assessor.type}
                          </code>
                        </div>
                      </div>

                      {assessor.overall !== undefined && (
                        <div style={{ marginBottom: "12px" }}>
                          <strong>Overall Score:</strong> {assessor.overall}
                        </div>
                      )}

                      {assessor.label && (
                        <div style={{ marginBottom: "12px" }}>
                          <strong>CEFR Label:</strong> {assessor.label}
                        </div>
                      )}

                      {assessor.dimensions && (
                        <div style={{ marginBottom: "12px" }}>
                          <strong>Dimensions:</strong>
                          <pre
                            style={{
                              marginTop: "8px",
                              padding: "8px",
                              backgroundColor: "white",
                              borderRadius: "4px",
                              fontSize: "12px",
                              overflow: "auto",
                            }}
                          >
                            {JSON.stringify(assessor.dimensions, null, 2)}
                          </pre>
                        </div>
                      )}

                      {assessor.errors &&
                        Array.isArray(assessor.errors) &&
                        assessor.errors.length > 0 && (
                          <div style={{ marginBottom: "12px" }}>
                            <strong>Errors ({assessor.errors.length}):</strong>
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
                              {assessor.errors.map(
                                (error: LanguageToolError, errorIndex: number) => (
                                  <div
                                    key={errorIndex}
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
                                    <div
                                      style={{ color: "var(--text-secondary)", marginTop: "4px" }}
                                    >
                                      {error.message}
                                    </div>
                                    {error.suggestions && error.suggestions.length > 0 && (
                                      <div style={{ marginTop: "4px", color: "#059669" }}>
                                        Suggestions: {error.suggestions.join(", ")}
                                      </div>
                                    )}
                                  </div>
                                ),
                              )}
                            </div>
                          </div>
                        )}

                      {assessor.meta && (
                        <div>
                          <strong>Metadata:</strong>
                          <pre
                            style={{
                              marginTop: "8px",
                              padding: "8px",
                              backgroundColor: "white",
                              borderRadius: "4px",
                              fontSize: "12px",
                              overflow: "auto",
                              maxHeight: "200px",
                            }}
                          >
                            {JSON.stringify(assessor.meta, null, 2)}
                          </pre>
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
                          <div
                            style={{ position: "absolute", top: "8px", right: "8px", zIndex: 1 }}
                          >
                            <CopyButton text={JSON.stringify(assessor, null, 2)} />
                          </div>
                          <pre
                            style={{
                              padding: "12px",
                              backgroundColor: "white",
                              borderRadius: "4px",
                              fontSize: "11px",
                              overflow: "auto",
                              maxHeight: "400px",
                              margin: 0,
                            }}
                          >
                            {JSON.stringify(assessor, null, 2)}
                          </pre>
                        </div>
                      </details>
                    </div>
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
          <div
            style={{
              display: "flex",
              justifyContent: "space-between",
              alignItems: "center",
              marginBottom: "16px",
            }}
          >
            <h2 style={{ fontSize: "24px", fontWeight: 600, margin: 0 }}>Answer Text</h2>
            <CopyButton text={answerText} />
          </div>
          <pre
            style={{
              padding: "16px",
              backgroundColor: "var(--bg-secondary)",
              borderRadius: "8px",
              fontSize: "14px",
              whiteSpace: "pre-wrap",
              wordWrap: "break-word",
              maxHeight: "500px",
              overflow: "auto",
              margin: 0,
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
          <pre
            style={{
              padding: "16px",
              backgroundColor: "var(--bg-secondary)",
              borderRadius: "8px",
              fontSize: "12px",
              overflow: "auto",
              maxHeight: "600px",
              margin: 0,
            }}
          >
            {JSON.stringify(data, null, 2)}
          </pre>
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
