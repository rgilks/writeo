"use client";

import { useState } from "react";
import { useRouter, useParams } from "next/navigation";
import Link from "next/link";
import { submitEssay } from "@/app/lib/actions";

// Task data - matches tasks from home page
const taskData: Record<string, { title: string; prompt: string }> = {
  "1": {
    title: "Education: Practical vs Theoretical",
    prompt:
      "Some people believe that universities should focus more on practical skills rather than theoretical knowledge. To what extent do you agree or disagree?",
  },
  "2": {
    title: "Technology: Social Media Impact",
    prompt:
      "Some people think that social media has a negative impact on society. Others believe it brings people together and has positive effects. Discuss both views and give your own opinion.",
  },
  "3": {
    title: "Environment: Individual vs Government",
    prompt:
      "Some people think that individuals should be responsible for protecting the environment. Others believe that governments should take the lead. What is your view?",
  },
  "4": {
    title: "Work: Remote Working",
    prompt:
      "More and more people are working from home rather than in offices. What are the advantages and disadvantages of this trend?",
  },
  "5": {
    title: "Health: Fast Food Problem",
    prompt:
      "Fast food consumption is increasing worldwide, leading to health problems. What problems does this cause, and what solutions can you suggest?",
  },
  "6": {
    title: "Society: Ageing Population",
    prompt:
      "In many countries, the population is ageing. What are the causes of this trend, and what effects might it have on society?",
  },
  "7": {
    title: "Culture: Global vs Local",
    prompt:
      "Some people think that globalization means losing local culture and traditions. Others believe it enriches culture by bringing people together. To what extent do you agree or disagree?",
  },
  "8": {
    title: "Crime: Punishment vs Rehabilitation",
    prompt:
      "Some people think that criminals should be punished harshly to deter crime. Others believe that rehabilitation programs are more effective. Discuss both views and give your opinion.",
  },
};

export default function WritePage() {
  const params = useParams();
  const router = useRouter();
  const taskId = params.id as string;
  const task = taskData[taskId] || {
    title: "Writing Practice",
    prompt: "Write your essay here.",
  };

  const [answer, setAnswer] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [selfEval, setSelfEval] = useState({
    answeredAllParts: false,
    supportedOpinion: false,
    variedStructure: false,
  });

  // Handle textarea change with explicit state update
  const handleAnswerChange = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    const newValue = e.target.value;
    setAnswer(newValue);
  };

  // Return prompt as-is (no additional reminder text)
  const getPrompt = (basePrompt: string) => {
    return basePrompt;
  };

  // Calculate word count
  const wordCount = answer
    .trim()
    .split(/\s+/)
    .filter((w) => w.length > 0).length;
  const MIN_WORDS = 250;
  const MAX_WORDS = 500; // Soft cap - warn but allow

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();

    if (!answer.trim()) {
      setError("Please write your essay before submitting. Add your answer to receive feedback.");
      return;
    }

    // Validate word count
    if (wordCount < MIN_WORDS) {
      setError(
        `Your essay is too short. Please write at least ${MIN_WORDS} words (currently ${wordCount} words).`
      );
      return;
    }

    if (wordCount > MAX_WORDS) {
      setError(
        `Your essay is too long. Please keep it under ${MAX_WORDS} words (currently ${wordCount} words).`
      );
      return;
    }

    setLoading(true);
    setError(null);

    try {
      // Wrap Server Action call with timeout to prevent hanging
      const submitPromise = submitEssay(task.prompt, answer);
      const timeoutPromise = new Promise<never>((_, reject) => {
        setTimeout(() => reject(new Error("Request timed out. Please try again.")), 60000);
      });

      const { submissionId, results } = await Promise.race([submitPromise, timeoutPromise]);

      if (!submissionId || !results) {
        throw new Error("No submission ID or results returned");
      }
      // Store results in sessionStorage for immediate display (no loading page needed)
      if (typeof window !== "undefined") {
        sessionStorage.setItem(`results_${submissionId}`, JSON.stringify(results));
      }
      // Redirect to results page - results will be available immediately
      router.push(`/results/${submissionId}`);
    } catch (err) {
      console.error("Submission error:", err);
      // Extract error message safely and make it user-friendly
      let errorMessage = "We couldn't submit your essay. Please try again.";

      if (err instanceof Error) {
        const message = err.message;
        // Handle Server Component errors (production builds omit details)
        if (
          message.includes("Server Components render") ||
          message.includes("omitted in production builds") ||
          message.includes("digest property")
        ) {
          errorMessage =
            "We encountered an issue while processing your submission. Please try again.";
        } else if (
          message.includes("Server configuration error") ||
          message.includes("API_KEY") ||
          message.includes("API_BASE_URL")
        ) {
          errorMessage = "There's a server configuration issue. Please try again later.";
        } else if (message.includes("timeout") || message.includes("timed out")) {
          errorMessage = "The request took too long. Please try again.";
        } else if (
          message.includes("network") ||
          message.includes("fetch") ||
          message.includes("Failed to fetch")
        ) {
          errorMessage =
            "Unable to connect to our servers. Please check your internet connection and try again.";
        } else if (
          message.length > 0 &&
          message.length < 200 &&
          !message.includes("Error:") &&
          !message.includes("at ")
        ) {
          // Use the error message if it's user-friendly (short, no stack traces)
          errorMessage = message;
        }
      } else if (typeof err === "string") {
        // Check if it's a user-friendly string
        if (err.length < 200 && !err.includes("Error:") && !err.includes("at ")) {
          errorMessage = err;
        }
      }

      setError(errorMessage);
      setLoading(false);
    }
  };

  return (
    <>
      <header className="header" lang="en">
        <div className="header-content">
          <div className="logo-group">
            <Link href="/" className="logo" lang="en">
              Writeo
            </Link>
          </div>
          <nav className="header-actions" aria-label="Writing actions" lang="en">
            <Link href="/" className="nav-back-link" lang="en">
              <span aria-hidden="true">←</span> Back to Tasks
            </Link>
          </nav>
        </div>
      </header>

      <div className="container">
        <div style={{ marginBottom: "32px" }} lang="en">
          <h1 className="page-title">{task.title}</h1>
          <p className="page-subtitle">
            Write your essay and get detailed feedback to improve your writing.
          </p>
        </div>

        <div className="writing-container">
          <div className="card question-card">
            <h2
              lang="en"
              style={{
                fontSize: "20px",
                marginBottom: "16px",
                display: "flex",
                alignItems: "center",
                gap: "8px",
              }}
            >
              <span>📝</span> Question
            </h2>
            <div
              className="prompt-box notranslate"
              style={{ whiteSpace: "pre-wrap" }}
              translate="no"
              lang="en"
            >
              {getPrompt(task.prompt)}
            </div>
          </div>

          <div className="card answer-card">
            <form onSubmit={handleSubmit}>
              <label htmlFor="answer" className="label" lang="en">
                Your Answer
                <div
                  style={{
                    display: "flex",
                    gap: "var(--spacing-md)",
                    alignItems: "center",
                    fontSize: "14px",
                    color: "var(--text-secondary)",
                  }}
                >
                  <span>
                    {wordCount} {wordCount === 1 ? "word" : "words"}
                  </span>
                  {wordCount < MIN_WORDS && (
                    <span style={{ color: "var(--error-color)", fontWeight: 600 }}>
                      (Need at least {MIN_WORDS} words)
                    </span>
                  )}
                  {wordCount >= MIN_WORDS && wordCount <= MAX_WORDS && (
                    <span style={{ color: "var(--secondary-accent)" }}>✓</span>
                  )}
                  {wordCount > MAX_WORDS && (
                    <span style={{ color: "var(--error-color)", fontWeight: 600 }}>
                      (Too long - maximum {MAX_WORDS} words)
                    </span>
                  )}
                </div>
              </label>
              <textarea
                id="answer"
                className="textarea notranslate"
                value={answer}
                onChange={handleAnswerChange}
                onInput={handleAnswerChange}
                placeholder="Write your essay here. Minimum 250 words required. Aim for 250-300 words and address all parts of the question."
                rows={20}
                disabled={loading}
                autoFocus={false}
                translate="no"
                lang="en"
              />

              {/* Self-Evaluation Checklist */}
              {answer.trim().length > 50 && (
                <div
                  lang="en"
                  style={{
                    marginTop: "var(--spacing-md)",
                    padding: "var(--spacing-md)",
                    backgroundColor: "rgba(102, 126, 234, 0.1)",
                    borderRadius: "var(--border-radius)",
                  }}
                >
                  <p
                    style={{
                      marginBottom: "var(--spacing-sm)",
                      fontSize: "14px",
                      fontWeight: 600,
                    }}
                    lang="en"
                  >
                    ✓ Self-Evaluation Checklist (optional)
                  </p>
                  <div
                    style={{
                      display: "flex",
                      flexDirection: "column",
                      gap: "var(--spacing-sm)",
                    }}
                    lang="en"
                  >
                    <label
                      style={{
                        display: "flex",
                        alignItems: "center",
                        gap: "var(--spacing-sm)",
                        fontSize: "14px",
                        cursor: "pointer",
                        lineHeight: "1.5",
                      }}
                      lang="en"
                    >
                      <input
                        type="checkbox"
                        checked={selfEval.answeredAllParts}
                        onChange={(e) =>
                          setSelfEval({
                            ...selfEval,
                            answeredAllParts: e.target.checked,
                          })
                        }
                        style={{ cursor: "pointer" }}
                      />
                      Did I answer all parts of the question?
                    </label>
                    <label
                      style={{
                        display: "flex",
                        alignItems: "center",
                        gap: "var(--spacing-sm)",
                        fontSize: "14px",
                        cursor: "pointer",
                        lineHeight: "1.5",
                      }}
                      lang="en"
                    >
                      <input
                        type="checkbox"
                        checked={selfEval.supportedOpinion}
                        onChange={(e) =>
                          setSelfEval({
                            ...selfEval,
                            supportedOpinion: e.target.checked,
                          })
                        }
                        style={{ cursor: "pointer" }}
                      />
                      Did I support my opinion with at least two reasons?
                    </label>
                    <label
                      style={{
                        display: "flex",
                        alignItems: "center",
                        gap: "var(--spacing-sm)",
                        fontSize: "14px",
                        cursor: "pointer",
                        lineHeight: "1.5",
                      }}
                      lang="en"
                    >
                      <input
                        type="checkbox"
                        checked={selfEval.variedStructure}
                        onChange={(e) =>
                          setSelfEval({
                            ...selfEval,
                            variedStructure: e.target.checked,
                          })
                        }
                        style={{ cursor: "pointer" }}
                      />
                      Did I vary my sentence structure?
                    </label>
                  </div>
                </div>
              )}

              <div
                style={{
                  marginTop: "var(--spacing-md)",
                  display: "flex",
                  gap: "var(--spacing-md)",
                }}
              >
                <button
                  type="submit"
                  className="btn btn-primary"
                  disabled={loading || !answer.trim()}
                  style={{ flex: 1 }}
                  title="We value your privacy – see our policy"
                >
                  {loading ? (
                    <span
                      style={{
                        display: "flex",
                        alignItems: "center",
                        gap: "var(--spacing-sm)",
                      }}
                      lang="en"
                    >
                      <span className="spinner"></span>
                      Analyzing your writing…
                    </span>
                  ) : (
                    <span lang="en">Get Feedback →</span>
                  )}
                </button>
                <Link href="/" className="btn btn-secondary" lang="en">
                  Cancel
                </Link>
              </div>
              <p
                style={{
                  fontSize: "14px",
                  color: "var(--text-secondary)",
                  fontStyle: "italic",
                  marginTop: "var(--spacing-sm)",
                  lineHeight: "1.5",
                }}
                lang="en"
              >
                Your text is processed by an AI model; no one else reads it.{" "}
                <Link
                  href="/privacy"
                  style={{
                    color: "var(--primary-color)",
                    textDecoration: "underline",
                  }}
                >
                  See our privacy policy
                </Link>
                .
              </p>
            </form>
          </div>
        </div>

        {error && (
          <div className="error" role="alert" style={{ marginTop: "var(--spacing-md)" }}>
            <strong>⚠️ {error}</strong>
          </div>
        )}
      </div>
    </>
  );
}
