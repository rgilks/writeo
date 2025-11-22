/**
 * Streaming utilities for AI feedback
 */

import { callLLMAPI, streamLLMAPI, type LLMProvider } from "../../services/llm";
import { truncateEssayText, truncateQuestionText } from "../../utils/text-processing";

export function buildStreamingPrompt(
  questionText: string,
  answerText: string,
  essayScores?: any,
  ltErrors?: any[]
): string {
  const truncatedAnswerText = truncateEssayText(answerText);
  const truncatedQuestionText = truncateQuestionText(questionText);

  let essayContext = "";
  if (essayScores) {
    essayContext = `\n\nAssessment Results:
- Overall Score: ${essayScores.overall ?? essayScores.dimensions?.Overall ?? "N/A"} / 9.0
- Task Achievement (TA): ${essayScores.dimensions?.TA ?? "N/A"} / 9.0
- Coherence & Cohesion (CC): ${essayScores.dimensions?.CC ?? "N/A"} / 9.0
- Vocabulary: ${essayScores.dimensions?.Vocab ?? "N/A"} / 9.0
- Grammar: ${essayScores.dimensions?.Grammar ?? "N/A"} / 9.0`;
  }

  let grammarContext = "";
  if (ltErrors && ltErrors.length > 0) {
    const errorSummary = ltErrors
      .slice(0, 10)
      .map(
        (err, idx) =>
          `${idx + 1}. ${err.message} (${err.category})${err.suggestions ? ` - Suggestions: ${err.suggestions.slice(0, 2).join(", ")}` : ""}`
      )
      .join("\n");
    grammarContext = `\n\nGrammar & Language Issues Found (${ltErrors.length} total):\n${errorSummary}${ltErrors.length > 10 ? `\n... and ${ltErrors.length - 10} more issues` : ""}`;
  }

  return `You are an expert English language tutor specializing in academic argumentative writing. Analyze the following essay answer and provide detailed, contextual feedback.

Question: ${truncatedQuestionText}

Answer: ${truncatedAnswerText}${essayContext}${grammarContext}

Provide comprehensive feedback focusing on:

1. Task Achievement:
- Does the answer address ALL parts of the question?
- Is there a clear position/opinion stated (for opinion questions)?
- Are ideas fully developed with specific examples and explanations?
- For discussion questions: Are both sides addressed before giving an opinion?

2. Structure & Organization:
- Is there a clear introduction that presents the topic and position?
- Are body paragraphs well-organized with topic sentences?
- Does each paragraph focus on one main idea?
- Is there a conclusion that summarizes the main points?

3. Coherence & Cohesion:
- Are ideas connected smoothly using linking words (however, furthermore, therefore, etc.)?
- Do paragraphs flow logically from one to the next?
- Is the writing cohesive and easy to follow?

4. Vocabulary:
- Is there a good range of vocabulary (avoiding repetition)?
- Are words used accurately and appropriately?
- Are there attempts at less common vocabulary?

5. Grammar:
- Is grammar accurate?
- Is there a variety of sentence structures?

Provide feedback covering:
- Relevance: Does the answer fully address the question? (explain which parts were addressed and which might be missing)
- Strengths: 2-3 specific things the student did well (be specific, e.g., "Clear position stated in introduction" or "Good use of linking words")
- Improvements: 2-3 specific, actionable areas to improve (prioritize the most important issues, e.g., "Develop your second paragraph with a concrete example" or "Use more varied vocabulary instead of repeating 'important'")
- Overall: A brief summary comment (1-2 sentences) that focuses on improvement and suggests next steps

Respond in a clear, professional manner. Don't mention technical terms like "CEFR" or "band scores" - focus on actionable feedback to improve their academic writing skills.`;
}

export async function createStreamingResponse(
  llmProvider: LLMProvider,
  apiKey: string,
  aiModel: string,
  prompt: string
): Promise<ReadableStream> {
  return new ReadableStream({
    async start(controller) {
      const encoder = new TextEncoder();
      try {
        controller.enqueue(
          encoder.encode(
            `data: ${JSON.stringify({ type: "start", message: "Starting AI feedback generation..." })}\n\n`
          )
        );

        let hasContent = false;
        try {
          for await (const chunk of streamLLMAPI(
            llmProvider,
            apiKey,
            aiModel,
            [
              {
                role: "system",
                content:
                  "You are a professional writing tutor specializing in academic argumentative writing. Provide clear, direct feedback focused on actionable improvements. Never mention technical terms like , CEFR, or band scores.",
              },
              { role: "user", content: prompt },
            ],
            1000
          )) {
            hasContent = true;
            controller.enqueue(
              encoder.encode(`data: ${JSON.stringify({ type: "chunk", text: chunk })}\n\n`)
            );
          }
        } catch (streamError) {
          console.error("[Stream] Streaming failed, falling back to non-streaming:", streamError);
          const responseText = await callLLMAPI(
            llmProvider,
            apiKey,
            aiModel,
            [
              {
                role: "system",
                content:
                  "You are a professional writing tutor specializing in academic argumentative writing. Provide clear, direct feedback focused on actionable improvements. Never mention technical terms like , CEFR, or band scores.",
              },
              { role: "user", content: prompt },
            ],
            1000
          );

          if (responseText && responseText.trim().length > 0) {
            hasContent = true;
            const words = responseText.match(/\S+|\s+/g) || [];
            if (words.length === 0) {
              controller.enqueue(
                encoder.encode(`data: ${JSON.stringify({ type: "chunk", text: responseText })}\n\n`)
              );
            } else {
              for (const word of words) {
                controller.enqueue(
                  encoder.encode(`data: ${JSON.stringify({ type: "chunk", text: word })}\n\n`)
                );
                await new Promise((resolve) => setTimeout(resolve, 20));
              }
            }
          }
        }

        if (!hasContent) {
          controller.enqueue(
            encoder.encode(
              `data: ${JSON.stringify({ type: "chunk", text: "Unable to generate feedback at this time. Please try again." })}\n\n`
            )
          );
        }

        controller.enqueue(
          encoder.encode(
            `data: ${JSON.stringify({ type: "done", message: "Feedback generation complete" })}\n\n`
          )
        );
      } catch (error) {
        controller.enqueue(
          encoder.encode(
            `data: ${JSON.stringify({ type: "error", message: error instanceof Error ? error.message : String(error) })}\n\n`
          )
        );
      } finally {
        controller.close();
      }
    },
  });
}
