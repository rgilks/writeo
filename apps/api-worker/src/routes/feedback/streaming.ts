/**
 * Streaming utilities for AI feedback
 */

import { callLLMAPI, streamLLMAPI, type LLMProvider } from "../../services/llm";
import { truncateEssayText, truncateQuestionText } from "../../utils/text-processing";
import {
  buildEssayContext,
  buildGrammarContext,
  normalizeEssayScores,
} from "../../services/feedback/context";
import type { FeedbackData } from "./storage";

type EssayScores = FeedbackData["essayScores"];
type GrammarErrors = FeedbackData["ltErrors"];

const STREAMING_SYSTEM_MESSAGE =
  "You are a professional writing tutor specializing in academic argumentative writing. Provide clear, direct feedback focused on actionable improvements. Never mention technical terms like CEFR or band scores.";
const STREAMING_MAX_TOKENS = 1000;

export function buildStreamingPrompt(
  questionText: string,
  answerText: string,
  essayScores?: EssayScores,
  ltErrors?: GrammarErrors,
): string {
  const normalizedScores = normalizeEssayScores(essayScores);
  const truncatedAnswerText = truncateEssayText(answerText);
  const truncatedQuestionText = truncateQuestionText(questionText);
  const essayContext = buildEssayContext(normalizedScores);
  const grammarContext = buildGrammarContext(ltErrors, undefined);

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
  prompt: string,
): Promise<ReadableStream> {
  return new ReadableStream({
    async start(controller) {
      const encoder = new TextEncoder();
      try {
        enqueueSSE(controller, encoder, {
          type: "start",
          message: "Starting AI feedback generation...",
        });

        let hasContent = false;
        try {
          for await (const chunk of streamLLMAPI(
            llmProvider,
            apiKey,
            aiModel,
            buildLLMMessages(prompt),
            STREAMING_MAX_TOKENS,
          )) {
            hasContent = true;
            enqueueSSE(controller, encoder, { type: "chunk", text: chunk });
          }
        } catch (streamError) {
          console.error("[Stream] Streaming failed, falling back to non-streaming:", streamError);
          const responseText = await callLLMAPI(
            llmProvider,
            apiKey,
            aiModel,
            buildLLMMessages(prompt),
            STREAMING_MAX_TOKENS,
          );

          if (responseText && responseText.trim().length > 0) {
            hasContent = true;
            await enqueueFallbackChunks(controller, encoder, responseText);
          }
        }

        if (!hasContent) {
          enqueueSSE(controller, encoder, {
            type: "chunk",
            text: "Unable to generate feedback at this time. Please try again.",
          });
        }

        enqueueSSE(controller, encoder, {
          type: "done",
          message: "Feedback generation complete",
        });
      } catch (error) {
        console.error("[Stream] Unexpected error during streaming response:", error);
        enqueueSSE(controller, encoder, {
          type: "error",
          message: error instanceof Error ? error.message : String(error),
        });
      } finally {
        controller.close();
      }
    },
  });
}

function buildLLMMessages(prompt: string) {
  return [
    {
      role: "system",
      content: STREAMING_SYSTEM_MESSAGE,
    },
    { role: "user", content: prompt },
  ];
}

function enqueueSSE(
  controller: ReadableStreamDefaultController,
  encoder: TextEncoder,
  payload: Record<string, unknown>,
) {
  controller.enqueue(encoder.encode(`data: ${JSON.stringify(payload)}\n\n`));
}

async function enqueueFallbackChunks(
  controller: ReadableStreamDefaultController,
  encoder: TextEncoder,
  responseText: string,
) {
  const chunks = responseText.match(/\S+|\s+/g) || [];
  if (chunks.length === 0) {
    enqueueSSE(controller, encoder, { type: "chunk", text: responseText });
    return;
  }

  for (const chunk of chunks) {
    enqueueSSE(controller, encoder, { type: "chunk", text: chunk });
    await new Promise((resolve) => setTimeout(resolve, 20));
  }
}
