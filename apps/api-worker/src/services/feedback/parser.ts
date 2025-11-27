/**
 * JSON parsing and validation utilities for feedback responses
 */

import { z } from "zod";
import type { CombinedFeedback } from "./types";

const CombinedFeedbackSchema = z
  .object({
    detailed: z.object({
      relevance: z.object({
        addressesQuestion: z.boolean(),
        score: z.number().min(0).max(1),
        explanation: z.string().min(1),
      }),
      feedback: z.object({
        strengths: z.array(z.string()).default([]),
        improvements: z.array(z.string()).default([]),
        overall: z.string().min(1),
      }),
    }),
    teacher: z.object({
      message: z.string().min(1),
      focusArea: z.string().optional(),
      clues: z.string().optional(),
      explanation: z.string().optional(),
    }),
  })
  .strict();

export function parseFeedbackResponse(responseText: string): CombinedFeedback {
  const extractedJson = extractJsonPayload(responseText);
  if (!extractedJson) {
    throw new Error(
      `Could not extract JSON from AI response. Snippet: ${buildSnippet(responseText)}`,
    );
  }

  let parsed: unknown;
  try {
    parsed = JSON.parse(extractedJson);
  } catch (parseError) {
    const reason = parseError instanceof Error ? parseError.message : String(parseError);
    throw new Error(
      `Failed to parse AI JSON response (${reason}). Snippet: ${buildSnippet(extractedJson)}`,
    );
  }

  const validation = CombinedFeedbackSchema.safeParse(parsed);
  if (!validation.success) {
    throw new Error(
      `LLM JSON failed validation: ${validation.error.issues
        .map((issue) => `${issue.path.join(".") || "<root>"}: ${issue.message}`)
        .join("; ")}`,
    );
  }

  return validation.data;
}

function extractJsonPayload(rawResponse: string): string | null {
  const trimmed = rawResponse.trim();
  const fencedMatch = trimmed.match(/```(?:json)?\s*([\s\S]*?)```/i);
  const candidate = fencedMatch?.[1] ?? trimmed;
  return findFirstJsonObject(candidate);
}

function findFirstJsonObject(text: string): string | null {
  let startIndex = -1;
  let depth = 0;
  let inString = false;
  let escape = false;

  for (let i = 0; i < text.length; i++) {
    const char = text[i];

    if (inString) {
      if (escape) {
        escape = false;
      } else if (char === "\\") {
        escape = true;
      } else if (char === '"') {
        inString = false;
      }
      continue;
    }

    if (char === '"') {
      inString = true;
      continue;
    }

    if (char === "{") {
      if (depth === 0) {
        startIndex = i;
      }
      depth += 1;
      continue;
    }

    if (char === "}") {
      if (depth === 0) {
        continue;
      }
      depth -= 1;
      if (depth === 0 && startIndex !== -1) {
        return text.slice(startIndex, i + 1);
      }
    }
  }

  return null;
}

function buildSnippet(text: string, maxLength = 200): string {
  const normalized = text.replace(/\s+/g, " ").trim();
  return normalized.length <= maxLength ? normalized : `${normalized.slice(0, maxLength)}…`;
}
