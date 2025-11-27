/**
 * JSON parsing and validation utilities for feedback responses
 */

import type { CombinedFeedback } from "./types";

export function parseFeedbackResponse(responseText: string): CombinedFeedback {
  const trimmedResponseText = responseText.trim();
  const jsonMatch =
    trimmedResponseText.match(/```(?:json)?\s*(\{[\s\S]*\})\s*```/) ||
    trimmedResponseText.match(/(\{[\s\S]*\})/);

  if (jsonMatch && jsonMatch[1]) {
    try {
      const parsed = JSON.parse(jsonMatch[1]) as CombinedFeedback;
      if (!parsed.detailed || !parsed.teacher) {
        throw new Error("Missing required fields");
      }
      if (!parsed.detailed.relevance || !parsed.detailed.feedback) {
        throw new Error("Detailed feedback missing required fields");
      }
      if (!parsed.teacher.message) {
        throw new Error("Teacher feedback missing message");
      }
      return parsed;
    } catch (parseError) {
      throw new Error(
        `Failed to parse AI JSON response: ${parseError instanceof Error ? parseError.message : String(parseError)}`,
      );
    }
  }

  throw new Error(`Could not extract JSON from AI response`);
}
