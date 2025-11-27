/**
 * Shared mock implementation for LLM APIs (OpenAI, Groq)
 * Returns deterministic responses to avoid API costs during tests
 */

export async function mockCallLLMAPI(
  _apiKey: string,
  _modelName: string,
  messages: Array<{ role: string; content: string }>,
  _maxTokens: number,
): Promise<string> {
  await new Promise((resolve) => setTimeout(resolve, Math.random() * 40 + 10));

  const userMessage = messages.find((m) => m.role === "user")?.content || "";

  const isGrammarCheck =
    userMessage.includes("grammar and language checker") ||
    userMessage.includes("identify ALL grammar");
  const isFeedbackRequest =
    userMessage.includes("expert English language tutor") ||
    userMessage.includes("Provide feedback");
  const isTeacherFeedback =
    userMessage.includes("professional writing tutor") ||
    userMessage.includes("Give clear, direct feedback");

  if (isGrammarCheck) {
    // Return pipe-delimited format: start|end|errorText|category|message|suggestions|errorType|explanation|severity
    return `0|5|I go|GRAMMAR|Verb tense error: Use past tense for past events|I went|Verb tense|The verb 'go' should be in past tense 'went' when describing past events|error
20|27|We was|GRAMMAR|Subject-verb agreement error|We were|Subject-verb agreement|'We' requires 'were', not 'was'|error`;
  }

  if (isTeacherFeedback) {
    if (userMessage.includes("clues")) {
      return "Try checking your verb tenses - look for words like 'yesterday' or 'last week' that indicate past time.";
    }
    if (userMessage.includes("explanation")) {
      return `## Overall Assessment
The student's essay shows good effort but needs improvement in grammar accuracy and task completion.

## Task Achievement
- The student addressed the question but could develop ideas further
- Some parts of the question were not fully answered

## Grammar & Language Accuracy
- Multiple verb tense errors detected
- Subject-verb agreement issues present
- These errors affect clarity and should be corrected`;
    }
    return "Your essay shows good ideas, but there are some grammar errors that need attention. Focus on using correct verb tenses, especially when describing past events.";
  }

  if (isFeedbackRequest) {
    return JSON.stringify({
      detailed: {
        relevance: {
          addressesQuestion: true,
          score: 0.85,
          explanation:
            "The answer addresses most parts of the question, though some aspects could be developed further.",
        },
        feedback: {
          strengths: [
            "Clear position stated in the introduction",
            "Good use of linking words to connect ideas",
            "Appropriate essay structure with introduction, body, and conclusion",
          ],
          improvements: [
            "Develop your second paragraph with a concrete example to support your main point",
            "Use more varied vocabulary instead of repeating 'important' - try 'crucial', 'significant', or 'vital'",
            "Check verb tenses for consistency, especially when describing past events",
          ],
          overall:
            "Your essay shows good structure and clear ideas. Focus on developing your arguments with specific examples and improving grammar accuracy.",
        },
      },
      teacher: {
        message:
          "Good structure and clear ideas. Work on adding specific examples to support your points and check your verb tenses.",
        focusArea: "improving grammar accuracy and using more varied sentence structures",
      },
    });
  }

  return "Mock LLM API response";
}

export async function* mockStreamLLMAPI(
  apiKey: string,
  modelName: string,
  messages: Array<{ role: string; content: string }>,
  maxTokens: number,
): AsyncGenerator<string, void, unknown> {
  const fullResponse = await mockCallLLMAPI(apiKey, modelName, messages, maxTokens);
  const words = fullResponse.match(/\S+|\s+/g) || [];
  for (const word of words) {
    yield word;
    await new Promise((resolve) => setTimeout(resolve, Math.random() * 10 + 5));
  }
}
