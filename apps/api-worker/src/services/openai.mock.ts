/**
 * Mock implementation of OpenAI API for testing
 * Returns deterministic responses to avoid API costs during tests
 */

export interface MockOpenAIResponse {
  content: string;
  usage?: {
    prompt_tokens: number;
    completion_tokens: number;
    total_tokens: number;
  };
}

/**
 * Mock OpenAI API call - returns deterministic responses based on input
 */
export async function mockCallOpenAIAPI(
  apiKey: string,
  modelName: string,
  messages: Array<{ role: string; content: string }>,
  maxTokens: number
): Promise<string> {
  // Simulate API delay (10-50ms)
  await new Promise((resolve) => setTimeout(resolve, Math.random() * 40 + 10));

  const userMessage = messages.find((m) => m.role === "user")?.content || "";
  const systemMessage = messages.find((m) => m.role === "system")?.content || "";

  // Detect what type of request this is based on the prompt
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
    // Mock grammar error detection response
    return JSON.stringify({
      errors: [
        {
          start: 0,
          end: 5,
          errorText: "I go",
          category: "GRAMMAR",
          message: "Verb tense error: Use past tense for past events",
          suggestions: ["I went"],
          errorType: "Verb tense",
          explanation: "The verb 'go' should be in past tense 'went' when describing past events",
          severity: "error",
        },
        {
          start: 20,
          end: 27,
          errorText: "We was",
          category: "GRAMMAR",
          message: "Subject-verb agreement error",
          suggestions: ["We were"],
          errorType: "Subject-verb agreement",
          explanation: "'We' requires 'were', not 'was'",
          severity: "error",
        },
      ],
    });
  }

  if (isTeacherFeedback) {
    // Mock teacher feedback response
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
    // Initial feedback
    return "Your essay shows good ideas, but there are some grammar errors that need attention. Focus on using correct verb tenses, especially when describing past events.";
  }

  if (isFeedbackRequest) {
    // Mock detailed feedback response
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

  // Default mock response
  return "Mock OpenAI API response";
}

/**
 * Mock OpenAI streaming API - yields text chunks asynchronously
 */
export async function* mockStreamOpenAIAPI(
  apiKey: string,
  modelName: string,
  messages: Array<{ role: string; content: string }>,
  maxTokens: number
): AsyncGenerator<string, void, unknown> {
  // Get the full response using the non-streaming mock
  const fullResponse = await mockCallOpenAIAPI(apiKey, modelName, messages, maxTokens);

  // Simulate streaming by yielding word by word with small delays
  const words = fullResponse.match(/\S+|\s+/g) || [];
  for (const word of words) {
    yield word;
    // Small delay to simulate network streaming (5-15ms per word)
    await new Promise((resolve) => setTimeout(resolve, Math.random() * 10 + 5));
  }
}
