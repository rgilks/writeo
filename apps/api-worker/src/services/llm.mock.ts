/**
 * Shared mock implementation for LLM APIs (OpenAI, Groq)
 * Returns deterministic responses to avoid API costs during tests
 */

const MOCK_DELAY_MS_MIN = 10;
const MOCK_DELAY_MS_MAX = 50;
const STREAM_DELAY_MS_MIN = 5;
const STREAM_DELAY_MS_MAX = 15;

const GRAMMAR_CHECK_KEYWORDS = ["grammar and language checker", "identify ALL grammar"];
const FEEDBACK_REQUEST_KEYWORDS = ["expert English language tutor", "Provide feedback"];
const TEACHER_FEEDBACK_KEYWORDS = ["professional writing tutor", "Give clear, direct feedback"];

const MOCK_GRAMMAR_RESPONSE = `0|5|I go|GRAMMAR|Verb tense error: Use past tense for past events|I went|Verb tense|The verb 'go' should be in past tense 'went' when describing past events|error
20|27|We was|GRAMMAR|Subject-verb agreement error|We were|Subject-verb agreement|'We' requires 'were', not 'was'|error`;

const MOCK_TEACHER_FEEDBACK_CLUES =
  "Try checking your verb tenses - look for words like 'yesterday' or 'last week' that indicate past time.";

const MOCK_TEACHER_FEEDBACK_EXPLANATION = `## Overall Assessment
The student's essay shows good effort but needs improvement in grammar accuracy and task completion.

## Task Achievement
- The student addressed the question but could develop ideas further
- Some parts of the question were not fully answered

## Grammar & Language Accuracy
- Multiple verb tense errors detected
- Subject-verb agreement issues present
- These errors affect clarity and should be corrected`;

const MOCK_TEACHER_FEEDBACK_DEFAULT =
  "Your essay shows good ideas, but there are some grammar errors that need attention. Focus on using correct verb tenses, especially when describing past events.";

const MOCK_FEEDBACK_RESPONSE = {
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
};

function getRandomDelay(min: number, max: number): number {
  return Math.random() * (max - min) + min;
}

function getUserMessage(messages: Array<{ role: string; content: string }>): string {
  return messages.find((m) => m.role === "user")?.content || "";
}

function detectRequestType(userMessage: string): "grammar" | "feedback" | "teacher" | "default" {
  if (GRAMMAR_CHECK_KEYWORDS.some((keyword) => userMessage.includes(keyword))) {
    return "grammar";
  }
  if (FEEDBACK_REQUEST_KEYWORDS.some((keyword) => userMessage.includes(keyword))) {
    return "feedback";
  }
  if (TEACHER_FEEDBACK_KEYWORDS.some((keyword) => userMessage.includes(keyword))) {
    return "teacher";
  }
  return "default";
}

function getTeacherFeedbackResponse(userMessage: string): string {
  if (userMessage.includes("clues")) {
    return MOCK_TEACHER_FEEDBACK_CLUES;
  }
  if (userMessage.includes("explanation")) {
    return MOCK_TEACHER_FEEDBACK_EXPLANATION;
  }
  return MOCK_TEACHER_FEEDBACK_DEFAULT;
}

export async function mockCallLLMAPI(
  _apiKey: string,
  _modelName: string,
  messages: Array<{ role: string; content: string }>,
  _maxTokens: number,
): Promise<string> {
  await new Promise((resolve) =>
    setTimeout(resolve, getRandomDelay(MOCK_DELAY_MS_MIN, MOCK_DELAY_MS_MAX)),
  );

  const userMessage = getUserMessage(messages);
  const requestType = detectRequestType(userMessage);

  switch (requestType) {
    case "grammar":
      return MOCK_GRAMMAR_RESPONSE;
    case "teacher":
      return getTeacherFeedbackResponse(userMessage);
    case "feedback":
      return JSON.stringify(MOCK_FEEDBACK_RESPONSE);
    default:
      return "Mock LLM API response";
  }
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
    await new Promise((resolve) =>
      setTimeout(resolve, getRandomDelay(STREAM_DELAY_MS_MIN, STREAM_DELAY_MS_MAX)),
    );
  }
}
