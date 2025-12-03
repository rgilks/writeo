/**
 * Shared mock implementation for LLM APIs (OpenAI, Groq)
 * Returns deterministic responses to avoid API costs during tests
 * Enhanced with better detection, error scenarios, and input validation
 */

// Minimal delays for fast tests - can be overridden for specific scenarios
const MOCK_DELAY_MS_MIN = 1;
const MOCK_DELAY_MS_MAX = 5;
const STREAM_DELAY_MS_MIN = 1;
const STREAM_DELAY_MS_MAX = 3;

// Comprehensive keyword detection for better mock routing
const GRAMMAR_CHECK_KEYWORDS = [
  "grammar and language checker",
  "identify ALL grammar",
  "Find ALL grammar",
  "Check the ENTIRE text systematically",
  "expert English grammar checker",
  "pipe-delimited",
  "errorText|wordBefore|wordAfter",
];
const FEEDBACK_REQUEST_KEYWORDS = [
  "expert English language tutor",
  "Provide feedback",
  "detailed feedback",
  "strengths",
  "improvements",
  "JSON only",
];
const TEACHER_FEEDBACK_KEYWORDS = [
  "professional writing tutor",
  "Give clear, direct feedback",
  "writing instructor",
  "experienced writing instructor",
];

// Enhanced grammar response with more error patterns
const MOCK_GRAMMAR_RESPONSE = `I go|weekend|to|GRAMMAR|Verb tense error: Use past tense for past events|went|Verb tense|The verb 'go' should be in past tense 'went' when describing past events|error
We was|park.|playing|GRAMMAR|Subject-verb agreement error|were|Subject-verb agreement|'We' requires 'were', not 'was'|error
I have|football.|a|GRAMMAR|Verb tense error: Use past tense for past events|had|Verb tense|The verb 'have' should be in past tense 'had' when describing past events|error
they plays|together|GRAMMAR|Subject-verb agreement error|play|Subject-verb agreement|'They' requires plural verb 'play'|error
he are|nice|GRAMMAR|Subject-verb agreement error|is|Subject-verb agreement|'He' requires singular verb 'is'|error`;

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

// Mock error scenarios for testing error handling
export const MOCK_ERROR_SCENARIOS = {
  TIMEOUT: "MOCK_TIMEOUT",
  RATE_LIMIT: "MOCK_RATE_LIMIT",
  SERVER_ERROR: "MOCK_SERVER_ERROR",
  INVALID_RESPONSE: "MOCK_INVALID_RESPONSE",
} as const;

// Global flag to enable error scenarios in tests
let mockErrorScenario: string | null = null;

export function setMockErrorScenario(scenario: string | null): void {
  mockErrorScenario = scenario;
}

export function getMockErrorScenario(): string | null {
  return mockErrorScenario;
}

function getRandomDelay(min: number, max: number): number {
  return Math.random() * (max - min) + min;
}

function getUserMessage(messages: Array<{ role: string; content: string }>): string {
  return messages.find((m) => m.role === "user")?.content || "";
}

function getSystemMessage(messages: Array<{ role: string; content: string }>): string {
  return messages.find((m) => m.role === "system")?.content || "";
}

function detectRequestType(
  userMessage: string,
  systemMessage: string,
): "grammar" | "feedback" | "teacher" | "default" {
  const combinedText = `${systemMessage} ${userMessage}`.toLowerCase();

  // Check for grammar check patterns
  if (GRAMMAR_CHECK_KEYWORDS.some((keyword) => combinedText.includes(keyword.toLowerCase()))) {
    return "grammar";
  }

  // Check for combined feedback patterns (prioritize this over teacher feedback as it's more specific)
  if (FEEDBACK_REQUEST_KEYWORDS.some((keyword) => combinedText.includes(keyword.toLowerCase()))) {
    return "feedback";
  }

  // Check for teacher feedback patterns
  if (TEACHER_FEEDBACK_KEYWORDS.some((keyword) => combinedText.includes(keyword.toLowerCase()))) {
    return "teacher";
  }

  return "default";
}

function getTeacherFeedbackResponse(userMessage: string, systemMessage: string): string {
  const combinedText = `${systemMessage} ${userMessage}`.toLowerCase();

  if (combinedText.includes("clues") || combinedText.includes("hint")) {
    return MOCK_TEACHER_FEEDBACK_CLUES;
  }
  if (combinedText.includes("explanation") || combinedText.includes("detailed analysis")) {
    return MOCK_TEACHER_FEEDBACK_EXPLANATION;
  }
  return MOCK_TEACHER_FEEDBACK_DEFAULT;
}

function validateInput(messages: Array<{ role: string; content: string }>): void {
  if (!Array.isArray(messages) || messages.length === 0) {
    throw new Error("Mock LLM API: messages array is required and cannot be empty");
  }

  for (const msg of messages) {
    if (!msg.role || !msg.content) {
      throw new Error("Mock LLM API: each message must have 'role' and 'content'");
    }
  }
}

export async function mockCallLLMAPI(
  _apiKey: string,
  _modelName: string,
  messages: Array<{ role: string; content: string }>,
  _maxTokens: number,
): Promise<string> {
  // Validate input
  validateInput(messages);

  // Handle error scenarios for testing
  if (mockErrorScenario === MOCK_ERROR_SCENARIOS.TIMEOUT) {
    await new Promise((resolve) => setTimeout(resolve, 35000)); // Longer than timeout
    throw new Error("Mock LLM API: Request timeout");
  }

  if (mockErrorScenario === MOCK_ERROR_SCENARIOS.RATE_LIMIT) {
    const error = new Error("Mock LLM API: Rate limit exceeded") as any;
    error.status = 429;
    throw error;
  }

  if (mockErrorScenario === MOCK_ERROR_SCENARIOS.SERVER_ERROR) {
    const error = new Error("Mock LLM API: Internal server error") as any;
    error.status = 500;
    throw error;
  }

  if (mockErrorScenario === MOCK_ERROR_SCENARIOS.INVALID_RESPONSE) {
    return "Invalid JSON response {";
  }

  // Minimal delay for fast tests
  await new Promise((resolve) =>
    setTimeout(resolve, getRandomDelay(MOCK_DELAY_MS_MIN, MOCK_DELAY_MS_MAX)),
  );

  const userMessage = getUserMessage(messages);
  const systemMessage = getSystemMessage(messages);
  const requestType = detectRequestType(userMessage, systemMessage);

  switch (requestType) {
    case "grammar":
      return MOCK_GRAMMAR_RESPONSE;
    case "teacher":
      return getTeacherFeedbackResponse(userMessage, systemMessage);
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
  // Validate input
  validateInput(messages);

  // Handle error scenarios
  if (mockErrorScenario === MOCK_ERROR_SCENARIOS.TIMEOUT) {
    await new Promise((resolve) => setTimeout(resolve, 35000));
    throw new Error("Mock LLM API: Stream timeout");
  }

  if (mockErrorScenario === MOCK_ERROR_SCENARIOS.RATE_LIMIT) {
    const error = new Error("Mock LLM API: Rate limit exceeded") as any;
    error.status = 429;
    throw error;
  }

  const fullResponse = await mockCallLLMAPI(apiKey, modelName, messages, maxTokens);
  const words = fullResponse.match(/\S+|\s+/g) || [];

  for (const word of words) {
    yield word;
    // Minimal delay for fast streaming
    await new Promise((resolve) =>
      setTimeout(resolve, getRandomDelay(STREAM_DELAY_MS_MIN, STREAM_DELAY_MS_MAX)),
    );
  }
}
