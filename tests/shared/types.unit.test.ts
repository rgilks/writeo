import { describe, it, expect } from "vitest";
import {
  mapScoreToCEFR,
  isValidUUID,
  findAssessorResultById,
  isAssessorResultWithId,
  getEssayAssessorResult,
  getFeedbackAssessorResult,
  getLanguageToolAssessorResult,
  getLLMAssessorResult,
  getTeacherFeedbackAssessorResult,
  getRelevanceCheckAssessorResult,
  type AssessorResult,
  type LanguageToolError,
} from "../../packages/shared/ts/types";

describe("mapScoreToCEFR", () => {
  it("should return C2 for scores >= 8.5", () => {
    expect(mapScoreToCEFR(8.5)).toBe("C2");
    expect(mapScoreToCEFR(9.0)).toBe("C2");
    expect(mapScoreToCEFR(10.0)).toBe("C2");
  });

  it("should return C1 for scores >= 7.0 and < 8.5", () => {
    expect(mapScoreToCEFR(7.0)).toBe("C1");
    expect(mapScoreToCEFR(8.4)).toBe("C1");
    expect(mapScoreToCEFR(7.5)).toBe("C1");
  });

  it("should return B2 for scores >= 5.5 and < 7.0", () => {
    expect(mapScoreToCEFR(5.5)).toBe("B2");
    expect(mapScoreToCEFR(6.9)).toBe("B2");
    expect(mapScoreToCEFR(6.0)).toBe("B2");
  });

  it("should return B1 for scores >= 4.0 and < 5.5", () => {
    expect(mapScoreToCEFR(4.0)).toBe("B1");
    expect(mapScoreToCEFR(5.4)).toBe("B1");
    expect(mapScoreToCEFR(4.5)).toBe("B1");
  });

  it("should return A2 for scores < 4.0", () => {
    expect(mapScoreToCEFR(3.9)).toBe("A2");
    expect(mapScoreToCEFR(0)).toBe("A2");
    expect(mapScoreToCEFR(1.5)).toBe("A2");
  });
});

describe("isValidUUID", () => {
  it("should validate correct UUID v4 format", () => {
    expect(isValidUUID("550e8400-e29b-41d4-a716-446655440000")).toBe(true);
    expect(isValidUUID("123e4567-e89b-12d3-a456-426614174000")).toBe(true);
  });

  it("should be case-insensitive", () => {
    expect(isValidUUID("550E8400-E29B-41D4-A716-446655440000")).toBe(true);
    expect(isValidUUID("550e8400-E29B-41d4-a716-446655440000")).toBe(true);
  });

  it("should reject invalid UUID formats", () => {
    expect(isValidUUID("not-a-uuid")).toBe(false);
    expect(isValidUUID("550e8400-e29b-41d4-a716")).toBe(false);
    expect(isValidUUID("550e8400e29b41d4a716446655440000")).toBe(false);
    expect(isValidUUID("")).toBe(false);
  });

  it("should reject non-UUID strings", () => {
    expect(isValidUUID("hello-world")).toBe(false);
    expect(isValidUUID("12345")).toBe(false);
  });
});

describe("findAssessorResultById", () => {
  const results: AssessorResult[] = [
    { id: "AES-ESSAY", name: "Essay", type: "grader", overall: 7.5 },
    { id: "GEC-LT", name: "LanguageTool", type: "feedback", errors: [] },
    { id: "GEC-LLM", name: "LLM", type: "feedback", errors: [] },
  ];

  it("should find assessor result by ID", () => {
    const result = findAssessorResultById(results, "AES-ESSAY");
    expect(result).toBeDefined();
    expect(result?.id).toBe("AES-ESSAY");
  });

  it("should return undefined if not found", () => {
    const result = findAssessorResultById(results, "NOT-FOUND" as any);
    expect(result).toBeUndefined();
  });

  it("should return undefined for empty array", () => {
    const result = findAssessorResultById([], "AES-ESSAY");
    expect(result).toBeUndefined();
  });
});

describe("isAssessorResultWithId", () => {
  const result: AssessorResult = { id: "AES-ESSAY", name: "Essay", type: "grader" };

  it("should return true for matching ID", () => {
    expect(isAssessorResultWithId(result, "AES-ESSAY")).toBe(true);
  });

  it("should return false for non-matching ID", () => {
    expect(isAssessorResultWithId(result, "GEC-LT")).toBe(false);
  });

  it("should act as type guard", () => {
    if (isAssessorResultWithId(result, "AES-ESSAY")) {
      // TypeScript should know result.id is "AES-ESSAY" here
      expect(result.id).toBe("AES-ESSAY");
    }
  });
});

describe("getEssayAssessorResult", () => {
  it("should return essay assessor with required fields", () => {
    const results: AssessorResult[] = [
      {
        id: "AES-ESSAY",
        name: "Essay",
        type: "grader",
        overall: 7.5,
        dimensions: { TA: 7, CC: 8, Vocab: 7, Grammar: 8 },
      },
    ];

    const result = getEssayAssessorResult(results);
    expect(result).toBeDefined();
    expect(result?.id).toBe("AES-ESSAY");
    expect(result?.overall).toBe(7.5);
    expect(result?.dimensions).toBeDefined();
  });

  it("should return undefined if not found", () => {
    const results: AssessorResult[] = [{ id: "GEC-LT", name: "LanguageTool", type: "feedback" }];
    expect(getEssayAssessorResult(results)).toBeUndefined();
  });

  it("should return undefined if overall is missing", () => {
    const results: AssessorResult[] = [
      { id: "AES-ESSAY", name: "Essay", type: "grader", dimensions: {} },
    ];
    expect(getEssayAssessorResult(results)).toBeUndefined();
  });

  it("should return undefined if dimensions is missing", () => {
    const results: AssessorResult[] = [
      { id: "AES-ESSAY", name: "Essay", type: "grader", overall: 7.5 },
    ];
    expect(getEssayAssessorResult(results)).toBeUndefined();
  });
});

describe("getFeedbackAssessorResult", () => {
  it("should return feedback assessor with overall and cefr", () => {
    const results: AssessorResult[] = [
      {
        id: "AES-FEEDBACK",
        name: "AES-FEEDBACK (Multi-Task)",
        type: "grader",
        overall: 3.87,
        cefr: "B1",
        meta: {
          errorSpans: [],
          errorTypes: { grammar: 0.6, vocabulary: 0.2 },
        },
      },
    ];

    const result = getFeedbackAssessorResult(results);
    expect(result).toBeDefined();
    expect(result?.id).toBe("AES-FEEDBACK");
    expect(result?.overall).toBe(3.87);
    expect(result?.cefr).toBe("B1");
  });

  it("should return undefined if not found", () => {
    const results: AssessorResult[] = [{ id: "AES-ESSAY", name: "Essay", type: "grader" }];
    expect(getFeedbackAssessorResult(results)).toBeUndefined();
  });

  it("should return undefined if overall is missing", () => {
    const results: AssessorResult[] = [
      { id: "AES-FEEDBACK", name: "Feedback", type: "grader", cefr: "B1" },
    ];
    expect(getFeedbackAssessorResult(results)).toBeUndefined();
  });

  it("should return undefined if cefr is missing", () => {
    const results: AssessorResult[] = [
      { id: "AES-FEEDBACK", name: "Feedback", type: "grader", overall: 3.87 },
    ];
    expect(getFeedbackAssessorResult(results)).toBeUndefined();
  });
});

describe("getLanguageToolAssessorResult", () => {
  it("should return LanguageTool assessor with errors", () => {
    const errors: LanguageToolError[] = [
      {
        start: 0,
        end: 5,
        length: 5,
        category: "GRAMMAR",
        rule_id: "test",
        message: "Test error",
        source: "LT",
        severity: "error",
      },
    ];

    const results: AssessorResult[] = [
      { id: "GEC-LT", name: "LanguageTool", type: "feedback", errors },
    ];

    const result = getLanguageToolAssessorResult(results);
    expect(result).toBeDefined();
    expect(result?.id).toBe("GEC-LT");
    expect(result?.errors).toEqual(errors);
  });

  it("should return undefined if not found", () => {
    const results: AssessorResult[] = [{ id: "AES-ESSAY", name: "Essay", type: "grader" }];
    expect(getLanguageToolAssessorResult(results)).toBeUndefined();
  });

  it("should return undefined if errors is missing", () => {
    const results: AssessorResult[] = [{ id: "GEC-LT", name: "LanguageTool", type: "feedback" }];
    expect(getLanguageToolAssessorResult(results)).toBeUndefined();
  });
});

describe("getLLMAssessorResult", () => {
  it("should return LLM assessor with errors", () => {
    const errors: LanguageToolError[] = [
      {
        start: 0,
        end: 5,
        length: 5,
        category: "GRAMMAR",
        rule_id: "test",
        message: "Test error",
        source: "LLM",
        severity: "error",
      },
    ];

    const results: AssessorResult[] = [{ id: "GEC-LLM", name: "LLM", type: "feedback", errors }];

    const result = getLLMAssessorResult(results);
    expect(result).toBeDefined();
    expect(result?.id).toBe("GEC-LLM");
    expect(result?.errors).toEqual(errors);
  });

  it("should return undefined if not found", () => {
    const results: AssessorResult[] = [{ id: "AES-ESSAY", name: "Essay", type: "grader" }];
    expect(getLLMAssessorResult(results)).toBeUndefined();
  });
});

describe("getTeacherFeedbackAssessorResult", () => {
  it("should return teacher feedback assessor with meta", () => {
    const results: AssessorResult[] = [
      {
        id: "TEACHER-FEEDBACK",
        name: "Teacher Feedback",
        type: "feedback",
        meta: {
          message: "Great work!",
          focusArea: "Grammar",
          cluesMessage: "Check verb tenses",
          explanationMessage: "You need to work on verb tenses",
        },
      },
    ];

    const result = getTeacherFeedbackAssessorResult(results);
    expect(result).toBeDefined();
    expect(result?.id).toBe("TEACHER-FEEDBACK");
    expect(result?.meta.message).toBe("Great work!");
  });

  it("should return undefined if not found", () => {
    const results: AssessorResult[] = [{ id: "AES-ESSAY", name: "Essay", type: "grader" }];
    expect(getTeacherFeedbackAssessorResult(results)).toBeUndefined();
  });

  it("should return undefined if meta.message is missing", () => {
    const results: AssessorResult[] = [
      {
        id: "TEACHER-FEEDBACK",
        name: "Teacher Feedback",
        type: "feedback",
        meta: {},
      },
    ];
    expect(getTeacherFeedbackAssessorResult(results)).toBeUndefined();
  });
});

describe("getRelevanceCheckAssessorResult", () => {
  it("should return relevance check assessor with meta", () => {
    const results: AssessorResult[] = [
      {
        id: "RELEVANCE-CHECK",
        name: "Relevance Check",
        type: "ard",
        meta: {
          addressesQuestion: true,
          similarityScore: 0.85,
          threshold: 0.5,
        },
      },
    ];

    const result = getRelevanceCheckAssessorResult(results);
    expect(result).toBeDefined();
    expect(result?.id).toBe("RELEVANCE-CHECK");
    expect(result?.meta.addressesQuestion).toBe(true);
    expect(result?.meta.similarityScore).toBe(0.85);
  });

  it("should return undefined if not found", () => {
    const results: AssessorResult[] = [{ id: "AES-ESSAY", name: "Essay", type: "grader" }];
    expect(getRelevanceCheckAssessorResult(results)).toBeUndefined();
  });

  it("should return undefined if meta fields are missing", () => {
    const results: AssessorResult[] = [
      {
        id: "RELEVANCE-CHECK",
        name: "Relevance Check",
        type: "ard",
        meta: {},
      },
    ];
    expect(getRelevanceCheckAssessorResult(results)).toBeUndefined();
  });
});
