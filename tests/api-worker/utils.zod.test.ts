/**
 * Unit tests for Zod utility functions
 */

import { describe, it, expect } from "vitest";
import { z } from "zod";
import { uuidStringSchema, formatZodMessage } from "../../apps/api-worker/src/utils/zod";

describe("uuidStringSchema", () => {
  it("should validate correct UUID v4 format", () => {
    const schema = uuidStringSchema("test_id");
    const result = schema.safeParse("550e8400-e29b-41d4-a716-446655440000");
    expect(result.success).toBe(true);
    if (result.success) {
      expect(result.data).toBe("550e8400-e29b-41d4-a716-446655440000");
    }
  });

  it("should reject invalid UUID format", () => {
    const schema = uuidStringSchema("test_id");
    const result = schema.safeParse("not-a-uuid");
    expect(result.success).toBe(false);
    if (!result.success) {
      expect(result.error.issues[0].message).toContain("Invalid test_id format");
    }
  });

  it("should reject empty string", () => {
    const schema = uuidStringSchema("test_id");
    const result = schema.safeParse("");
    expect(result.success).toBe(false);
  });

  it("should reject non-string input", () => {
    const schema = uuidStringSchema("test_id");
    const result = schema.safeParse(123);
    expect(result.success).toBe(false);
  });

  it("should use field name in error message", () => {
    const schema = uuidStringSchema("submission_id");
    const result = schema.safeParse("invalid");
    expect(result.success).toBe(false);
    if (!result.success) {
      expect(result.error.issues[0].message).toContain("submission_id");
    }
  });

  it("should validate multiple different UUIDs", () => {
    const schema = uuidStringSchema("test_id");
    const uuids = [
      "123e4567-e89b-12d3-a456-426614174000",
      "00000000-0000-4000-8000-000000000000",
      "ffffffff-ffff-4fff-bfff-ffffffffffff",
    ];

    for (const uuid of uuids) {
      const result = schema.safeParse(uuid);
      expect(result.success).toBe(true);
    }
  });

  it("should be case-insensitive", () => {
    const schema = uuidStringSchema("test_id");
    const result = schema.safeParse("550E8400-E29B-41D4-A716-446655440000");
    expect(result.success).toBe(true);
  });
});

describe("formatZodMessage", () => {
  it("should extract first error message from ZodError", () => {
    const schema = z.string().min(5);
    const result = schema.safeParse("abc");
    expect(result.success).toBe(false);
    if (!result.success) {
      const message = formatZodMessage(result.error, "Validation failed");
      expect(message).toBeDefined();
      expect(typeof message).toBe("string");
      expect(message.length).toBeGreaterThan(0);
    }
  });

  it("should use fallback message when no issues found", () => {
    const error = new z.ZodError([]);
    const message = formatZodMessage(error, "Default error message");
    expect(message).toBe("Default error message");
  });

  it("should use fallback message when issue has no message", () => {
    const error = new z.ZodError([
      {
        code: "custom",
        path: [],
        message: undefined as any,
      },
    ]);
    const message = formatZodMessage(error, "Default error message");
    expect(message).toBe("Default error message");
  });

  it("should return first issue message when available", () => {
    const schema = z.object({
      name: z.string().min(3),
      email: z.string().email(),
    });
    const result = schema.safeParse({ name: "ab", email: "invalid" });
    expect(result.success).toBe(false);
    if (!result.success) {
      const message = formatZodMessage(result.error, "Validation failed");
      // Should return the first error message
      expect(message).toBeDefined();
      expect(message).not.toBe("Validation failed");
    }
  });

  it("should handle multiple issues and return first", () => {
    const schema = z.object({
      name: z.string().min(5),
      age: z.number().min(18),
    });
    const result = schema.safeParse({ name: "ab", age: 10 });
    expect(result.success).toBe(false);
    if (!result.success) {
      const message = formatZodMessage(result.error, "Validation failed");
      // Should return first error (name validation)
      expect(message).toBeDefined();
      expect(message).not.toBe("Validation failed");
    }
  });
});
