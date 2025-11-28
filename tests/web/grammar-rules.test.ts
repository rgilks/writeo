/**
 * Unit tests for grammar rules utility
 */

import { describe, it, expect } from "vitest";
import {
  getGrammarRule,
  getAvailableGrammarRuleTypes,
  type GrammarRule,
} from "../../apps/web/app/lib/utils/grammar-rules";

describe("getGrammarRule", () => {
  it("should return rule for exact match", () => {
    const result = getGrammarRule("Subject-verb agreement");
    expect(result).not.toBeNull();
    expect(result?.why).toBeDefined();
    expect(result?.rule).toBeDefined();
    expect(result?.examples).toBeDefined();
    expect(Array.isArray(result?.examples)).toBe(true);
  });

  it("should return null for undefined input", () => {
    expect(getGrammarRule(undefined)).toBeNull();
  });

  it("should return null for empty string", () => {
    expect(getGrammarRule("")).toBeNull();
  });

  it("should be case-insensitive", () => {
    const result1 = getGrammarRule("Subject-verb agreement");
    const result2 = getGrammarRule("subject-verb agreement");
    const result3 = getGrammarRule("SUBJECT-VERB AGREEMENT");

    expect(result1).not.toBeNull();
    expect(result2).toEqual(result1);
    expect(result3).toEqual(result1);
  });

  it("should match partial patterns", () => {
    const result = getGrammarRule("subject verb agreement");
    expect(result).not.toBeNull();
    expect(result?.rule).toContain("subject");
  });

  it("should return Grammar rule as fallback for unknown types", () => {
    const result = getGrammarRule("Unknown Error Type");
    expect(result).not.toBeNull();
    // Should return the default Grammar rule
    expect(result?.rule).toBeDefined();
  });

  it("should return correct rule for Verb tense", () => {
    const result = getGrammarRule("Verb tense");
    expect(result).not.toBeNull();
    expect(result?.rule).toContain("consistent verb tenses");
  });

  it("should return correct rule for Article use", () => {
    const result = getGrammarRule("Article use");
    expect(result).not.toBeNull();
    expect(result?.rule).toContain("a' before consonant");
  });

  it("should return correct rule for Preposition use", () => {
    const result = getGrammarRule("Preposition use");
    expect(result).not.toBeNull();
    expect(result?.rule).toContain("relationships");
  });

  it("should return correct rule for Spelling", () => {
    const result = getGrammarRule("Spelling");
    expect(result).not.toBeNull();
    expect(result?.rule).toContain("spelling");
  });

  it("should return correct rule for Punctuation", () => {
    const result = getGrammarRule("Punctuation");
    expect(result).not.toBeNull();
    expect(result?.rule).toContain("Punctuation marks");
  });

  it("should return correct rule for Word order", () => {
    const result = getGrammarRule("Word order");
    expect(result).not.toBeNull();
    expect(result?.rule).toContain("Subject-Verb-Object");
  });

  it("should return rule with examples array", () => {
    const result = getGrammarRule("Subject-verb agreement");
    expect(result).not.toBeNull();
    expect(Array.isArray(result?.examples)).toBe(true);
    expect(result?.examples.length).toBeGreaterThan(0);
    expect(typeof result?.examples[0]).toBe("string");
  });

  it("should return rule with why and rule properties", () => {
    const result = getGrammarRule("Verb tense");
    expect(result).not.toBeNull();
    expect(result?.why).toBeDefined();
    expect(result?.rule).toBeDefined();
    expect(typeof result?.why).toBe("string");
    expect(typeof result?.rule).toBe("string");
  });
});

describe("getAvailableGrammarRuleTypes", () => {
  it("should return array of rule type strings", () => {
    const types = getAvailableGrammarRuleTypes();
    expect(Array.isArray(types)).toBe(true);
    expect(types.length).toBeGreaterThan(0);
  });

  it("should return strings for all rule types", () => {
    const types = getAvailableGrammarRuleTypes();
    types.forEach((type) => {
      expect(typeof type).toBe("string");
      expect(type.length).toBeGreaterThan(0);
    });
  });

  it("should include common rule types", () => {
    const types = getAvailableGrammarRuleTypes();
    expect(types).toContain("Subject-verb agreement");
    expect(types).toContain("Verb tense");
    expect(types).toContain("Article use");
    expect(types).toContain("Spelling");
    expect(types).toContain("Punctuation");
  });

  it("should return consistent results", () => {
    const types1 = getAvailableGrammarRuleTypes();
    const types2 = getAvailableGrammarRuleTypes();
    expect(types1).toEqual(types2);
  });

  it("should allow getting rules for all returned types", () => {
    const types = getAvailableGrammarRuleTypes();
    types.forEach((type) => {
      const rule = getGrammarRule(type);
      expect(rule).not.toBeNull();
      expect(rule?.why).toBeDefined();
      expect(rule?.rule).toBeDefined();
    });
  });
});
