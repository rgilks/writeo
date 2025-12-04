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

  it.each([[undefined], [""]])("should return null for invalid input: %j", (input) => {
    expect(getGrammarRule(input as any)).toBeNull();
  });

  it.each([["Subject-verb agreement"], ["subject-verb agreement"], ["SUBJECT-VERB AGREEMENT"]])(
    "should be case-insensitive: %s",
    (input) => {
      const result = getGrammarRule(input);
      expect(result).not.toBeNull();
      expect(result?.rule).toBeDefined();
    },
  );

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

  it.each([
    ["Verb tense", "consistent verb tenses"],
    ["Article use", "a' before consonant"],
    ["Preposition use", "relationships"],
    ["Spelling", "spelling"],
    ["Punctuation", "Punctuation marks"],
    ["Word order", "Subject-Verb-Object"],
  ])("should return correct rule for %s", (ruleType, expectedContain) => {
    const result = getGrammarRule(ruleType);
    expect(result).not.toBeNull();
    expect(result?.rule).toContain(expectedContain);
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

  it.each([
    ["Subject-verb agreement"],
    ["Verb tense"],
    ["Article use"],
    ["Spelling"],
    ["Punctuation"],
  ])("should include common rule type: %s", (ruleType) => {
    const types = getAvailableGrammarRuleTypes();
    expect(types).toContain(ruleType);
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
