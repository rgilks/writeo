/**
 * Unit tests for text utility functions
 */

import { describe, it, expect } from "vitest";
import { pluralize } from "../../apps/web/app/lib/utils/text-utils";

describe("pluralize", () => {
  it("should return singular for count of 1", () => {
    expect(pluralize(1, "cat")).toBe("cat");
    expect(pluralize(1, "dog")).toBe("dog");
    expect(pluralize(1, "child")).toBe("child");
  });

  it("should return plural for count of 0", () => {
    expect(pluralize(0, "cat")).toBe("cats");
    expect(pluralize(0, "dog")).toBe("dogs");
  });

  it("should return plural for count greater than 1", () => {
    expect(pluralize(2, "cat")).toBe("cats");
    expect(pluralize(5, "dog")).toBe("dogs");
    expect(pluralize(100, "item")).toBe("items");
  });

  it("should use default plural form (add 's') when not provided", () => {
    expect(pluralize(2, "cat")).toBe("cats");
    expect(pluralize(2, "dog")).toBe("dogs");
    expect(pluralize(2, "book")).toBe("books");
  });

  it("should use custom plural form when provided", () => {
    expect(pluralize(2, "child", "children")).toBe("children");
    expect(pluralize(2, "mouse", "mice")).toBe("mice");
    expect(pluralize(2, "person", "people")).toBe("people");
  });

  it("should return singular for count of 1 even with custom plural", () => {
    expect(pluralize(1, "child", "children")).toBe("child");
    expect(pluralize(1, "mouse", "mice")).toBe("mouse");
    expect(pluralize(1, "person", "people")).toBe("person");
  });

  it("should handle negative counts", () => {
    expect(pluralize(-1, "cat")).toBe("cats");
    expect(pluralize(-5, "dog")).toBe("dogs");
  });

  it("should handle large numbers", () => {
    expect(pluralize(1000, "item")).toBe("items");
    expect(pluralize(1000000, "user")).toBe("users");
  });

  it("should handle decimal numbers", () => {
    expect(pluralize(1.5, "cat")).toBe("cats");
    expect(pluralize(0.5, "dog")).toBe("dogs");
  });
});
