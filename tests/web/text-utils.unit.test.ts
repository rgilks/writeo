/**
 * Unit tests for text utility functions
 */

import { describe, it, expect } from "vitest";
import { pluralize } from "../../apps/web/app/lib/utils/text-utils";

describe("pluralize", () => {
  it.each([
    [1, "cat", undefined, "cat"],
    [1, "dog", undefined, "dog"],
    [1, "child", undefined, "child"],
    [1, "child", "children", "child"],
    [1, "mouse", "mice", "mouse"],
    [1, "person", "people", "person"],
  ])(
    "should return singular for count of 1: pluralize(%d, %s, %s) = %s",
    (count, singular, plural, expected) => {
      expect(pluralize(count, singular, plural)).toBe(expected);
    },
  );

  it.each([
    [0, "cat", undefined, "cats"],
    [0, "dog", undefined, "dogs"],
    [2, "cat", undefined, "cats"],
    [5, "dog", undefined, "dogs"],
    [100, "item", undefined, "items"],
    [1000, "item", undefined, "items"],
    [1000000, "user", undefined, "users"],
  ])(
    "should return plural for count != 1: pluralize(%d, %s, %s) = %s",
    (count, singular, plural, expected) => {
      expect(pluralize(count, singular, plural)).toBe(expected);
    },
  );

  it.each([
    [2, "child", "children", "children"],
    [2, "mouse", "mice", "mice"],
    [2, "person", "people", "people"],
    [5, "child", "children", "children"],
  ])(
    "should use custom plural form when provided: pluralize(%d, %s, %s) = %s",
    (count, singular, plural, expected) => {
      expect(pluralize(count, singular, plural)).toBe(expected);
    },
  );

  it.each([
    [-1, "cat", undefined, "cats"],
    [-5, "dog", undefined, "dogs"],
    [1.5, "cat", undefined, "cats"],
    [0.5, "dog", undefined, "dogs"],
  ])(
    "should handle edge cases: pluralize(%d, %s, %s) = %s",
    (count, singular, plural, expected) => {
      expect(pluralize(count, singular, plural)).toBe(expected);
    },
  );
});
