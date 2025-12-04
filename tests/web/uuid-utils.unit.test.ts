/**
 * Unit tests for UUID utility functions
 */

import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { generateUUID } from "../../apps/web/app/lib/utils/uuid-utils";

describe("generateUUID", () => {
  const originalCrypto = global.crypto;
  const uuidRegex = /^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i;

  afterEach(() => {
    // Don't try to restore crypto - it's read-only in some environments
    vi.restoreAllMocks();
  });

  it("should generate valid UUID v4 format when crypto.randomUUID is available", () => {
    const mockUUID = "550e8400-e29b-41d4-a716-446655440000";
    if (!global.crypto) {
      global.crypto = {} as any;
    }
    global.crypto.randomUUID = vi.fn(() => mockUUID);

    const result = generateUUID();
    expect(result).toBe(mockUUID);
    expect(result).toMatch(uuidRegex);
  });

  it.each([
    [undefined, "crypto.randomUUID not available"],
    [{ randomUUID: undefined }, "crypto exists but randomUUID is undefined"],
  ])("should use fallback when %j: %s", (cryptoConfig, description) => {
    if (cryptoConfig === undefined) {
      // @ts-ignore
      delete global.crypto;
    } else {
      global.crypto = cryptoConfig as any;
    }

    const result = generateUUID();
    expect(result).toMatch(uuidRegex);
    expect(result.length).toBe(36); // UUID format: 8-4-4-4-12
  });

  it("should generate different UUIDs on each call (fallback)", () => {
    // @ts-ignore
    delete global.crypto;

    const uuid1 = generateUUID();
    const uuid2 = generateUUID();
    // Very unlikely to be the same (1 in 2^122 chance)
    expect(uuid1).not.toBe(uuid2);
  });

  it("should generate UUIDs with correct v4 format (fallback)", () => {
    // @ts-ignore
    delete global.crypto;

    for (let i = 0; i < 10; i++) {
      const uuid = generateUUID();
      expect(uuid).toMatch(uuidRegex);
      // Check version digit (13th character should be '4')
      expect(uuid[14]).toBe("4");
      // Check variant bits (17th character should be 8, 9, a, or b)
      expect(["8", "9", "a", "b"]).toContain(uuid[19].toLowerCase());
    }
  });

  it("should generate valid UUID structure (8-4-4-4-12 format)", () => {
    // @ts-ignore
    delete global.crypto;

    const uuid = generateUUID();
    const parts = uuid.split("-");
    expect(parts).toHaveLength(5);
    expect(parts[0].length).toBe(8);
    expect(parts[1].length).toBe(4);
    expect(parts[2].length).toBe(4);
    expect(parts[3].length).toBe(4);
    expect(parts[4].length).toBe(12);
  });
});
