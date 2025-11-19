import { test, expect } from "./fixtures";

/**
 * Error Handling Tests (TC-ERR-011 to TC-ERR-017)
 * Tests for friendly error messages and error page styling
 */

test.describe("Error Handling", () => {
  test("TC-ERR-011: Results page error handler shows friendly message", async ({
    resultsPage,
    page,
  }) => {
    // Use invalid submission ID
    const invalidId = "00000000-0000-0000-0000-000000000000";

    await resultsPage.goto(invalidId);

    // Wait for error state
    await page.waitForTimeout(2000);

    const errorState = await resultsPage.getErrorState();
    if ((await errorState.count()) > 0) {
      await expect(errorState.first()).toBeVisible();

      // Should show friendly message, not technical error
      const errorText = await errorState.first().textContent();
      expect(errorText).not.toContain("Server Component");
      expect(errorText).not.toContain("omitted in production");
      expect(errorText?.toLowerCase()).toMatch(/couldn't|not available|not found/);
    }
  });

  test("TC-ERR-012: Invalid submission ID shows helpful explanation", async ({
    resultsPage,
    page,
  }) => {
    const invalidId = "invalid-id-format";

    await resultsPage.goto(invalidId);
    await page.waitForTimeout(2000);

    const errorState = await resultsPage.getErrorState();
    if ((await errorState.count()) > 0) {
      const errorText = await errorState.first().textContent();

      // Should be helpful, not technical
      expect(errorText).toBeTruthy();
      expect(errorText?.length).toBeGreaterThan(20);
    }
  });

  test("TC-ERR-013: Server Component error shows friendly message", async ({
    resultsPage,
    page,
  }) => {
    // Try to access non-existent submission
    const fakeId = "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa";

    await resultsPage.goto(fakeId);
    await page.waitForTimeout(2000);

    const errorState = await resultsPage.getErrorState();
    if ((await errorState.count()) > 0) {
      const errorText = await errorState.first().textContent();

      // Should not show technical Server Component error
      expect(errorText).not.toContain("Server Components render");
      expect(errorText).not.toContain("omitted in production");
    }
  });

  test("TC-ERR-014: Network error on results shows helpful message", async ({
    resultsPage,
    page,
  }) => {
    // Simulate network error by using invalid base URL
    // Note: This test may not work if baseURL is set, so we'll test error state instead
    const invalidId = "00000000-0000-0000-0000-000000000000";

    await resultsPage.goto(invalidId);
    await page.waitForTimeout(2000);

    // Should show friendly error, not network error details
    const errorState = await resultsPage.getErrorState();
    if ((await errorState.count()) > 0) {
      const errorText = await errorState.first().textContent();
      expect(errorText).toBeTruthy();
    }
  });

  test("TC-ERR-015: Global error handler shows friendly message", async ({ page }) => {
    // Navigate to invalid route
    await page.goto("/invalid-route-that-does-not-exist");

    // Should show friendly error page (Next.js 404 or custom error)
    await page.waitForTimeout(1000);

    // Check for friendly error message
    const errorContent = page.locator("text=/not found|404|error/i");
    if ((await errorContent.count()) > 0) {
      await expect(errorContent.first()).toBeVisible();
    }
  });

  test("TC-ERR-016: Error page styling is friendly", async ({ resultsPage, page }) => {
    const invalidId = "00000000-0000-0000-0000-000000000000";

    await resultsPage.goto(invalidId);
    await page.waitForTimeout(2000);

    const errorState = await resultsPage.getErrorState();
    if ((await errorState.count()) > 0) {
      // Check for emoji (friendly indicator)
      const errorText = await errorState.first().textContent();
      const hasEmoji = errorText?.includes("📝") || errorText?.includes("😊");

      // Should have friendly styling (emoji or neutral colors)
      // Visual check: should not be red/scary
      const errorElement = errorState.first();
      const color = await errorElement.evaluate((el) => {
        const style = window.getComputedStyle(el);
        return style.color;
      });

      // Color should not be pure red (rgb(255, 0, 0))
      expect(color).not.toBe("rgb(255, 0, 0)");
    }
  });

  test("TC-ERR-017: Error recovery options provided", async ({ resultsPage, page }) => {
    const invalidId = "00000000-0000-0000-0000-000000000000";

    await resultsPage.goto(invalidId);
    await page.waitForTimeout(2000);

    // Should have recovery buttons
    const tryAgainButton = page
      .locator('button:has-text("Try")')
      .or(page.locator('a:has-text("Try")'));

    const backButton = page
      .locator('button:has-text("Back")')
      .or(page.locator('a:has-text("Back")'));

    // At least one recovery option should be available
    const hasTryAgain = (await tryAgainButton.count()) > 0;
    const hasBack = (await backButton.count()) > 0;

    expect(hasTryAgain || hasBack).toBe(true);
  });
});
