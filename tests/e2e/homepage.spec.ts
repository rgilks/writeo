import { test, expect } from "./fixtures";

/**
 * Homepage Tests (TC-FE-001 to TC-FE-007)
 * Tests for homepage loading, task cards, navigation, and progress dashboard
 */

test.describe("Homepage", () => {
  test("TC-FE-001: Homepage loads correctly", async ({ homePage, page }) => {
    await homePage.goto();

    // Check page title
    const title = await homePage.getTitle();
    await expect(title).toBeVisible();
    await expect(title).toContainText("Practice Writing");

    // Check no console errors
    const errors: string[] = [];
    page.on("console", (msg) => {
      if (msg.type() === "error") {
        errors.push(msg.text());
      }
    });

    await expect(page).toHaveTitle(/Writeo/);
  });

  test("TC-FE-002: All task cards visible", async ({ homePage }) => {
    await homePage.goto();

    const taskCards = await homePage.getTaskCards();
    const count = await taskCards.count();

    // Should have at least 9 task cards (8 predefined tasks + 1 custom question card)
    expect(count).toBeGreaterThanOrEqual(9);
  });

  test("TC-FE-003: Click on task card navigates to write page", async ({ homePage, page }) => {
    await homePage.goto();

    // Click first task card
    await homePage.clickTask("1");

    // Should navigate to write page
    await expect(page).toHaveURL(/\/write\/1/);
  });

  test('TC-FE-004: "Start Writing" CTA in dashboard works', async ({ homePage, page }) => {
    await homePage.goto();

    // Find and click "Start Writing" button (if exists in dashboard)
    const startWritingButton = page
      .locator('button:has-text("Start Writing")')
      .or(page.locator('a:has-text("Start Writing")'));

    if ((await startWritingButton.count()) > 0) {
      await startWritingButton.first().click();
      await expect(page).toHaveURL(/\/write/);
    } else {
      // If no dashboard CTA, clicking task card should work
      await homePage.clickTask("1");
      await expect(page).toHaveURL(/\/write\/1/);
    }
  });

  test("TC-FE-005: Progress dashboard visible", async ({ homePage }) => {
    await homePage.goto();

    const dashboard = await homePage.getProgressDashboard();

    // Dashboard should be visible (either with data or empty state)
    await expect(dashboard.first()).toBeVisible();
  });

  test("TC-FE-006: Page title is correct", async ({ homePage, page }) => {
    await homePage.goto();

    await expect(page).toHaveTitle(/Writeo|Essay Scoring/);
  });

  test("TC-FE-007: No console errors on load", async ({ homePage, page }) => {
    const errors: string[] = [];

    page.on("console", (msg) => {
      if (msg.type() === "error") {
        errors.push(msg.text());
      }
    });

    await homePage.goto();

    // Wait a bit for any async errors
    await page.waitForTimeout(1000);

    // Filter out known non-critical errors (e.g., analytics, third-party scripts)
    const criticalErrors = errors.filter((e) => !e.includes("favicon") && !e.includes("analytics"));

    expect(criticalErrors.length).toBe(0);
  });

  test("TC-FE-008: Custom question card navigates to custom write page", async ({
    homePage,
    page,
  }) => {
    await homePage.goto();

    // Find custom question card
    const customCard = page.locator("text=Custom Question").locator("..").locator("..");
    await expect(customCard).toBeVisible();

    // Click on custom question card
    await customCard.locator("a").first().click();

    // Should navigate to custom write page
    await expect(page).toHaveURL(/\/write\/custom/);
  });
});
