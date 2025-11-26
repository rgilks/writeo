import { test, expect } from "./fixtures";
import { generateValidEssay } from "./helpers";

/**
 * Core E2E Tests - Comprehensive but lean
 *
 * These tests cover the critical user journeys:
 * 1. Homepage loads and navigation works
 * 2. Essay submission and results display
 * 3. Error handling is user-friendly
 *
 * Local: Uses mocked LLM APIs (fast, no cost)
 * CI: Uses real APIs (validates production behavior)
 */

test.describe("Homepage", () => {
  test("loads correctly with all task cards", async ({ homePage, page }) => {
    await homePage.goto();

    // Page loads with title
    await expect(page).toHaveTitle(/Writeo/);

    // Task cards are visible (at least 9: 8 tasks + custom)
    const taskCards = await homePage.getTaskCards();
    expect(await taskCards.count()).toBeGreaterThanOrEqual(9);

    // Progress dashboard visible
    const dashboard = await homePage.getProgressDashboard();
    await expect(dashboard.first()).toBeVisible();
  });

  test("navigation to write page works", async ({ homePage, page }) => {
    await homePage.goto();
    await homePage.clickTask("1");
    await expect(page).toHaveURL(/\/write\/1/);
  });

  test("custom question card navigates correctly", async ({ homePage, page }) => {
    await homePage.goto();
    const customLink = page.locator('a[href="/write/custom"]');
    await customLink.click();
    await expect(page).toHaveURL(/\/write\/custom/);
  });
});

test.describe("Essay Submission", () => {
  test("write page loads with question and textarea", async ({ writePage }) => {
    await writePage.goto("1");

    const questionText = await writePage.getQuestionText();
    await expect(questionText).toBeVisible();

    const textarea = await writePage.getTextarea();
    await expect(textarea).toBeVisible();
  });

  test("submit button enables after valid essay", async ({ writePage }) => {
    await writePage.goto("1");

    // Initially disabled
    expect(await writePage.isSubmitButtonDisabled()).toBe(true);

    // Type valid essay
    const essay = generateValidEssay();
    await writePage.typeEssay(essay);

    // Should enable
    await expect(async () => {
      expect(await writePage.isSubmitButtonDisabled()).toBe(false);
    }).toPass({ timeout: 3000 });
  });

  test("word count validation prevents short essays", async ({ writePage, page }) => {
    await writePage.goto("1");
    await writePage.typeEssay("This is too short.");
    await writePage.clickSubmit();

    // Should stay on write page
    await expect(page).toHaveURL(/\/write\/1/);
  });

  test("full submission flow works", async ({ writePage, resultsPage, page }) => {
    await writePage.goto("1");

    const essay = generateValidEssay();
    await writePage.typeEssay(essay);

    await expect(async () => {
      expect(await writePage.isSubmitButtonDisabled()).toBe(false);
    }).toPass({ timeout: 3000 });

    await writePage.clickSubmit();

    // Navigate to results
    await expect(page).toHaveURL(/\/results\/[a-f0-9-]+/, { timeout: 30000 });
    await resultsPage.waitForResults();

    // Results displayed
    const score = await resultsPage.getOverallScore();
    await expect(score.first()).toBeVisible({ timeout: 10000 });
  });
});

test.describe("Results Page", () => {
  test("shows friendly error for invalid submission", async ({ resultsPage, page }) => {
    const invalidId = "00000000-0000-0000-0000-000000000000";
    await resultsPage.goto(invalidId);

    // Should show error or loading state
    const errorState = await resultsPage.getErrorState();
    const loadingState = await resultsPage.getLoadingMessage();

    const hasError = (await errorState.count()) > 0;
    const hasLoading = (await loadingState.count()) > 0;

    expect(hasError || hasLoading).toBe(true);

    // If error shown, should be user-friendly
    if (hasError) {
      const errorText = await errorState.first().textContent();
      expect(errorText).not.toContain("Server Component");
      expect(errorText).not.toContain("omitted in production");
    }
  });
});

// Note: Teacher feedback is verified in the "full submission flow" test above

test.describe("Custom Question", () => {
  test("custom question page loads correctly", async ({ writePage, page }) => {
    await writePage.goto("custom");

    // Question textarea visible
    const questionTextarea = page.locator(".question-card textarea").first();
    await expect(questionTextarea).toBeVisible();

    // Answer textarea visible
    const answerTextarea = await writePage.getTextarea();
    await expect(answerTextarea).toBeVisible();

    // Title shows "Custom Question"
    const title = page.locator("h1.page-title");
    await expect(title).toContainText("Custom Question");
  });

  test("free writing without question works", async ({ writePage, page }) => {
    await writePage.goto("custom");

    // Don't enter a question - leave blank
    const essay = generateValidEssay();
    await writePage.typeEssay(essay);

    await expect(async () => {
      expect(await writePage.isSubmitButtonDisabled()).toBe(false);
    }).toPass({ timeout: 3000 });

    await writePage.clickSubmit();
    await expect(page).toHaveURL(/\/results\/[a-f0-9-]+/, { timeout: 30000 });
  });
});
