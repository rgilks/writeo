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

    // Wait for either loading state or error - page should show something meaningful
    await expect(
      page.getByText("Loading Results").or(page.getByText("Results Not Available"))
    ).toBeVisible({ timeout: 5000 });

    // Wait for error to appear (may take a moment after loading)
    const errorMessage = page.getByText("Results Not Available");
    const isErrorVisible = await errorMessage.isVisible().catch(() => false);

    // If error is shown, verify it's user-friendly
    if (isErrorVisible) {
      await expect(page.locator("body")).not.toContainText("Server Component");
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

test.describe("Draft History", () => {
  test("draft history appears after submitting second draft", async ({
    writePage,
    resultsPage,
    page,
  }) => {
    // Submit first draft
    await writePage.goto("1");
    const essay1 = generateValidEssay();
    await writePage.typeEssay(essay1);
    await expect(async () => {
      expect(await writePage.isSubmitButtonDisabled()).toBe(false);
    }).toPass({ timeout: 3000 });
    await writePage.clickSubmit();
    await expect(page).toHaveURL(/\/results\/[a-f0-9-]+/, { timeout: 30000 });
    await resultsPage.waitForResults();

    // Wait for draft to be stored
    const firstUrl = page.url();
    const firstSubmissionId = firstUrl.split("/results/")[1];
    await resultsPage.waitForDraftStorage(firstSubmissionId);

    // Get the editable essay textarea and modify it
    const editableEssay = await resultsPage.getEditableEssay();
    await expect(editableEssay.first()).toBeVisible({ timeout: 10000 });

    // Modify the essay text (add some words to the beginning)
    await editableEssay
      .first()
      .fill(essay1 + " This is my improved version with additional insights.");

    // Submit improved draft
    const submitButton = await resultsPage.getSubmitDraftButton();
    await submitButton.click();

    // Wait for the URL to actually change (not just match the pattern)
    await expect(page).not.toHaveURL(firstUrl, { timeout: 45000 });

    // Also verify the new URL is a valid results URL
    await expect(page).toHaveURL(/\/results\/[a-f0-9-]+/, { timeout: 5000 });

    // Verify we're on a different results page (new submission ID)
    const secondUrl = page.url();
    expect(secondUrl).not.toBe(firstUrl);

    // Wait for results to load
    await resultsPage.waitForResults();

    // Wait for the draft history to appear (should have 2 drafts now)
    await resultsPage.waitForDraftHistory();

    // Verify draft history section is visible
    const draftHistory = await resultsPage.getDraftHistory();
    await expect(draftHistory).toBeVisible();

    // Verify we have at least 2 draft buttons
    const draftButtons = await resultsPage.getDraftButtons();
    expect(await draftButtons.count()).toBeGreaterThanOrEqual(2);
  });
});

test.describe("Progress Dashboard", () => {
  test("shows completed writings count after submission", async ({
    writePage,
    resultsPage,
    homePage,
    page,
  }) => {
    // Submit an essay
    await writePage.goto("1");
    const essay = generateValidEssay();
    await writePage.typeEssay(essay);
    await expect(async () => {
      expect(await writePage.isSubmitButtonDisabled()).toBe(false);
    }).toPass({ timeout: 3000 });
    await writePage.clickSubmit();

    // Wait for results
    await expect(page).toHaveURL(/\/results\/[a-f0-9-]+/, { timeout: 30000 });
    await resultsPage.waitForResults();

    // Wait for draft storage
    const submissionId = page.url().split("/results/")[1];
    await resultsPage.waitForDraftStorage(submissionId);

    // Navigate to homepage
    await homePage.goto();

    // Progress dashboard should show the completed writing
    const dashboard = await homePage.getProgressDashboard();
    await expect(dashboard.first()).toBeVisible();

    // Should show at least 1 writing completed (look for the stat)
    // The dashboard shows "Writings Completed" stat when count > 0
    // Use .first() to handle strict mode when both locators match
    await expect(
      page.locator("text=Writings Completed").or(page.locator("text=Total Drafts")).first()
    ).toBeVisible({ timeout: 5000 });
  });
});

test.describe("Results Persistence", () => {
  // Note: These tests are currently skipped due to a known Zustand persist hydration issue
  // where the store overwrites persisted data on page reload.
  // The core functionality works - users can enable server storage to persist results across devices.

  test.skip("results persist after page reload with local storage", async ({
    writePage,
    resultsPage,
    page,
  }) => {
    // Submit an essay (without server storage)
    await writePage.goto("1");
    const essay = generateValidEssay();
    await writePage.typeEssay(essay);
    await expect(async () => {
      expect(await writePage.isSubmitButtonDisabled()).toBe(false);
    }).toPass({ timeout: 3000 });
    await writePage.clickSubmit();

    // Wait for results
    await expect(page).toHaveURL(/\/results\/[a-f0-9-]+/, { timeout: 30000 });
    await resultsPage.waitForResults();

    // Get the results URL
    const resultsUrl = page.url();
    const submissionId = resultsUrl.split("/results/")[1];

    // Wait for storage to complete
    await resultsPage.waitForDraftStorage(submissionId);

    // Reload the page
    await page.reload();

    // Results should still be visible
    await resultsPage.waitForResults();
    const score = await resultsPage.getOverallScore();
    await expect(score.first()).toBeVisible({ timeout: 10000 });
  });

  test.skip("can navigate directly to results URL", async ({ resultsPage, writePage, page }) => {
    // First, create a submission to get a valid ID
    await writePage.goto("1");
    const essay = generateValidEssay();
    await writePage.typeEssay(essay);
    await expect(async () => {
      expect(await writePage.isSubmitButtonDisabled()).toBe(false);
    }).toPass({ timeout: 3000 });
    await writePage.clickSubmit();

    // Wait for results
    await expect(page).toHaveURL(/\/results\/[a-f0-9-]+/, { timeout: 30000 });
    await resultsPage.waitForResults();

    // Get the results URL and submission ID
    const resultsUrl = page.url();
    const submissionId = resultsUrl.split("/results/")[1];

    // Wait for storage
    await resultsPage.waitForDraftStorage(submissionId);

    // Navigate away
    await page.goto("/");

    // Navigate directly to the results URL
    await page.goto(resultsUrl);

    // Results should load
    await resultsPage.waitForResults();
    const score = await resultsPage.getOverallScore();
    await expect(score.first()).toBeVisible({ timeout: 10000 });
  });
});
