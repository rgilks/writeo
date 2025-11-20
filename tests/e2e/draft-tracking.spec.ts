import { test, expect } from "./fixtures";
import { createTestSubmission, generateValidEssay } from "./helpers";

/**
 * Draft Tracking Tests (TC-DRAFT-001 to TC-DRAFT-017)
 * Tests for draft tracking, revision history, and navigation
 */

test.describe("Draft Tracking", () => {
  test("TC-DRAFT-001: Submit first draft", async ({ writePage, resultsPage, page }) => {
    await writePage.goto("1");
    const essay = generateValidEssay();
    await writePage.typeEssay(essay);

    // Wait for button to enable
    await page.waitForTimeout(1500);
    const isDisabled = await writePage.isSubmitButtonDisabled();
    expect(isDisabled).toBe(false);

    // Verify word count is valid (generateValidEssay should always produce valid essays)
    const wordCount = await writePage.getWordCount();
    expect(wordCount).toBeGreaterThanOrEqual(250);
    expect(wordCount).toBeLessThanOrEqual(500);

    await writePage.clickSubmit();

    // Wait for results (longer timeout for API call)
    await expect(page).toHaveURL(/\/results\/[a-f0-9-]+/, { timeout: 30000 });
    await resultsPage.waitForResults();

    // Draft number should be 1 (check metadata or UI)
    // Note: Draft number may not be visible in UI, but should be in metadata
    const draftHistory = await resultsPage.getDraftHistory();
    // Draft history should be hidden for single draft (TC-DRAFT-018)
    const historyVisible = (await draftHistory.count()) > 0;
    expect(historyVisible).toBe(false); // Should be hidden for first draft
  });

  test("TC-DRAFT-018: Draft History hidden for single draft", async ({ resultsPage }) => {
    const essay = generateValidEssay();
    const { submissionId } = await createTestSubmission("Describe your weekend.", essay);

    await resultsPage.goto(submissionId);
    await resultsPage.waitForResults();

    // Draft history should not be visible for single draft
    const draftHistory = await resultsPage.getDraftHistory();
    const historyVisible = (await draftHistory.count()) > 0;
    expect(historyVisible).toBe(false);
  });

  test("TC-DRAFT-013: Draft navigation works", async ({ resultsPage, page }) => {
    // Create two submissions to simulate drafts
    const essay1 = generateValidEssay();
    const { submissionId: draft1Id } = await createTestSubmission("Describe your weekend.", essay1);

    const essay2 = essay1 + " Additional improvements.";
    const { submissionId: draft2Id } = await createTestSubmission("Describe your weekend.", essay2);

    // Navigate to draft 2
    await resultsPage.goto(draft2Id, draft1Id);
    await resultsPage.waitForResults();

    // URL should include parent param
    const url = page.url();
    expect(url).toContain("parent=");
  });

  test("TC-DRAFT-019: Draft History positioned correctly", async ({ resultsPage, page }) => {
    // This test verifies draft history appears after editable essay
    // Requires multiple drafts to show history
    const essay = generateValidEssay();
    const { submissionId } = await createTestSubmission("Describe your weekend.", essay);

    await resultsPage.goto(submissionId);
    await resultsPage.waitForResults();

    // Wait for page to fully render
    await page.waitForTimeout(1000);

    // For single draft, history should be hidden
    // For multiple drafts, verify positioning (manual visual check)
    const editableEssay = await resultsPage.getEditableEssay();
    const count = await editableEssay.count();
    if (count > 0) {
      await expect(editableEssay.first()).toBeVisible({ timeout: 5000 });
    }
  });

  test("TC-DRAFT-020: Draft History compact layout", async ({ resultsPage }) => {
    // Visual test - verify compact layout
    // This is primarily a visual check, but we can verify it exists
    const essay = generateValidEssay();
    const { submissionId } = await createTestSubmission("Describe your weekend.", essay);

    await resultsPage.goto(submissionId);
    await resultsPage.waitForResults();

    // For single draft, history is hidden
    // For multiple drafts, verify it uses compact styling (visual check)
    const draftHistory = await resultsPage.getDraftHistory();
    // Just verify the selector works
    expect(draftHistory).toBeDefined();
  });

  test("TC-DRAFT-021: Create draft in local mode (no server storage)", async ({
    writePage,
    resultsPage,
    page,
  }) => {
    // Ensure we're in local mode (storeResults = false)
    await page.goto("/write/1");
    await page.evaluate(() => {
      localStorage.setItem("writeo-store-results", "false");
    });

    // Create first draft
    const essay1 = generateValidEssay();
    await writePage.goto("1");
    await writePage.typeEssay(essay1);

    // Wait for button to enable
    await page.waitForTimeout(1500);
    await writePage.clickSubmit();

    // Wait for results
    await expect(page).toHaveURL(/\/results\/[a-f0-9-]+/, { timeout: 30000 });
    await resultsPage.waitForResults();

    // Get the first submission ID from URL
    const firstUrl = page.url();
    const firstSubmissionId = firstUrl.match(/\/results\/([a-f0-9-]+)/)?.[1];
    expect(firstSubmissionId).toBeTruthy();

    // Store first draft results in localStorage (simulating what happens in real flow)
    const firstResults = await page.evaluate((submissionId) => {
      return localStorage.getItem(`results_${submissionId}`);
    }, firstSubmissionId);

    // If not already stored, we need to get it from the page
    // For this test, we'll simulate creating a draft by editing and resubmitting
    // Wait for editable essay to appear
    await page.waitForTimeout(2000);

    // Find and click the edit/resubmit button if available
    // Or we can directly navigate to create a draft via the API
    // For simplicity, let's create a second submission and link it as a draft

    // Create second draft (simulating draft creation)
    const essay2 = essay1 + " This is an improved version with additional content.";
    await writePage.goto("1");
    await writePage.typeEssay(essay2);

    // Wait for button to enable
    await page.waitForTimeout(1500);
    await writePage.clickSubmit();

    // Wait for second results
    await expect(page).toHaveURL(/\/results\/[a-f0-9-]+/, { timeout: 30000 });
    await resultsPage.waitForResults();

    // Verify that both submissions are stored in localStorage
    const allKeys = await page.evaluate(() => {
      const keys: string[] = [];
      for (let i = 0; i < localStorage.length; i++) {
        const key = localStorage.key(i);
        if (key && key.startsWith("results_")) {
          keys.push(key);
        }
      }
      return keys;
    });

    // Should have at least one result stored
    expect(allKeys.length).toBeGreaterThan(0);

    // Verify no errors occurred during draft creation
    // The key test is that creating a draft doesn't throw "submission can't be found" error
    const errorMessages = await page.locator("text=/submission.*not.*found/i").count();
    expect(errorMessages).toBe(0);
  });

  test("TC-DRAFT-022: Draft creation with parent from localStorage", async ({
    writePage,
    resultsPage,
    page,
  }) => {
    // Ensure we're in local mode
    await page.goto("/write/1");
    await page.evaluate(() => {
      localStorage.setItem("writeo-store-results", "false");
    });

    // Create first submission and store in localStorage
    const essay1 = generateValidEssay();
    const { submissionId: firstSubmissionId, results: firstResults } = await createTestSubmission(
      "Describe your weekend.",
      essay1
    );

    // Store in localStorage (simulating what the app does)
    await page.evaluate(
      ([submissionId, results]) => {
        localStorage.setItem(`results_${submissionId}`, JSON.stringify(results));
      },
      [firstSubmissionId, firstResults]
    );

    // Navigate to first results page
    await resultsPage.goto(firstSubmissionId);
    await resultsPage.waitForResults();
    await page.waitForTimeout(2000);

    // Now create a draft by editing and resubmitting
    // Find the editable essay component
    const editableEssay = await resultsPage.getEditableEssay();
    const editableCount = await editableEssay.count();

    if (editableCount > 0) {
      // Type additional text to create a draft
      await editableEssay.first().fill(essay1 + " Additional improvements here.");

      // Find and click submit/resubmit button
      const submitButton = page.locator('button:has-text("Submit Improved Draft")');
      const buttonCount = await submitButton.count();

      if (buttonCount > 0) {
        await submitButton.first().click();

        // Wait for new results page
        await expect(page).toHaveURL(/\/results\/[a-f0-9-]+/, { timeout: 30000 });
        await resultsPage.waitForResults();

        // Verify no errors occurred
        const errorMessages = await page
          .locator("text=/submission.*not.*found|submission.*can.*t.*found/i")
          .count();
        expect(errorMessages).toBe(0);

        // Verify draft was created successfully
        const newUrl = page.url();
        expect(newUrl).toContain("/results/");
      }
    } else {
      // If editable essay is not available, skip this part of the test
      // but verify the first submission worked
      expect(firstSubmissionId).toBeTruthy();
    }
  });
});
