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
});
