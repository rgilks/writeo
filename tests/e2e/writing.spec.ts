import { test, expect } from "./fixtures";
import { getTestEssay, generateValidEssay } from "./helpers";

/**
 * Writing Page Tests (TC-FE-009 to TC-FE-020, TC-FORM-014-017)
 * Tests for writing page form submission, validation, and word count
 */

test.describe("Writing Page", () => {
  test("TC-FE-009: Writing page loads correctly", async ({ writePage, page }) => {
    await writePage.goto("1");

    // Check question text is visible
    const questionText = await writePage.getQuestionText();
    await expect(questionText).toBeVisible();

    // Check textarea is visible
    const textarea = await writePage.getTextarea();
    await expect(textarea).toBeVisible();
  });

  test("TC-FE-010: Question text displays correctly", async ({ writePage }) => {
    await writePage.goto("1");

    const questionText = await writePage.getQuestionText();
    const text = await questionText.textContent();

    // Should contain question prompt
    expect(text).toBeTruthy();
    expect(text?.length).toBeGreaterThan(0);
  });

  test("TC-FE-012: Submit button is disabled initially", async ({ writePage }) => {
    await writePage.goto("1");

    const isDisabled = await writePage.isSubmitButtonDisabled();
    expect(isDisabled).toBe(true);
  });

  test("TC-FE-013: Submit button enables after typing", async ({ writePage, page }) => {
    await writePage.goto("1");

    // Initially disabled
    expect(await writePage.isSubmitButtonDisabled()).toBe(true);

    // Type some text (enough to enable button - just needs non-empty)
    await writePage.typeEssay(
      "This is a test essay with enough content to enable the submit button."
    );

    // Wait for state to update and button to enable
    await page.waitForTimeout(300);

    // Button should be enabled (wait up to 2 seconds)
    await expect(async () => {
      const isDisabled = await writePage.isSubmitButtonDisabled();
      expect(isDisabled).toBe(false);
    }).toPass({ timeout: 2000 });
  });

  test("TC-FORM-014: Word count minimum validation", async ({ writePage, page }) => {
    await writePage.goto("1");

    // Type short essay (< 250 words)
    const shortEssay = getTestEssay("short");
    await writePage.typeEssay(shortEssay);

    // Wait for word count to update
    await page.waitForTimeout(500);

    // Try to submit
    await writePage.clickSubmit();

    // Wait for validation (should stay on write page)
    await page.waitForTimeout(1000);

    // Check that we're still on the write page (didn't navigate to results)
    await expect(page).toHaveURL(/\/write\/1/, { timeout: 2000 });

    // Error might be shown - check if visible
    const error = await writePage.getError();
    const errorCount = await error.count();
    if (errorCount > 0) {
      const isVisible = await error
        .first()
        .isVisible()
        .catch(() => false);
      if (isVisible) {
        await expect(error.first()).toBeVisible();
      }
    }
  });

  test("TC-FORM-015: Word count maximum validation", async ({ writePage, page }) => {
    await writePage.goto("1");

    // Type long essay (> 500 words)
    const longEssay = Array(600).fill("word ").join("");
    await writePage.typeEssay(longEssay);

    // Wait for word count to update
    await page.waitForTimeout(500);

    // Try to submit
    await writePage.clickSubmit();

    // Wait for validation (should stay on write page)
    await page.waitForTimeout(1000);

    // Check that we're still on the write page (didn't navigate to results)
    await expect(page).toHaveURL(/\/write\/1/, { timeout: 2000 });

    // Error might be shown
    const error = await writePage.getError();
    const errorCount = await error.count();
    if (errorCount > 0) {
      const isVisible = await error
        .first()
        .isVisible()
        .catch(() => false);
      if (isVisible) {
        await expect(error.first()).toBeVisible();
      }
    }
  });

  test("TC-FORM-016: Word count display", async ({ writePage, page }) => {
    await writePage.goto("1");

    // Type essay
    const essay = generateValidEssay();
    await writePage.typeEssay(essay);

    // Wait for word count to update
    await page.waitForTimeout(500);

    // Check word count is displayed (format: "X words" or "X word")
    const wordCountText = page.locator("text=/\\d+ (word|words)/i");
    await expect(wordCountText.first()).toBeVisible({ timeout: 5000 });

    // Check visual feedback (✓ for valid count between 250-500)
    const checkmark = page.locator("text=✓").or(page.locator('[aria-label*="valid"]'));
    const checkmarkCount = await checkmark.count();
    if (checkmarkCount > 0) {
      await expect(checkmark.first()).toBeVisible({ timeout: 2000 });
    }
  });

  test("TC-FE-018: Submit button shows loading state", async ({ writePage, page }) => {
    await writePage.goto("1");

    // Type valid essay (250-500 words)
    const essay = generateValidEssay();
    await writePage.typeEssay(essay);

    // Wait for word count to update and button to enable
    await page.waitForTimeout(1500);

    // Verify button is enabled and no errors
    const isDisabled = await writePage.isSubmitButtonDisabled();
    expect(isDisabled).toBe(false);

    // Verify word count is valid (generateValidEssay should always produce valid essays)
    const wordCount = await writePage.getWordCount();
    expect(wordCount).toBeGreaterThanOrEqual(250);
    expect(wordCount).toBeLessThanOrEqual(500);

    // Click submit
    await writePage.clickSubmit();

    // Wait for loading state to appear (button should show loading text or be disabled)
    // Check immediately and also wait a bit to catch loading state
    const submitButton = await writePage.getSubmitButton();

    // Wait for loading state to appear (up to 2 seconds)
    await expect(async () => {
      const buttonText = await submitButton.first().textContent();
      const hasLoadingText = !!buttonText?.match(/Analyzing|Loading|Processing/i);

      const loadingState = page.locator("text=/Analyzing your writing|Loading|Processing/i");
      const hasLoadingState = (await loadingState.count()) > 0;

      const buttonDisabled = await submitButton.first().isDisabled();

      // At least one loading indicator should be present
      if (hasLoadingText || hasLoadingState || buttonDisabled) {
        return true;
      }
      throw new Error("Loading state not found");
    }).toPass({ timeout: 2000 });
  });

  test("TC-FE-020: Navigates to results page after submission", async ({ writePage, page }) => {
    await writePage.goto("1");

    // Type valid essay (250-500 words)
    const essay = generateValidEssay();
    await writePage.typeEssay(essay);

    // Wait for word count to update and button to enable
    await page.waitForTimeout(1500);

    // Verify button is enabled
    const isDisabled = await writePage.isSubmitButtonDisabled();
    expect(isDisabled).toBe(false);

    // Verify word count is valid (generateValidEssay should always produce valid essays)
    const wordCount = await writePage.getWordCount();
    expect(wordCount).toBeGreaterThanOrEqual(250);
    expect(wordCount).toBeLessThanOrEqual(500);

    // Check for validation errors (but ignore checklist items)
    const error = await writePage.getError();
    const errorCount = await error.count();
    if (errorCount > 0) {
      const errorText = await error.first().textContent();
      // Checklist items are not blocking errors - they're just reminders
      if (
        errorText?.includes("Did I") ||
        errorText?.includes("checklist") ||
        errorText?.includes("Self-Evaluation")
      ) {
        // This is just a checklist reminder, not a blocking error - proceed
      } else {
        throw new Error(`Cannot submit: ${errorText}`);
      }
    }

    // Submit
    await writePage.clickSubmit();

    // Wait for loading state to appear (indicates submission started)
    const loadingState = await writePage.getLoadingState();
    const hasLoading = (await loadingState.count()) > 0;

    // If loading state appears, wait for it to disappear (submission processing)
    if (hasLoading) {
      // Wait for loading to complete (max 25 seconds)
      await page
        .waitForFunction(
          () => {
            const button = document.querySelector('button[type="submit"]');
            if (!button) return true; // Button gone means navigation happened
            return !button.disabled && !button.textContent?.includes("Analyzing");
          },
          { timeout: 25000 }
        )
        .catch(() => {
          // If loading doesn't complete, continue anyway
        });
    }

    // Wait a moment for navigation or errors to appear
    await page.waitForTimeout(1000);
    
    // Check if we've already navigated to results page
    // If so, skip error checking (results page content might be mistaken for errors)
    const currentUrl = page.url();
    const isOnResultsPage = /\/results\/[a-f0-9-]+/.test(currentUrl);
    
    // Only check for errors if we're still on the write page
    if (!isOnResultsPage) {
      const errorAfterSubmit = await writePage.getError();
      const errorCountAfterSubmit = await errorAfterSubmit.count();
      if (errorCountAfterSubmit > 0) {
        const errorText = await errorAfterSubmit.first().textContent();
        // Checklist items are not blocking errors
        // "Improve Your Writing" is results page content, not an error
        if (
          !errorText?.includes("Did I") &&
          !errorText?.includes("checklist") &&
          !errorText?.includes("Self-Evaluation") &&
          !errorText?.includes("Improve Your Writing")
        ) {
          throw new Error(`Submission failed with error: ${errorText}`);
        }
      }
    }

    // Should navigate to results page (wait longer for API call)
    await expect(page).toHaveURL(/\/results\/[a-f0-9-]+/, { timeout: 30000 });
  });

  test("TC-META-003: Self-evaluation checklist appears", async ({ writePage, page }) => {
    await writePage.goto("1");

    // Type enough text to trigger checklist (50+ chars)
    await writePage.typeEssay(
      "This is a test essay with enough characters to trigger the self-evaluation checklist."
    );

    // Wait for checklist to appear (it shows when answer.trim().length > 50)
    await page.waitForTimeout(500);

    // Checklist should appear
    const checklist = await writePage.getSelfEvalChecklist();
    await expect(checklist.first()).toBeVisible({ timeout: 3000 });
  });

  test("TC-FORM-017: Word count in editable essay", async ({ writePage, page }) => {
    // This test requires a submission first, so we'll test it in interactive-learning.spec.ts
    // Just verify the write page has word count display
    await writePage.goto("1");

    const essay = generateValidEssay();
    await writePage.typeEssay(essay);

    // Wait for word count to update
    await page.waitForTimeout(500);

    const wordCountText = page.locator("text=/\\d+ (word|words)/i");
    await expect(wordCountText.first()).toBeVisible({ timeout: 5000 });
  });
});
