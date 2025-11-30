import { test, expect } from "./fixtures";
import { generateValidEssay, waitForResultsNavigation } from "./helpers";

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

// Clear localStorage before each test for better isolation
test.beforeEach(async ({ page }) => {
  // Clear localStorage without navigating (non-blocking)
  // This avoids navigation conflicts when tests start from different pages
  try {
    // Try to clear storage directly if page context is available
    await page.evaluate(() => {
      localStorage.clear();
      sessionStorage.clear();
    });
  } catch {
    // If page context isn't ready, that's okay - storage will be cleared when test navigates
    // Don't navigate here as it can conflict with test setup
  }
});

test.describe("Homepage", () => {
  test("loads correctly with all task cards", async ({ homePage, page }) => {
    await homePage.goto();

    // Page loads with title
    await expect(page).toHaveTitle(/Writeo/, { timeout: 10000 });

    // Task cards are visible (at least 9: 8 tasks + custom)
    const taskCards = await homePage.getTaskCards();
    expect(await taskCards.count()).toBeGreaterThanOrEqual(9);

    // Progress dashboard visible
    const dashboard = await homePage.getProgressDashboard();
    await expect(dashboard.first()).toBeVisible();
  });

  test("navigation to write page works", async ({ homePage, page }) => {
    await homePage.goto();
    // Wait for task cards to be visible (ensures page is fully rendered)
    const taskCards = await homePage.getTaskCards();
    await taskCards.first().waitFor({ state: "visible", timeout: 15000 });
    // clickTask already waits for navigation, so we just verify the URL
    await homePage.clickTask("1");
    await expect(page).toHaveURL(/\/write\/1/);
  });

  test("custom question card navigates correctly", async ({ homePage, page }) => {
    await homePage.goto();
    // Wait for task cards to be visible (ensures page is fully rendered)
    const taskCards = await homePage.getTaskCards();
    await taskCards.first().waitFor({ state: "visible", timeout: 15000 });
    // clickTask already waits for navigation, so we just verify the URL
    await homePage.clickTask("custom");
    await expect(page).toHaveURL(/\/write\/custom/);
  });
});

test.describe("Essay Submission", () => {
  test("write page loads with question and textarea", async ({ writePage, page }) => {
    await writePage.goto("1");

    const questionText = await writePage.getQuestionText();
    await expect(questionText).toBeVisible();

    // Wait for textarea to be visible using data-testid
    const textarea = page.locator('[data-testid="answer-textarea"]');
    await textarea.waitFor({ state: "visible", timeout: 15000 });
    await expect(textarea).toBeVisible();
  });

  // Auto-save feature removed - content no longer persists after refresh
  // Users can still manually save drafts from the history page if needed
  test.skip("content persists after page refresh", async ({ writePage, page }) => {
    // Test that content persists after page refresh using Zustand persist
    await writePage.goto("1");
    // Wait for textarea to be ready
    const textarea = await writePage.getTextarea();
    await textarea.waitFor({ state: "visible", timeout: 15000 });

    // Wait for store to be hydrated before typing
    // This ensures auto-save will work properly
    await page.waitForFunction(
      () => {
        // Check if the page has loaded and React is ready
        // The store will hydrate automatically when the component mounts
        return document.readyState === "complete";
      },
      { timeout: 5000 },
    );

    // Type some content
    const testContent = "This is a test essay that should persist after refresh. ";
    const repeatedContent = testContent.repeat(10); // Make it long enough to trigger auto-save
    await writePage.typeEssay(repeatedContent);

    // Wait for auto-saved indicator to appear
    // Auto-save has a 2-second delay, so we need to wait at least that long plus some buffer
    // First verify the draft was saved to localStorage (more reliable than waiting for UI)
    await page.waitForFunction(
      () => {
        try {
          const stored = localStorage.getItem("writeo-draft-store");
          if (!stored) return false;
          const parsed = JSON.parse(stored);
          const state = parsed.state || parsed;
          const drafts = state.contentDrafts || [];
          return drafts.length > 0 && drafts[0].content && drafts[0].content.length > 0;
        } catch {
          return false;
        }
      },
      { timeout: 5000 },
    );

    // Then wait for the indicator to appear (should be quick once draft is saved)
    const autoSavedIndicator = page.locator('[data-testid="auto-saved-indicator"]');
    await autoSavedIndicator.waitFor({ state: "visible", timeout: 5000 });

    // Verify content is in the textarea before refresh
    const textareaBefore = await writePage.getTextarea();
    const contentBeforeRefresh = await textareaBefore.inputValue();
    expect(contentBeforeRefresh).toContain(testContent);

    // Wait for Zustand to persist - check localStorage directly instead of arbitrary timeout
    await page.waitForFunction(
      () => {
        const stored = localStorage.getItem("writeo-draft-store");
        return stored && stored.length > 0;
      },
      { timeout: 5000 },
    );

    // Verify content is saved to localStorage
    const storedContent = await page.evaluate(() => {
      const stored = localStorage.getItem("writeo-draft-store");
      if (!stored) return null;
      try {
        const parsed = JSON.parse(stored);
        // Zustand persist stores as { state: {...}, version: 0 }
        return parsed.state?.currentContent || parsed.currentContent || null;
      } catch {
        return null;
      }
    });
    expect(storedContent).toContain(testContent);

    // Refresh the page
    await page.reload();

    // Wait for page to load and textarea to be visible
    const textareaAfter = await writePage.getTextarea();
    await textareaAfter.waitFor({ state: "visible", timeout: 10000 });

    // Wait for store hydration - check if store has hydrated by waiting for content to appear
    await page.waitForFunction(
      () => {
        const textarea = document.querySelector(
          '[data-testid="answer-textarea"]',
        ) as HTMLTextAreaElement;
        return textarea && textarea.value.length > 0;
      },
      { timeout: 10000 },
    );

    // Verify content is restored after refresh
    const contentAfterRefresh = await textareaAfter.inputValue();
    expect(contentAfterRefresh.length).toBeGreaterThan(0);
    expect(contentAfterRefresh).toContain(testContent);
  });

  test.skip("submit button enables after valid essay", async ({ writePage, page }) => {
    await writePage.goto("1");
    const textarea = await writePage.getTextarea();
    await textarea.waitFor({ state: "visible", timeout: 15000 });

    // Initially disabled - wait for button to be in DOM and check state
    const submitButton = page.locator('[data-testid="submit-button"]');
    await submitButton.waitFor({ state: "visible", timeout: 15000 });
    expect(await writePage.isSubmitButtonDisabled()).toBe(true);

    // Type valid essay
    const essay = generateValidEssay();
    await writePage.typeEssay(essay);

    // Wait for word count to update and button to be enabled
    await page.waitForFunction(
      () => {
        const btn = document.querySelector('[data-testid="submit-button"]') as HTMLButtonElement;
        return btn && !btn.disabled;
      },
      { timeout: 5000 },
    );
  });

  test.skip("word count validation prevents short essays", async ({ writePage, page }) => {
    await writePage.goto("1");
    await writePage.typeEssay("This is too short.");

    // Wait for word count to appear and be less than 250
    await page.waitForFunction(
      () => {
        const wordCountElement = document.querySelector('[data-testid="word-count-value"]');
        if (!wordCountElement) return false;
        const text = wordCountElement.textContent || "";
        const match = text.match(/(\d+)/);
        if (!match) return false;
        const count = parseInt(match[1] || "0");
        return count > 0 && count < 250;
      },
      { timeout: 15000 },
    );

    // Verify word count is correct
    const wordCount = await writePage.getWordCount();
    expect(wordCount).toBeGreaterThan(0);
    expect(wordCount).toBeLessThan(250);

    // Try to submit - should stay on write page due to validation
    // Note: Button may or may not be disabled depending on React state timing,
    // but form submission should be prevented by validation
    const submitButton = page.locator('[data-testid="submit-button"]').first();
    const initialUrl = page.url();

    // Try clicking submit - if button is enabled, form validation should prevent navigation
    try {
      await submitButton.click({ timeout: 1000 });
    } catch (e) {
      // Button might be disabled, which is fine
    }

    // Wait briefly to see if navigation happens (validation should prevent it)
    await page.waitForTimeout(1000);

    // Should stay on write page (validation prevents submission)
    await expect(page).toHaveURL(/\/write\/1/);
    expect(page.url()).toBe(initialUrl);
  });

  test.skip("full submission flow works", async ({ writePage, resultsPage, page }) => {
    await writePage.goto("1");

    const essay = generateValidEssay();
    await writePage.typeEssay(essay);

    await expect(async () => {
      expect(await writePage.isSubmitButtonDisabled()).toBe(false);
    }).toPass({ timeout: 10000 });

    await writePage.clickSubmit();
    await waitForResultsNavigation(page);
    await resultsPage.waitForResults();

    // Results displayed
    const score = await resultsPage.getOverallScore();
    await expect(score.first()).toBeVisible({ timeout: 10000 });
  });
});

test.describe("Results Page Features", () => {
  test.skip("displays all score components", async ({ writePage, resultsPage, page }) => {
    // Submit an essay (uses mocked LLM - no cost)
    await writePage.goto("1");
    const essay = generateValidEssay();
    await writePage.typeEssay(essay);
    await expect(async () => {
      expect(await writePage.isSubmitButtonDisabled()).toBe(false);
    }).toPass({ timeout: 10000 });
    await writePage.clickSubmit();
    await waitForResultsNavigation(page);
    await resultsPage.waitForResults();

    // Overall score visible
    const overallScore = await resultsPage.getOverallScore();
    await expect(overallScore.first()).toBeVisible({ timeout: 10000 });

    // CEFR level displayed
    const cefrLevel = await resultsPage.getCEFRLevel();
    await expect(cefrLevel.first()).toBeVisible({ timeout: 10000 });

    // Dimension scores visible (TA, CC, Vocab, Grammar)
    const dimensions = await resultsPage.getDimensionScores();
    await expect(dimensions.TA.first()).toBeVisible({ timeout: 10000 });
    await expect(dimensions.CC.first()).toBeVisible({ timeout: 10000 });
    await expect(dimensions.Vocab.first()).toBeVisible({ timeout: 10000 });
    await expect(dimensions.Grammar.first()).toBeVisible({ timeout: 10000 });
  });

  test.skip("displays grammar errors section", async ({ writePage, resultsPage, page }) => {
    // Submit an essay (uses mocked LLM - no cost)
    await writePage.goto("1");
    const essay = generateValidEssay();
    await writePage.typeEssay(essay);
    await expect(async () => {
      expect(await writePage.isSubmitButtonDisabled()).toBe(false);
    }).toPass({ timeout: 10000 });
    await writePage.clickSubmit();
    await waitForResultsNavigation(page);
    await resultsPage.waitForResults();

    // Grammar errors section visible
    const grammarSection = await resultsPage.getGrammarErrorsSection();
    await expect(grammarSection.first()).toBeVisible({ timeout: 10000 });
  });

  test.skip("displays teacher feedback", async ({ writePage, resultsPage, page }) => {
    // Submit an essay (uses mocked LLM - no cost)
    await writePage.goto("1");
    const essay = generateValidEssay();
    await writePage.typeEssay(essay);
    await expect(async () => {
      expect(await writePage.isSubmitButtonDisabled()).toBe(false);
    }).toPass({ timeout: 10000 });
    await writePage.clickSubmit();
    await waitForResultsNavigation(page);
    await resultsPage.waitForResults();

    // Teacher feedback visible (may take a moment to load)
    const teacherFeedback = await resultsPage.getTeacherFeedback();
    await expect(teacherFeedback.first()).toBeVisible({ timeout: 15000 });
  });

  test.skip("heat map displays and errors are interactive", async ({
    writePage,
    resultsPage,
    page,
  }) => {
    // Submit an essay (uses mocked LLM - no cost)
    await writePage.goto("1");
    const essay = generateValidEssay();
    await writePage.typeEssay(essay);
    await expect(async () => {
      expect(await writePage.isSubmitButtonDisabled()).toBe(false);
    }).toPass({ timeout: 10000 });
    await writePage.clickSubmit();
    await waitForResultsNavigation(page);
    await resultsPage.waitForResults();

    // Wait for heat map section to appear
    const heatMapSection = page.locator('[data-testid="heat-map-section"]');
    await expect(heatMapSection.first()).toBeVisible({ timeout: 15000 });

    // Check if heat map content is visible (either reveal prompt or revealed text)
    // The heat map may show reveal prompt or already revealed text depending on state
    const heatMapContent = heatMapSection.locator("p, div").first();

    // Heat map section should have content
    await expect(heatMapContent).toBeVisible({ timeout: 10000 });
  });

  test.skip("editable essay section allows editing", async ({ writePage, resultsPage, page }) => {
    // Submit an essay (uses mocked LLM - no cost)
    await writePage.goto("1");
    const essay = generateValidEssay();
    await writePage.typeEssay(essay);
    await expect(async () => {
      expect(await writePage.isSubmitButtonDisabled()).toBe(false);
    }).toPass({ timeout: 10000 });
    await writePage.clickSubmit();
    await waitForResultsNavigation(page);
    await resultsPage.waitForResults();

    // Editable essay textarea should be visible
    const editableEssay = await resultsPage.getEditableEssay();
    await expect(editableEssay.first()).toBeVisible({ timeout: 15000 });

    // Should be able to edit the text
    const initialText = await editableEssay.first().inputValue();
    expect(initialText.length).toBeGreaterThan(0);
  });
});

test.describe("Results Page", () => {
  test("shows friendly error for invalid submission", async ({ resultsPage, page }) => {
    const invalidId = "00000000-0000-0000-0000-000000000000";
    await resultsPage.goto(invalidId);

    // Wait for either loading state or error - page should show something meaningful
    await expect(
      page.getByText("Loading Results").or(page.getByText("Results Not Available")),
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
    const questionTextarea = page.locator('[data-testid="custom-question-textarea"]');
    await expect(questionTextarea).toBeVisible();

    // Answer textarea visible
    const answerTextarea = await writePage.getTextarea();
    await expect(answerTextarea).toBeVisible();

    // Title shows "Custom Question"
    const title = page.locator('[data-testid="page-title"]');
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
    // Submit first draft (uses mocked LLM - no cost)
    await writePage.goto("1");
    const essay1 = generateValidEssay();
    await writePage.typeEssay(essay1);
    await expect(async () => {
      expect(await writePage.isSubmitButtonDisabled()).toBe(false);
    }).toPass({ timeout: 10000 });
    await writePage.clickSubmit();
    await waitForResultsNavigation(page);
    await resultsPage.waitForResults();

    // Wait for draft to be stored
    const firstUrl = page.url();
    const firstSubmissionId = firstUrl.split("/results/")[1];
    await resultsPage.waitForDraftStorage(firstSubmissionId);

    // Get the editable essay textarea and modify it
    const editableEssay = await resultsPage.getEditableEssay();
    await expect(editableEssay.first()).toBeVisible({ timeout: 15000 });

    // Modify the essay text (add some words to the beginning)
    await editableEssay
      .first()
      .fill(essay1 + " This is my improved version with additional insights.");

    // Submit improved draft (uses mocked LLM - no cost)
    const submitButton = await resultsPage.getSubmitDraftButton();
    await submitButton.click();

    // Wait for the URL to actually change (not just match the pattern)
    await expect(page).not.toHaveURL(firstUrl, { timeout: 60000 });

    // Also verify the new URL is a valid results URL
    await expect(page).toHaveURL(/\/results\/[a-f0-9-]+/, { timeout: 10000 });

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

  test("draft comparison table displays when multiple drafts exist", async ({
    writePage,
    resultsPage,
    page,
  }) => {
    // Submit first draft (uses mocked LLM - no cost)
    await writePage.goto("1");
    const essay1 = generateValidEssay();
    await writePage.typeEssay(essay1);
    await expect(async () => {
      expect(await writePage.isSubmitButtonDisabled()).toBe(false);
    }).toPass({ timeout: 10000 });
    await writePage.clickSubmit();
    await waitForResultsNavigation(page);
    await resultsPage.waitForResults();

    const firstSubmissionId = page.url().split("/results/")[1];
    await resultsPage.waitForDraftStorage(firstSubmissionId);

    // Submit second draft (uses mocked LLM - no cost)
    const editableEssay = await resultsPage.getEditableEssay();
    await editableEssay.first().fill(essay1 + " Improved version.");
    const submitButton = await resultsPage.getSubmitDraftButton();
    await submitButton.click();
    await waitForResultsNavigation(page);
    await resultsPage.waitForResults();

    // Wait for draft history
    await resultsPage.waitForDraftHistory();

    // Draft comparison table should be visible
    const comparisonTable = await resultsPage.getDraftComparisonTable();
    await expect(comparisonTable.first()).toBeVisible({ timeout: 10000 });
  });

  test("draft navigation switches between drafts", async ({ writePage, resultsPage, page }) => {
    // Submit first draft (uses mocked LLM - no cost)
    await writePage.goto("1");
    const essay1 = generateValidEssay();
    await writePage.typeEssay(essay1);
    await expect(async () => {
      expect(await writePage.isSubmitButtonDisabled()).toBe(false);
    }).toPass({ timeout: 10000 });
    await writePage.clickSubmit();
    await waitForResultsNavigation(page);
    await resultsPage.waitForResults();

    const firstUrl = page.url();
    const firstSubmissionId = firstUrl.split("/results/")[1];
    await resultsPage.waitForDraftStorage(firstSubmissionId);

    // Submit second draft (uses mocked LLM - no cost)
    const editableEssay = await resultsPage.getEditableEssay();
    await editableEssay.first().fill(essay1 + " Improved.");
    const submitButton = await resultsPage.getSubmitDraftButton();
    await submitButton.click();
    await waitForResultsNavigation(page);
    await resultsPage.waitForResults();

    // Wait for draft history
    await resultsPage.waitForDraftHistory();

    // Click on first draft button to navigate back
    await resultsPage.clickDraftButton(1);

    // Should navigate to first draft's results
    await expect(page).toHaveURL(new RegExp(`/results/${firstSubmissionId}`), { timeout: 15000 });
    await resultsPage.waitForResults();
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
    await waitForResultsNavigation(page);
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
      page.locator("text=Writings Completed").or(page.locator("text=Total Drafts")).first(),
    ).toBeVisible({ timeout: 5000 });
  });
});

test.describe("Results Persistence", () => {
  // Tests that results persist in localStorage after page reload
  // With simplified Zustand store, hydration should work correctly

  test("results persist after page reload with local storage", async ({
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
    await waitForResultsNavigation(page);
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

  test("can navigate directly to results URL", async ({ resultsPage, writePage, page }) => {
    // First, create a submission to get a valid ID
    await writePage.goto("1");
    const essay = generateValidEssay();
    await writePage.typeEssay(essay);
    await expect(async () => {
      expect(await writePage.isSubmitButtonDisabled()).toBe(false);
    }).toPass({ timeout: 3000 });

    await writePage.clickSubmit();
    await waitForResultsNavigation(page);
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

test.describe("History Page", () => {
  test("history page loads and displays empty state when no history", async ({
    page,
    homePage,
    historyPage,
  }) => {
    // Clear localStorage to ensure empty state (already done in beforeEach, but ensure it's clear)
    await page.goto("/", { waitUntil: "domcontentloaded", timeout: 15000 });
    await page.evaluate(() => localStorage.clear());

    // Navigate to history page
    await homePage.goto();
    await homePage.clickHistoryLink();
    await expect(page).toHaveURL("/history");

    // Check page loads with title
    const title = await historyPage.getTitle();
    await expect(title).toBeVisible();

    // Check empty state is displayed
    const emptyState = await historyPage.getEmptyState();
    await expect(emptyState).toBeVisible();
    await expect(emptyState.locator("text=No History Yet")).toBeVisible();
  });

  test("history page displays submissions after creating one", async ({
    page,
    writePage,
    resultsPage,
    homePage,
    historyPage,
  }) => {
    // Submit an essay
    await writePage.goto("1");
    const essay = generateValidEssay();
    await writePage.typeEssay(essay);
    await expect(async () => {
      expect(await writePage.isSubmitButtonDisabled()).toBe(false);
    }).toPass({ timeout: 3000 });

    await writePage.clickSubmit();
    await waitForResultsNavigation(page);
    await resultsPage.waitForResults();

    // Wait for draft storage
    const submissionId = page.url().split("/results/")[1];
    await resultsPage.waitForDraftStorage(submissionId);

    // Navigate to history page
    await homePage.goto();
    await homePage.clickHistoryLink();
    await expect(page).toHaveURL("/history");

    // Check page loads
    const title = await historyPage.getTitle();
    await expect(title).toBeVisible();

    // Check that history items container is visible (not empty state)
    const itemsContainer = await historyPage.getHistoryItemsContainer();
    await expect(itemsContainer).toBeVisible({ timeout: 5000 });

    // Check that at least one submission card is visible
    const submissionCards = await historyPage.getSubmissionCards();
    await expect(submissionCards.first()).toBeVisible({ timeout: 5000 });

    // Check that "View Results" button is present
    const viewResultsButtons = await historyPage.getViewResultsButtons();
    await expect(viewResultsButtons.first()).toBeVisible();
  });

  test("history page navigation works from header", async ({ page, homePage, historyPage }) => {
    // Navigate to home
    await homePage.goto();

    // Click history link in header
    await homePage.clickHistoryLink();
    await expect(page).toHaveURL("/history");

    // Verify history page loaded
    const title = await historyPage.getTitle();
    await expect(title).toBeVisible();
  });

  test("view results button navigates to results page", async ({
    page,
    writePage,
    resultsPage,
    homePage,
    historyPage,
  }) => {
    // Submit an essay
    await writePage.goto("1");
    const essay = generateValidEssay();
    await writePage.typeEssay(essay);
    await expect(async () => {
      expect(await writePage.isSubmitButtonDisabled()).toBe(false);
    }).toPass({ timeout: 3000 });
    await writePage.clickSubmit();

    // Wait for results and get submission ID
    await expect(page).toHaveURL(/\/results\/[a-f0-9-]+/, { timeout: 30000 });
    await resultsPage.waitForResults();
    const submissionId = page.url().split("/results/")[1];
    await resultsPage.waitForDraftStorage(submissionId);

    // Navigate to history
    await homePage.goto();
    await homePage.clickHistoryLink();
    await expect(page).toHaveURL("/history");

    // Wait for submission card to appear
    const submissionCards = await historyPage.getSubmissionCards();
    await expect(submissionCards.first()).toBeVisible({ timeout: 5000 });

    // Click "View Results" button
    await historyPage.clickViewResults(0);

    // Should navigate back to results page
    await expect(page).toHaveURL(new RegExp(`/results/${submissionId}`), { timeout: 10000 });
    await resultsPage.waitForResults();
  });
});
