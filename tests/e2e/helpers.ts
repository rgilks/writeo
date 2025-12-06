import { Page, expect as playwrightExpect } from "@playwright/test";
import { randomUUID } from "crypto";
// Note: Environment variables are already loaded in playwright.config.ts
// No need to load them again here to avoid duplicate messages

// Common error selectors for checking page errors
const ERROR_SELECTORS = [
  '[role="alert"]',
  '[data-testid="error"]',
  '[data-testid="error-message"]',
  "text=/error/i",
  "text=/rate limit/i",
  "text=/too many/i",
  "text=/failed/i",
] as const;

// Helper to format console errors for readability
function formatConsoleError(err: string): string {
  if (err.includes("Failed to fetch")) {
    return "Failed to fetch (network error - check if API server is running)";
  }
  if (err.includes("NEXT_CONSOLE_ERROR")) {
    // Extract the actual error message from Next.js error object
    try {
      const match = err.match(/message:\s*([^,}]+)/);
      if (match) return `Console: ${match[1].trim()}`;
    } catch {
      // Fall through to return full error
    }
  }
  // Truncate very long errors
  return err.length > 200 ? `${err.substring(0, 200)}...` : err;
}

// Helper to set up error monitoring for a page
function setupErrorMonitoring(page: Page) {
  const consoleErrors: string[] = [];
  const networkErrors: string[] = [];
  const failedRequests: Array<{ url: string; status: number; statusText: string }> = [];

  const consoleHandler = (msg: any) => {
    if (msg.type() === "error") {
      consoleErrors.push(msg.text());
    }
  };

  const responseHandler = (response: any) => {
    const status = response.status();
    const url = response.url();

    if (status >= 400 && url.includes("/submissions")) {
      failedRequests.push({
        url,
        status,
        statusText: response.statusText(),
      });
      networkErrors.push(`${status} ${response.statusText()}: ${url}`);
    } else if (status === 429) {
      networkErrors.push(`429 Rate Limit: ${url}`);
    }
  };

  const requestFailedHandler = (request: any) => {
    const url = request.url();
    if (url.includes("/submissions")) {
      networkErrors.push(`Request failed: ${url} (${request.failure()?.errorText || "unknown"})`);
    }
  };

  page.on("console", consoleHandler);
  page.on("response", responseHandler);
  page.on("requestfailed", requestFailedHandler);

  return {
    consoleErrors,
    networkErrors,
    failedRequests,
    cleanup: () => {
      page.off("console", consoleHandler);
      page.off("response", responseHandler);
      page.off("requestfailed", requestFailedHandler);
    },
  };
}

// Helper to check for visible error messages on page
async function checkForPageErrors(
  page: Page,
  selectors: readonly string[] = ERROR_SELECTORS,
): Promise<string | null> {
  for (const selector of selectors) {
    try {
      const errorEl = page.locator(selector).first();
      const isVisible = await errorEl.isVisible({ timeout: 1000 }).catch(() => false);
      if (isVisible) {
        const errorText = await errorEl.textContent();
        if (errorText && errorText.trim()) {
          return errorText.trim();
        }
      }
    } catch {
      // Continue checking other selectors
    }
  }
  return null;
}

// Test data - standard test essays for testing
export const TEST_ESSAYS = {
  short:
    "Last weekend I went to the park. I played with my dog. We had fun together. It was a nice day.",
  withErrors:
    "I goes to park yesterday. The dog was happy and we plays together. He are very nice. I has a good time.",
  corrected:
    "I went to the park yesterday. The dog was happy and we played together. He was very nice. I had a good time.",
  long: Array(50)
    .fill(
      "This is a test sentence with multiple words to create a longer essay for testing purposes. ",
    )
    .join(""),
  standard:
    Array(30).fill("This is a test sentence. ").join("") +
    "I goes to park yesterday. The dog was happy and we plays together.",
};

const MIN_ESSAY_WORDS = 250;

// API configuration - loaded from .env or .env.local
// Always prefer TEST_API_KEY for tests (higher rate limits)
const API_BASE = process.env.API_BASE || process.env.API_BASE_URL || "http://localhost:8787";
const API_KEY = process.env.TEST_API_KEY || process.env.API_KEY || "";

// Track last submission time per worker to add delays between parallel test submissions
// Use worker ID to avoid conflicts between workers
const workerId = process.env.TEST_WORKER_INDEX || "0";
const submissionTimers = new Map<string, number>();
// Minimal delay - only needed if not using mocks (mocks are instant)
// With mocks enabled, we can reduce delays significantly
const useMockServices = process.env.USE_MOCK_SERVICES === "true";
const MIN_DELAY_MS = useMockServices
  ? 10 // Minimal delay with mocks - just to avoid any potential race conditions
  : process.env.CI
    ? 500 // Longer delay in CI without mocks
    : 300; // Standard delay without mocks

/**
 * Create a test submission via API with retry logic for rate limiting
 * Returns submission ID and results
 */
export async function createTestSubmission(
  questionText: string,
  answerText: string,
  retries = 3,
): Promise<{ submissionId: string; results: any }> {
  if (!API_KEY) {
    throw new Error("TEST_API_KEY or API_KEY environment variable required for E2E tests");
  }

  // Add delay per worker to prevent hitting rate limits when tests run in parallel
  const now = Date.now();
  const lastTime = submissionTimers.get(workerId) || 0;
  const timeSinceLastSubmission = now - lastTime;
  if (timeSinceLastSubmission < MIN_DELAY_MS) {
    await new Promise((resolve) => setTimeout(resolve, MIN_DELAY_MS - timeSinceLastSubmission));
  }
  submissionTimers.set(workerId, Date.now());

  const submissionId = randomUUID();
  const questionId = randomUUID();
  const answerId = randomUUID();

  for (let attempt = 0; attempt <= retries; attempt++) {
    // Exponential backoff: wait longer on each retry
    if (attempt > 0) {
      // Add jitter to avoid thundering herd
      const baseBackoff = Math.min(1000 * Math.pow(2, attempt - 1), 10000);
      const jitter = Math.random() * 1000; // Random 0-1s
      const backoffMs = baseBackoff + jitter;
      await new Promise((resolve) => setTimeout(resolve, backoffMs));
    }

    const response = await fetch(`${API_BASE}/v1/text/submissions`, {
      method: "POST",
      headers: {
        Authorization: `Token ${API_KEY}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        submissionId,
        submission: [
          {
            part: 1,
            answers: [
              {
                id: answerId,
                questionId,
                ...(questionText?.trim()
                  ? { questionText: questionText.trim() }
                  : { questionText: null }),
                text: answerText,
              },
            ],
          },
        ],
        template: { name: "generic", version: 1 },
        storeResults: true, // Store results on server for test retrieval
      }),
    });

    if (response.ok) {
      const results = await response.json();
      return { submissionId, results };
    }

    // Handle rate limiting (429) with retry
    if (response.status === 429) {
      const errorText = await response.text();
      let errorJson: { error?: string } = { error: errorText };
      try {
        errorJson = JSON.parse(errorText);
      } catch {
        // If not JSON, use the text as error message
        errorJson = { error: errorText };
      }

      // Extract wait time from error message if available
      const waitMatch = errorJson.error?.match(/wait (\d+)/i);
      const waitSeconds = waitMatch ? parseInt(waitMatch[1]) : null;

      if (attempt < retries) {
        // If we have a specific wait time, use it; otherwise use exponential backoff
        const waitMs = waitSeconds
          ? waitSeconds * 1000
          : Math.min(1000 * Math.pow(2, attempt), 10000);
        console.log(
          `Rate limited, retrying after ${waitMs}ms (attempt ${attempt + 1}/${retries + 1})`,
        );
        await new Promise((resolve) => setTimeout(resolve, waitMs));
        continue;
      }

      // Last attempt failed
      throw new Error(
        `Failed to create submission after ${retries + 1} attempts: ${response.status} ${errorText}`,
      );
    }

    // For non-rate-limit errors, throw immediately
    const errorText = await response.text();
    throw new Error(`Failed to create submission: ${response.status} ${errorText}`);
  }

  // Should never reach here, but TypeScript needs it
  throw new Error("Unexpected error in createTestSubmission");
}

/**
 * Wait for results to appear on the results page
 */
export async function waitForResults(page: Page, timeout = 30000): Promise<void> {
  // Wait for either success state or error state
  // Prioritize data-testid selectors for reliability
  const selectors = [
    '[data-testid="overall-score-value"]',
    '[data-testid="teacher-feedback"]',
    '[data-testid="cefr-badge"]',
    "text=Your Writing Feedback",
    "text=Results Not Available", // Error state
  ];

  // Wait for at least one of these to appear
  await Promise.race(
    selectors.map((selector) => page.waitForSelector(selector, { timeout }).catch(() => {})),
  );
  // No arbitrary wait needed - selectors are sufficient
}

export class HomePage {
  constructor(private page: Page) {}

  async goto() {
    await this.page.goto("/", { waitUntil: "domcontentloaded" });
    await this.page.waitForSelector('[data-testid="task-card"]', { timeout: 15000 });
  }

  async getTaskCards() {
    return this.page.locator('[data-testid="task-card"]');
  }

  async clickTask(taskId: string) {
    const link = this.page.locator(`[data-testid="task-card-link-${taskId}"]`);
    await link.waitFor({ state: "visible", timeout: 15000 });
    await link.scrollIntoViewIfNeeded();

    // Capture current URL to detect navigation issues
    const initialUrl = this.page.url();

    // Click with retry pattern for hydration/animation issues
    const maxRetries = 3;
    for (let i = 0; i < maxRetries; i++) {
      try {
        const navigationPromise = this.page.waitForURL(new RegExp(`/write/${taskId}`), {
          timeout: 10000, // Increased from 5000 to be more robust
          waitUntil: "domcontentloaded",
        });

        await link.click({ force: true }); // Force click in case of overlay/animation

        await navigationPromise;
        return; // Success
      } catch (e) {
        // If last retry, throw the error
        if (i === maxRetries - 1) {
          const currentUrl = this.page.url();
          if (!new RegExp(`/write/${taskId}`).test(currentUrl)) {
            throw new Error(
              `Navigation to /write/${taskId} failed after ${maxRetries} attempts. Current URL: ${currentUrl}.`,
            );
          }
          return; // Actually succeeded but promise timed out?
        }
        // Otherwise wait a bit and retry
        console.log(`Click attempt ${i + 1} failed, retrying...`);
        await this.page.waitForTimeout(1000);
      }
    }

    // Wait for the appropriate textarea - form works regardless of hydration state
    // Use a more robust wait that checks for both visibility and interactivity
    const readySelector =
      taskId === "custom"
        ? '[data-testid="custom-question-textarea"]'
        : '[data-testid="answer-textarea"]';

    // Wait for element to be visible and in DOM
    await this.page.waitForSelector(readySelector, {
      timeout: 20000,
      state: "visible",
    });

    // Additional wait to ensure element is ready for interaction
    await this.page
      .waitForFunction(
        (selector) => {
          const el = document.querySelector(selector);
          return el && el instanceof HTMLElement && !el.hasAttribute("disabled");
        },
        readySelector,
        { timeout: 5000 },
      )
      .catch(() => {
        // If element doesn't have disabled attribute check, that's fine
        // Just ensure it's visible
      });
  }

  async getProgressDashboard() {
    return this.page
      .locator('[data-testid="progress-dashboard"]')
      .or(this.page.locator("text=Your Progress").or(this.page.locator("text=Writings Completed")));
  }

  async getTitle() {
    return this.page.locator("h1.hero-title");
  }

  async clickHistoryLink() {
    await this.page.click('a[href="/history"]');
  }
}

export class HistoryPage {
  constructor(private page: Page) {}

  async goto() {
    await this.page.goto("/history");
  }

  async getTitle() {
    return this.page.locator('[data-testid="history-page-title"]');
  }

  async getEmptyState() {
    return this.page.locator('[data-testid="history-empty-state"]');
  }

  async getHistoryItemsContainer() {
    return this.page.locator('[data-testid="history-items-container"]');
  }

  async getContentDraftCards() {
    return this.page.locator('[data-testid="content-draft-card"]');
  }

  async getSubmissionCards() {
    return this.page.locator('[data-testid="submission-card"]');
  }

  async getContinueEditingButtons() {
    return this.page.locator('[data-testid="continue-editing-button"]');
  }

  async getViewResultsButtons() {
    return this.page.locator('[data-testid="view-results-button"]');
  }

  async clickContinueEditing(index = 0) {
    const buttons = await this.getContinueEditingButtons();
    await buttons.nth(index).click();
  }

  async clickViewResults(index = 0) {
    const buttons = await this.getViewResultsButtons();
    await buttons.nth(index).click();
  }
}

/**
 * Helper to wait for navigation to results page with error handling
 */
export async function waitForResultsNavigation(page: Page, timeout?: number): Promise<void> {
  const actualTimeout = timeout || (process.env.CI ? 120000 : 60000);
  const errorMonitoring = setupErrorMonitoring(page);

  try {
    if (page.isClosed()) {
      throw new Error("Page was closed before navigation could complete");
    }

    await playwrightExpect(page).toHaveURL(/\/results\/[a-f0-9-]+/, { timeout: actualTimeout });
  } catch (error: any) {
    const isClosed = page.isClosed();
    let currentUrl: string | null = null;

    if (!isClosed) {
      try {
        currentUrl = page.url();
      } catch {
        currentUrl = null;
      }
    }

    const isOnResultsPage = currentUrl && /\/results\/[a-f0-9-]+/.test(currentUrl);

    // If we're on results page, that's success (even if page closed)
    if (isOnResultsPage) {
      return;
    }

    // If page closed but not on results page, that's a problem
    if (isClosed) {
      throw new Error(
        `Page was closed during navigation. Network: ${errorMonitoring.networkErrors.join("; ")}. Console: ${errorMonitoring.consoleErrors.map(formatConsoleError).join("; ")}`,
      );
    }

    // Check for rate limit errors
    if (errorMonitoring.networkErrors.some((e) => e.includes("429"))) {
      throw new Error(
        `Rate limit hit. Please reduce parallel workers or add delays. Errors: ${errorMonitoring.networkErrors.join("; ")}`,
      );
    }

    // Check for visible error messages
    const pageError = await checkForPageErrors(page);
    if (pageError) {
      throw new Error(
        `Submission failed. Error: "${pageError}". Network: ${errorMonitoring.networkErrors.join("; ")}. Console: ${errorMonitoring.consoleErrors.map(formatConsoleError).join("; ")}`,
      );
    }

    // Re-throw original error if we haven't handled it
    throw error;
  } finally {
    errorMonitoring.cleanup();
  }
}

export class WritePage {
  constructor(private page: Page) {}

  async goto(taskId: string) {
    // Use domcontentloaded for faster navigation, then wait for specific elements
    await this.page.goto(`/write/${taskId}`, {
      waitUntil: "domcontentloaded",
      timeout: 60000,
    });

    // Wait for the textarea to be visible - this is what we actually need
    // The form works regardless of Zustand hydration state (user input uses local state)
    // For custom pages, wait for custom question textarea; otherwise wait for answer textarea
    const selector =
      taskId === "custom"
        ? '[data-testid="custom-question-textarea"]'
        : '[data-testid="answer-textarea"]';

    await this.page.waitForSelector(selector, {
      timeout: 20000,
      state: "visible",
    });

    // Additional wait to ensure page is fully interactive
    await this.page.waitForLoadState("networkidle", { timeout: 10000 }).catch(() => {
      // If networkidle times out, that's okay - page might still be loading resources
      // The important thing is the textarea is visible
    });
  }

  async getQuestionText() {
    // For regular pages, use prompt-box testid. For custom page, use the custom question textarea
    return this.page
      .locator('[data-testid="prompt-box"]')
      .or(this.page.locator('[data-testid="custom-question-textarea"]'));
  }

  async getCustomQuestionTextarea() {
    return this.page.locator('[data-testid="custom-question-textarea"]');
  }

  async getTextarea() {
    return this.page.locator('[data-testid="answer-textarea"]');
  }

  async typeEssay(text: string) {
    // Wait for textarea to be visible and ready
    // The form works regardless of Zustand hydration - user input uses local React state
    const textarea = this.page.locator('[data-testid="answer-textarea"]');
    await textarea.waitFor({ state: "visible", timeout: 15000 });

    // Clear the textarea first
    await textarea.clear();

    // Fill the textarea - Playwright's fill() triggers input events
    await textarea.fill(text);

    // Trigger an input event to ensure React processes the change
    // This helps when React state updates are delayed
    await textarea.evaluate((el) => {
      const event = new Event("input", { bubbles: true });
      el.dispatchEvent(event);
    });

    // Small delay to allow React to process the input event
    await this.page.waitForTimeout(100);

    // Wait for word count to update - check both the attribute and the display element
    // The word count is calculated from the answer prop which uses local React state
    // Increased timeout for CI and to handle slower state updates
    await this.page.waitForFunction(
      (minChars) => {
        const textarea = document.querySelector(
          '[data-testid="answer-textarea"]',
        ) as HTMLTextAreaElement | null;
        if (!textarea) return false;

        // Check that text was actually filled
        if (textarea.value.length < minChars) return false;

        // Check word count attribute (set from React prop)
        const countAttr = parseInt(textarea.getAttribute("data-word-count") || "0", 10);
        if (countAttr > 0) return true;

        // Also check the word count display element as fallback
        const countDisplay = document.querySelector('[data-testid="word-count-value"]');
        if (countDisplay) {
          const displayText = countDisplay.textContent || "";
          const match = displayText.match(/(\d+)\s+words?/);
          if (match && parseInt(match[1], 10) > 0) return true;
        }

        return false;
      },
      Math.min(text.length, 1000),
      { timeout: 20000 }, // Increased timeout for CI and slower state updates
    );
  }

  async waitForSubmitButtonEnabled(timeout = 20000) {
    // Wait for word count to reach minimum AND button to be enabled
    // Check both conditions together to avoid race conditions
    // Give React time to process state updates and re-render
    await this.page.waitForFunction(
      (minWords) => {
        const textarea = document.querySelector(
          '[data-testid="answer-textarea"]',
        ) as HTMLTextAreaElement | null;
        if (!textarea) return false;

        // Check word count - can check attribute or display
        let count = parseInt(textarea.getAttribute("data-word-count") || "0", 10);
        if (count === 0) {
          // Fallback: check word count display
          const countDisplay = document.querySelector('[data-testid="word-count-value"]');
          if (countDisplay) {
            const displayText = countDisplay.textContent || "";
            const match = displayText.match(/(\d+)\s+words?/);
            if (match) count = parseInt(match[1], 10);
          }
        }
        if (count < minWords) return false;

        // Check button state directly
        const button = document.querySelector(
          '[data-testid="submit-button"]',
        ) as HTMLButtonElement | null;
        if (!button) return false;

        // Button should not be disabled if word count is sufficient and there's text
        // Also check that we're not in a loading state
        const isDisabled = button.disabled || button.getAttribute("aria-busy") === "true";
        return !isDisabled && textarea.value.trim().length > 0;
      },
      MIN_ESSAY_WORDS,
      { timeout },
    );
  }

  async getWordCount() {
    const wordCountElement = this.page.locator('[data-testid="word-count-value"]');
    const text = await wordCountElement.textContent().catch(() => null);
    return text ? parseInt(text.match(/\d+/)?.[0] || "0") : 0;
  }

  async getSubmitButton() {
    return this.page.locator('[data-testid="submit-button"]');
  }

  async isSubmitButtonDisabled() {
    const button = await this.getSubmitButton();
    return (await button.count()) === 0 || (await button.first().isDisabled());
  }

  async clickSubmit() {
    const button = await this.getSubmitButton();
    await button.waitFor({ state: "visible", timeout: 15000 });
    await playwrightExpect(button.first()).toBeEnabled({ timeout: 20000 });

    // Small delay to ensure button state is stable and reduce race conditions
    await this.page.waitForTimeout(100);

    // Set up error monitoring
    const errorMonitoring = setupErrorMonitoring(this.page);

    try {
      // Start navigation promise before clicking
      const navigationPromise = this.page.waitForURL(/\/results\/[a-f0-9-]+/, {
        timeout: 90000,
        waitUntil: "domcontentloaded",
      });

      await button.first().click();

      try {
        await navigationPromise;
      } catch (error: any) {
        // Check if already on results page (race condition)
        const currentUrl = this.page.url();
        if (/\/results\/[a-f0-9-]+/.test(currentUrl)) {
          return;
        }

        // Check for page errors
        const pageError = await checkForPageErrors(this.page);

        // Build comprehensive error message
        const errorParts: string[] = [];

        if (pageError) {
          errorParts.push(`Error: "${pageError}"`);
        } else {
          errorParts.push("Error: Unknown error (error element visible but no text)");
        }

        if (errorMonitoring.failedRequests.length > 0) {
          const requestDetails = errorMonitoring.failedRequests
            .map((r) => `${r.status} ${r.statusText}: ${r.url}`)
            .join("; ");
          errorParts.push(`Failed requests: ${requestDetails}`);
        }

        if (errorMonitoring.networkErrors.length > 0) {
          errorParts.push(`Network: ${errorMonitoring.networkErrors.join("; ")}`);
        }

        if (errorMonitoring.consoleErrors.length > 0) {
          const formatted = errorMonitoring.consoleErrors.map(formatConsoleError).join("; ");
          errorParts.push(`Console errors: ${formatted}`);
        }

        if (error?.message && !error.message.includes("Timeout")) {
          errorParts.push(`Original: ${error.message}`);
        }
        if (errorParts.length > 0) {
          throw new Error(`Submission failed. ${errorParts.join(". ")}`);
        }
      }
    } finally {
      errorMonitoring.cleanup();
    }
  }

  async getLoadingState() {
    return this.page
      .locator("text=/Analyzing your writing|Loading|Processing/i")
      .or(this.page.locator(".spinner").or(this.page.locator('[aria-busy="true"]')));
  }

  async getError() {
    return this.page
      .locator(".error")
      .or(
        this.page
          .locator('[role="alert"]')
          .or(this.page.locator("text=/too short|too long|at least|maximum|need.*words|minimum/i")),
      );
  }

  async getSelfEvalChecklist() {
    return this.page
      .locator("text=/Self-Evaluation Checklist|Self-Evaluation/i")
      .or(this.page.locator("text=/Did I answer|checklist/i"));
  }
}

export class ResultsPage {
  constructor(private page: Page) {}

  async goto(submissionId: string, parentId?: string) {
    // parentId parameter kept for backwards compatibility but not used in URL
    // parentSubmissionId is now read from results.meta instead of URL param
    await this.page.goto(`/results/${submissionId}`);
  }

  async waitForResults(timeout = 30000) {
    await waitForResults(this.page, timeout);
  }

  async getOverallScore() {
    return this.page.locator('[data-testid="overall-score-value"]');
  }

  async getCEFRLevel() {
    return this.page.locator('[data-testid="cefr-badge"]');
  }

  async getDimensionScores() {
    return {
      TA: this.page.locator('[data-testid="dimension-score-TA"]'),
      CC: this.page.locator('[data-testid="dimension-score-CC"]'),
      Vocab: this.page.locator('[data-testid="dimension-score-Vocab"]'),
      Grammar: this.page.locator('[data-testid="dimension-score-Grammar"]'),
    };
  }

  async getGrammarErrorsSection() {
    return this.page
      .locator('[data-testid="grammar-errors-section"]')
      .or(this.page.locator('[data-testid="heat-map-section"]'));
  }

  async getErrorCount() {
    return this.page.locator("text=/Found \\d+ issue/");
  }

  async getTeacherFeedback() {
    return this.page
      .locator("#teacher-feedback-container")
      .or(
        this.page
          .locator('[data-testid="teacher-feedback"]')
          .or(this.page.locator("text=/Teacher.*Feedback|Preparing feedback/i")),
      );
  }

  async getTeacherAnalysisButton() {
    return this.page
      .locator('button:has-text("Get Teacher Analysis")')
      .or(this.page.locator('button:has-text("View Detailed Analysis")'));
  }

  async clickTeacherAnalysis() {
    const button = await this.getTeacherAnalysisButton();
    await button.click();
  }

  async getEditableEssay() {
    // Use data-testid for reliable selection
    return this.page.locator('[data-testid="editable-essay-textarea"]');
  }

  async getDraftHistory() {
    return this.page
      .locator('[data-testid="draft-history"]')
      .or(this.page.locator("h2:has-text('Draft History')").locator(".."));
  }

  async getDraftButtons() {
    // Use data-testid pattern for draft buttons
    return this.page.locator('[data-testid^="draft-button-"]');
  }

  async getDraftButton(draftNumber: number) {
    return this.page.locator(`[data-testid="draft-button-${draftNumber}"]`);
  }

  async clickDraftButton(draftNumber: number) {
    const button = await this.getDraftButton(draftNumber);
    await button.click();
  }

  async getCurrentDraftNumber(): Promise<number | null> {
    // Find the draft button with primary color (current draft)
    const currentButton = this.page
      .locator('div[style*="var(--primary-color)"], div[style*="rgb"]')
      .filter({
        hasText: /Draft \d+/,
      });
    const count = await currentButton.count();
    if (count > 0) {
      const text = await currentButton.first().textContent();
      const match = text?.match(/Draft (\d+)/);
      if (match) {
        return parseInt(match[1], 10);
      }
    }
    return null;
  }

  async getDraftComparisonTable() {
    return this.page.locator('[data-testid="draft-comparison-table"]');
  }

  async getLoadingMessage() {
    return this.page
      .locator("text=Analyzing your writing…")
      .or(this.page.locator("text=/Loading Results|Fetching your essay results/"));
  }

  async getErrorState() {
    return this.page.locator("text=Results Not Available");
  }

  async getSubmitDraftButton() {
    return this.page.locator('[data-testid="submit-improved-draft-button"]');
  }

  /**
   * Wait for results and draft to be stored in Zustand store (localStorage)
   * This is needed because storage happens asynchronously via useEffect
   */
  async waitForDraftStorage(submissionId: string, timeout = 10000): Promise<void> {
    await this.page.waitForFunction(
      (id) => {
        try {
          const store = localStorage.getItem("writeo-draft-store");
          if (!store) return false;
          const parsed = JSON.parse(store);

          // Check if results exist for this submissionId
          if (parsed?.state?.results && parsed.state.results[id]) {
            return true;
          }

          // Also check draft arrays as fallback
          if (parsed?.state?.drafts) {
            const drafts = parsed.state.drafts;
            for (const key in drafts) {
              if (Array.isArray(drafts[key])) {
                if (drafts[key].some((d: any) => d.submissionId === id)) {
                  return true;
                }
              }
            }
          }
          return false;
        } catch {
          return false;
        }
      },
      submissionId,
      { timeout },
    );
  }

  /**
   * Wait for draft history to appear (requires 2+ drafts)
   * Simply waits for the Draft History UI to become visible
   */
  async waitForDraftHistory(timeout = 20000): Promise<void> {
    // Wait for the draft history section to be visible in the UI
    // This is the reliable indicator that drafts are being displayed
    // Increased timeout for CI/production
    const ciTimeout = process.env.CI ? timeout * 2 : timeout;

    try {
      await this.page.waitForSelector('[data-testid="draft-history"]', {
        timeout: ciTimeout,
        state: "visible",
      });
    } catch {
      // Fallback to h2 heading if testid selector fails
      await this.page.waitForSelector('h2:has-text("Draft History")', {
        timeout: 30000, // Increased timeout for draft history
        state: "visible",
      });
    }
  }

  /**
   * Wait for Zustand store to hydrate from localStorage
   * This is needed when manually setting localStorage before navigation
   */
  async waitForStoreHydration(timeout = 5000): Promise<void> {
    await this.page.waitForFunction(
      () => {
        try {
          const storeData = localStorage.getItem("writeo-draft-store");
          if (!storeData) return true; // Store will be created on first access
          const parsed = JSON.parse(storeData);
          return parsed && typeof parsed === "object" && ("state" in parsed || "drafts" in parsed);
        } catch {
          return false;
        }
      },
      { timeout },
    );
    // No arbitrary wait - the function condition is sufficient
  }
}

export function getTestEssay(type: keyof typeof TEST_ESSAYS): string {
  return TEST_ESSAYS[type];
}

export function generateValidEssay(): string {
  const sentences = [
    "I believe that universities should focus more on practical skills rather than theoretical knowledge.",
    "Theoretical knowledge is important but practical application matters more in the real world.",
    "Students need hands-on experience to succeed in their careers after graduation.",
    "Many employers value practical skills over academic achievements when hiring.",
    "However, theoretical knowledge provides a strong foundation for understanding complex concepts.",
    "A balance between both approaches would be ideal for comprehensive education.",
    "Practical skills help students apply what they learn in real situations.",
    "Theoretical knowledge helps students understand why things work the way they do.",
    "Both are necessary for a complete and well-rounded education system.",
    "Universities should integrate more practical training into their curriculum.",
    "This would better prepare students for the challenges they will face.",
    "Workplace demands are changing rapidly and education must adapt accordingly.",
    "Students who have practical experience are more confident in their abilities.",
    "They can solve real problems more effectively than those with only theory.",
    "The combination of theory and practice creates the best learning outcomes.",
    "Educational institutions should partner with industries to provide practical training.",
    "This collaboration benefits both students and employers in the long run.",
    "Students gain valuable experience while employers get well-prepared graduates.",
    "The future of education lies in balancing theoretical and practical learning.",
    "We must ensure our education system prepares students for real-world success.",
  ];

  // Build essay to ensure 250-400 words (safe range)
  let essay = "";
  let wordCount = 0;
  let i = 0;

  // Add sentences until we reach at least 250 words
  while (wordCount < 250 && i < sentences.length * 3) {
    essay += (essay ? " " : "") + sentences[i % sentences.length];
    const words = essay.split(/\s+/).filter((w) => w.length > 0);
    wordCount = words.length;
    i++;
  }

  // Ensure we have at least 250 words
  if (wordCount < 250) {
    // Repeat sentences if needed
    while (wordCount < 250) {
      essay += " " + sentences[i % sentences.length];
      const words = essay.split(/\s+/).filter((w) => w.length > 0);
      wordCount = words.length;
      i++;
    }
  }

  // Trim to max 500 words if needed
  const words = essay.split(/\s+/).filter((w) => w.length > 0);
  if (words.length > 500) {
    return words.slice(0, 500).join(" ");
  }

  // Verify word count is in valid range
  if (words.length < 250) {
    throw new Error(`generateValidEssay() produced only ${words.length} words, expected 250-500`);
  }

  return essay;
}
