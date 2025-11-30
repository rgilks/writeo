import { Page, expect } from "@playwright/test";
import { randomUUID } from "crypto";
// Note: Environment variables are already loaded in playwright.config.ts
// No need to load them again here to avoid duplicate messages

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

// API configuration - loaded from .env or .env.local
// Always prefer TEST_API_KEY for tests (higher rate limits)
const API_BASE = process.env.API_BASE || process.env.API_BASE_URL || "http://localhost:8787";
const API_KEY = process.env.TEST_API_KEY || process.env.API_KEY || "";

// Track last submission time to add delays between parallel test submissions
let lastSubmissionTime = 0;
const MIN_DELAY_MS = 100; // Reduced from 200ms - only needed for rate limiting

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

  // Add small delay to prevent hitting rate limits when tests run in parallel
  const now = Date.now();
  const timeSinceLastSubmission = now - lastSubmissionTime;
  if (timeSinceLastSubmission < MIN_DELAY_MS) {
    await new Promise((resolve) => setTimeout(resolve, MIN_DELAY_MS - timeSinceLastSubmission));
  }
  lastSubmissionTime = Date.now();

  const submissionId = randomUUID();
  const questionId = randomUUID();
  const answerId = randomUUID();

  for (let attempt = 0; attempt <= retries; attempt++) {
    // Exponential backoff: wait longer on each retry
    if (attempt > 0) {
      const backoffMs = Math.min(1000 * Math.pow(2, attempt - 1), 10000); // Max 10 seconds
      await new Promise((resolve) => setTimeout(resolve, backoffMs));
    }

    const response = await fetch(`${API_BASE}/text/submissions/${submissionId}`, {
      method: "PUT",
      headers: {
        Authorization: `Token ${API_KEY}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        submission: [
          {
            part: "1",
            answers: [
              {
                id: answerId,
                "question-number": 1,
                "question-id": questionId,
                ...(questionText?.trim() ? { "question-text": questionText.trim() } : {}),
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
  const selectors = [
    '[data-testid="results-loaded"]',
    "text=Your Writing Feedback",
    ".overall-score-value",
    "text=/Your Writing Level|Overall Score|Estimated Level/i",
    "text=/A[12]|B[12]|C[12]/", // CEFR level
    "text=Results Not Available", // Error state
    "#teacher-feedback-container", // Teacher feedback
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
    await this.page.goto("/");
  }

  async getTaskCards() {
    return this.page.locator(".task-card");
  }

  async clickTask(taskId: string) {
    await this.page.click(`a[href="/write/${taskId}"]`);
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
    return this.page.locator("h1:has-text('History')");
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

export class WritePage {
  constructor(private page: Page) {}

  async goto(taskId: string) {
    await this.page.goto(`/write/${taskId}`);
  }

  async getQuestionText() {
    // For regular pages, use .prompt-box. For custom page, use the first textarea (question input)
    // Use .first() to avoid strict mode violations when both exist
    return this.page.locator(".prompt-box").first();
  }

  async getCustomQuestionTextarea() {
    return this.page.locator(".question-card textarea").first();
  }

  async getTextarea() {
    // Answer textarea has id="answer" or is the last textarea (answer comes after question)
    return this.page.locator("textarea#answer").or(this.page.locator("textarea").last());
  }

  async typeEssay(text: string) {
    const textarea = this.page.locator("textarea#answer");
    await textarea.waitFor({ state: "visible", timeout: 10000 });
    // Use fill() for better performance, especially for long essays
    // Then trigger input event to ensure React state updates
    await textarea.fill(text);
    // Only evaluate if element still exists (might be destroyed during navigation)
    try {
      await textarea.evaluate((el) => {
        el.dispatchEvent(new Event("input", { bubbles: true }));
        el.dispatchEvent(new Event("change", { bubbles: true }));
      });
    } catch (e) {
      // Element might have been destroyed, which is fine - fill() already updated the value
    }
  }

  async getWordCount() {
    const text = await this.page
      .locator("text=/\\d+ (word|words)/i")
      .first()
      .textContent()
      .catch(() => null);
    return text ? parseInt(text.match(/\d+/)?.[0] || "0") : 0;
  }

  async getSubmitButton() {
    return this.page
      .locator('button[type="submit"]')
      .or(this.page.locator('button:has-text("Submit")'));
  }

  async isSubmitButtonDisabled() {
    const button = await this.getSubmitButton();
    return (await button.count()) === 0 || (await button.first().isDisabled());
  }

  async clickSubmit() {
    const button = await this.getSubmitButton();
    await button.waitFor({ state: "visible", timeout: 5000 });
    await button.first().click();
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
    return this.page
      .locator(".overall-score-value")
      .or(this.page.locator("text=/Overall Score|Estimated Level|Your Writing Level/"));
  }

  async getCEFRLevel() {
    return this.page
      .locator("text=/\\b(A1|A2|B1|B2|C1|C2)\\b/")
      .or(
        this.page.locator(
          "text=/CEFR|Level|Proficient|Independent|Basic|Elementary|Intermediate|Advanced/",
        ),
      );
  }

  async getDimensionScores() {
    const grid = this.page.locator(".dimensions-grid-responsive");
    return {
      TA: this.page
        .locator("text=/Task Achievement|TA|Answering the Question/")
        .or(grid.locator("text=/Answering the Question/")),
      CC: this.page
        .locator("text=/Coherence|CC|Organization/")
        .or(grid.locator("text=/Organization/")),
      Vocab: this.page.locator("text=/Vocabulary|Vocab/").or(grid.locator("text=/Vocabulary/")),
      Grammar: this.page.locator("text=/Grammar/").or(grid.locator("text=/Grammar/")),
    };
  }

  async getGrammarErrorsSection() {
    return this.page
      .locator("text=Grammar & Language Feedback")
      .or(
        this.page
          .locator("text=Your Writing with Feedback")
          .or(
            this.page
              .locator("text=Common Areas to Improve")
              .or(this.page.locator("text=No high-confidence errors")),
          ),
      );
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
    // Find the textarea within the "Improve Your Writing" section
    // The component should always render EditableEssay when finalAnswerText exists

    // Wait for the "Improve Your Writing" section to appear (this is the main indicator)
    const improveSection = this.page.locator("text=Improve Your Writing");
    const sectionExists = await improveSection
      .waitFor({ timeout: 10000 })
      .then(() => true)
      .catch(() => false);

    if (!sectionExists) {
      // Section doesn't exist, return empty locator
      return this.page.locator("textarea").filter({ hasText: /^$/ }); // Empty locator
    }

    // Look for textarea - it should be a sibling or descendant of the Improve Your Writing heading
    // Try multiple strategies to find the textarea
    let textarea = this.page.locator("text=Improve Your Writing").locator("..").locator("textarea");

    // If not found, try finding any textarea that's near the Improve Your Writing section
    if ((await textarea.count()) === 0) {
      // Find the parent container and look for textarea within it
      const container = this.page.locator("text=Improve Your Writing").locator("../..");
      textarea = container.locator("textarea");
    }

    // If still not found, look for any textarea on the page (fallback)
    if ((await textarea.count()) === 0) {
      textarea = this.page.locator("textarea").filter({ hasText: /./ });
    }

    // Wait for at least one textarea to be visible (with timeout)
    await textarea
      .first()
      .waitFor({ timeout: 5000 })
      .catch(() => {
        // If textarea doesn't appear, that's OK - return what we have
      });

    return textarea;
  }

  async getDraftHistory() {
    return this.page
      .locator('[data-testid="draft-history"]')
      .or(this.page.locator("h2:has-text('Draft History')").locator(".."));
  }

  async getDraftButtons() {
    const draftHistory = await this.getDraftHistory();
    // Get buttons: direct children of the flex container
    // Structure: Card > FlexContainer > Buttons
    // So we want direct div children of direct div children of the card
    // Filter by having "Draft" text to be safe
    return draftHistory.locator("> div > div").filter({ hasText: /^Draft \d+/ });
  }

  async getDraftButton(draftNumber: number) {
    const draftButtons = await this.getDraftButtons();
    return draftButtons.filter({ hasText: `Draft ${draftNumber}` }).first();
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
    return this.page.locator("text=Draft Comparison").or(this.page.locator("table"));
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
    return this.page
      .locator('button:has-text("Submit Improved Draft")')
      .or(this.page.locator('button:has-text("Submit")').filter({ hasText: /draft|improve/i }));
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
  async waitForDraftHistory(timeout = 10000): Promise<void> {
    // Wait for the draft history section to be visible in the UI
    // This is the reliable indicator that drafts are being displayed
    try {
      await this.page.waitForSelector('[data-testid="draft-history"]', {
        timeout,
        state: "visible",
      });
    } catch {
      // Fallback to h2 heading if testid selector fails
      await this.page.waitForSelector('h2:has-text("Draft History")', {
        timeout,
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
