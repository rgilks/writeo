import { Page, expect } from "@playwright/test";
import { randomUUID } from "crypto";
import { config } from "dotenv";
import { resolve } from "path";

// Load environment variables from .env and .env.local
// .env.local takes precedence over .env
config({ path: resolve(process.cwd(), ".env") });
config({ path: resolve(process.cwd(), ".env.local"), override: true });

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
      "This is a test sentence with multiple words to create a longer essay for testing purposes. "
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
// Note: This is per-worker (each Playwright worker has its own instance)
// This helps prevent hitting rate limits when multiple tests run in parallel
let lastSubmissionTime = 0;
const MIN_DELAY_MS = 200; // Minimum 200ms between submissions (helps with parallel workers)

/**
 * Create a test submission via API with retry logic for rate limiting
 * Returns submission ID and results
 */
export async function createTestSubmission(
  questionText: string,
  answerText: string,
  retries = 3
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
            part: 1,
            answers: [
              {
                id: answerId,
                "question-number": 1,
                "question-id": questionId,
                "question-text": questionText,
                text: answerText,
              },
            ],
          },
        ],
        template: { name: "generic", version: 1 },
        storeResults: false, // No server storage for tests
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
          `Rate limited, retrying after ${waitMs}ms (attempt ${attempt + 1}/${retries + 1})`
        );
        await new Promise((resolve) => setTimeout(resolve, waitMs));
        continue;
      }

      // Last attempt failed
      throw new Error(
        `Failed to create submission after ${retries + 1} attempts: ${response.status} ${errorText}`
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
  // Try multiple selectors to detect when results are loaded
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
    selectors.map((selector) => page.waitForSelector(selector, { timeout }).catch(() => {}))
  );

  // Additional wait to ensure content is rendered
  await page.waitForTimeout(500);
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

  async getCustomQuestionTextarea() {
    return this.page.locator("textarea").first();
  }

  async getTextarea() {
    // Answer textarea has id="answer" or is the last textarea (answer comes after question)
    return this.page.locator("textarea#answer").or(this.page.locator("textarea").last());
  }

  async typeEssay(text: string) {
    const textarea = this.page.locator("textarea#answer");
    await textarea.waitFor({ state: "visible", timeout: 10000 });
    await textarea.fill(text);
    await textarea.evaluate((el) => el.dispatchEvent(new Event("input", { bubbles: true })));
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
          .or(this.page.locator("text=/too short|too long|at least|maximum|need.*words|minimum/i"))
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
    const url = parentId
      ? `/results/${submissionId}?parent=${parentId}`
      : `/results/${submissionId}`;
    await this.page.goto(url);
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
          "text=/CEFR|Level|Proficient|Independent|Basic|Elementary|Intermediate|Advanced/"
        )
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
              .or(this.page.locator("text=No high-confidence errors"))
          )
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
          .or(this.page.locator("text=/Teacher.*Feedback|Preparing feedback/i"))
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
    // The main editing textarea is the first textarea in that section
    const improveSection = this.page.locator("text=Improve Your Writing").locator("..");
    return improveSection
      .locator("textarea")
      .first()
      .or(
        this.page
          .locator("text=Question text is not available for editing")
          .or(this.page.locator("textarea").nth(1))
      );
  }

  async getDraftHistory() {
    return this.page
      .locator('[data-testid="draft-history"]')
      .or(this.page.locator("text=Draft History"));
  }

  async getDraftButtons() {
    // Draft buttons are divs with "Draft {number}" text
    // They're inside the draft history section
    const draftHistory = await this.getDraftHistory();
    if ((await draftHistory.count()) > 0) {
      return draftHistory.locator("..").locator("div:has-text(/^Draft \\d+$/i)");
    }
    return this.page.locator("div:has-text(/^Draft \\d+$/i)");
  }

  async getDraftButton(draftNumber: number) {
    // Find draft button by number within draft history section
    const draftHistory = await this.getDraftHistory();
    if ((await draftHistory.count()) > 0) {
      return draftHistory.locator("..").locator(`div:has-text("Draft ${draftNumber}")`).first();
    }
    return this.page.locator(`div:has-text("Draft ${draftNumber}")`).first();
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
