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
const API_BASE = process.env.API_BASE || process.env.API_BASE_URL || "http://localhost:8787";
const API_KEY = process.env.API_KEY || "";

/**
 * Create a test submission via API
 * Returns submission ID and results
 */
export async function createTestSubmission(
  questionText: string,
  answerText: string
): Promise<{ submissionId: string; results: any }> {
  if (!API_KEY) {
    throw new Error("API_KEY environment variable required for E2E tests");
  }

  const submissionId = randomUUID();
  const questionId = randomUUID();
  const answerId = randomUUID();

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

  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(`Failed to create submission: ${response.status} ${errorText}`);
  }

  const results = await response.json();
  return { submissionId, results };
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
    return this.page.locator(".prompt-box");
  }

  async getTextarea() {
    return this.page.locator("textarea#answer").or(this.page.locator("textarea").first());
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
    await this.getTeacherAnalysisButton().click();
  }

  async getEditableEssay() {
    return this.page
      .locator("text=Improve Your Writing")
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

  async getLoadingMessage() {
    return this.page
      .locator("text=Analyzing your writing…")
      .or(this.page.locator("text=/Loading Results|Fetching your essay results/"));
  }

  async getErrorState() {
    return this.page.locator("text=Results Not Available");
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
