/**
 * Production smoke test - verifies deployment works with real APIs
 * This is the ONLY test that should hit real Modal/Groq/OpenAI endpoints
 * Run only after successful deployment to production
 */

import { test, expect } from "@playwright/test";
import { WritePage, ResultsPage, generateValidEssay } from "./helpers";

test.describe("Production Smoke Test", () => {
  test("smoke - full submission flow works with real APIs", async ({ page }) => {
    // Skip this test if running locally with mocks (only run in CI against production)
    const useMockServices = process.env.USE_MOCK_SERVICES === "true";
    const isLocalhost = process.env.PLAYWRIGHT_BASE_URL?.includes("localhost") ?? true;

    if (useMockServices || isLocalhost) {
      test.skip();
      return;
    }

    // Navigate to write page
    const writePage = new WritePage(page);
    await writePage.goto("1");

    // Verify page loaded
    const questionText = await writePage.getQuestionText();
    await expect(questionText).toBeVisible({ timeout: 10000 });

    // Type a valid essay
    const essay = generateValidEssay();
    await writePage.typeEssay(essay);

    // Wait for submit button to be enabled
    await writePage.waitForSubmitButtonEnabled();

    // Submit the essay
    await writePage.clickSubmit();

    // Wait for navigation to results page
    await page.waitForURL(/\/results\/[a-f0-9-]+/, {
      timeout: 120000, // 2 minutes for real API calls
      waitUntil: "domcontentloaded",
    });

    // Verify we're on the results page
    const resultsPage = new ResultsPage(page);
    await resultsPage.waitForResults();

    // Verify results are displayed - overall score indicates successful processing
    const overallScore = await resultsPage.getOverallScore();
    await expect(overallScore.first()).toBeVisible({ timeout: 15000 });

    // Verify CEFR level is displayed (indicates assessor results are present)
    const cefrLevel = await resultsPage.getCEFRLevel();
    await expect(cefrLevel.first()).toBeVisible({ timeout: 10000 });

    console.log("✅ Smoke test passed: Full submission flow works in production");
  });
});
