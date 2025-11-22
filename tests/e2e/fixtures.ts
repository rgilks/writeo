import { test as base, expect } from "@playwright/test";
import { HomePage, WritePage, ResultsPage } from "./helpers";
import { createTestSubmission, generateValidEssay, getTestEssay } from "./helpers";

type TestFixtures = {
  homePage: HomePage;
  writePage: WritePage;
  resultsPage: ResultsPage;
  // Shared submission fixtures to reduce API calls
  sharedSubmission: { submissionId: string; questionText: string; essay: string };
  sharedSubmissionWithErrors: { submissionId: string; questionText: string; essay: string };
  sharedSubmissionCorrected: { submissionId: string; questionText: string; essay: string };
};

export const test = base.extend<TestFixtures>({
  homePage: async ({ page }, use) => await use(new HomePage(page)),
  writePage: async ({ page }, use) => await use(new WritePage(page)),
  resultsPage: async ({ page }, use) => await use(new ResultsPage(page)),

  // Shared submission for tests that check the same results page
  // Created once per test file/worker and reused across tests
  sharedSubmission: async ({}, use) => {
    const questionText = "Describe your weekend.";
    const essay = generateValidEssay();
    const { submissionId } = await createTestSubmission(questionText, essay);
    await use({ submissionId, questionText, essay });
  },

  // Shared submission with errors for grammar error tests
  sharedSubmissionWithErrors: async ({}, use) => {
    const questionText = "Describe your weekend.";
    const essay = getTestEssay("withErrors");
    const { submissionId } = await createTestSubmission(questionText, essay);
    await use({ submissionId, questionText, essay });
  },

  // Shared submission with corrected essay for no-error tests
  sharedSubmissionCorrected: async ({}, use) => {
    const questionText = "Describe your weekend.";
    const essay = getTestEssay("corrected");
    const { submissionId } = await createTestSubmission(questionText, essay);
    await use({ submissionId, questionText, essay });
  },
});

export { expect };
