import { test as base, expect } from "@playwright/test";
import { HomePage, WritePage, ResultsPage } from "./helpers";
import { createTestSubmission, generateValidEssay, getTestEssay } from "./helpers";

type TestFixtures = {
  homePage: HomePage;
  writePage: WritePage;
  resultsPage: ResultsPage;
  // Shared submission fixtures - created once per test file and reused
  sharedSubmission: { submissionId: string; questionText: string; essay: string };
  sharedSubmissionWithErrors: { submissionId: string; questionText: string; essay: string };
  sharedSubmissionCorrected: { submissionId: string; questionText: string; essay: string };
};

export const test = base.extend<TestFixtures>({
  homePage: async ({ page }, use) => await use(new HomePage(page)),
  writePage: async ({ page }, use) => await use(new WritePage(page)),
  resultsPage: async ({ page }, use) => await use(new ResultsPage(page)),

  // Shared submission - created once per worker and reused across tests
  // Results are stored on server for later retrieval
  sharedSubmission: async ({}, use) => {
    const questionText = "Describe your weekend.";
    const essay = generateValidEssay();
    const { submissionId } = await createTestSubmission(questionText, essay);
    await use({ submissionId, questionText, essay });
  },

  sharedSubmissionWithErrors: async ({}, use) => {
    const questionText = "Describe your weekend.";
    const essay = getTestEssay("withErrors");
    const { submissionId } = await createTestSubmission(questionText, essay);
    await use({ submissionId, questionText, essay });
  },

  sharedSubmissionCorrected: async ({}, use) => {
    const questionText = "Describe your weekend.";
    const essay = getTestEssay("corrected");
    const { submissionId } = await createTestSubmission(questionText, essay);
    await use({ submissionId, questionText, essay });
  },
});

export { expect };
