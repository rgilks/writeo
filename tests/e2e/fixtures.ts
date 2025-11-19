import { test as base, expect } from "@playwright/test";
import { HomePage, WritePage, ResultsPage } from "./helpers";

type TestFixtures = {
  homePage: HomePage;
  writePage: WritePage;
  resultsPage: ResultsPage;
};

export const test = base.extend<TestFixtures>({
  homePage: async ({ page }, use) => await use(new HomePage(page)),
  writePage: async ({ page }, use) => await use(new WritePage(page)),
  resultsPage: async ({ page }, use) => await use(new ResultsPage(page)),
});

export { expect };
