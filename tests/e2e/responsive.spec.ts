import { test, expect } from "./fixtures";

/**
 * Responsive Design Tests - Essential viewport checks
 */

test.describe("Responsive Design", () => {
  test("mobile layout (375px) - no horizontal scroll", async ({ page, writePage }) => {
    await page.setViewportSize({ width: 375, height: 667 });
    await writePage.goto("1");

    const bodyWidth = await page.evaluate(() => document.body.scrollWidth);
    expect(bodyWidth).toBeLessThanOrEqual(395); // Allow small margin

    const textarea = await writePage.getTextarea();
    await expect(textarea.first()).toBeVisible();
  });

  test("tablet layout (768px) - content visible", async ({ page, writePage }) => {
    await page.setViewportSize({ width: 768, height: 1024 });
    await writePage.goto("1");

    const textarea = await writePage.getTextarea();
    await expect(textarea.first()).toBeVisible();
  });

  test("desktop layout (1920px) - content visible", async ({ page, writePage }) => {
    await page.setViewportSize({ width: 1920, height: 1080 });
    await writePage.goto("1");

    const textarea = await writePage.getTextarea();
    await expect(textarea.first()).toBeVisible();
  });
});
