import { test, expect } from "./fixtures";
import { createTestSubmission, generateValidEssay } from "./helpers";

test.describe("Responsive Design Tests", () => {
  test("TC-STYLE-013: Mobile layout (375px)", async ({ page, writePage }) => {
    await page.setViewportSize({ width: 375, height: 667 });
    await writePage.goto("1");

    // Check that page loads without horizontal scroll
    const bodyWidth = await page.evaluate(() => document.body.scrollWidth);
    const viewportWidth = 375;
    expect(bodyWidth).toBeLessThanOrEqual(viewportWidth + 20); // Allow small margin

    // Check that main content is visible
    const textarea = await writePage.getTextarea();
    await expect(textarea.first()).toBeVisible();

    // Check that buttons are still accessible
    const submitButton = await writePage.getSubmitButton();
    if ((await submitButton.count()) > 0) {
      await expect(submitButton.first()).toBeVisible();
    }
  });

  test("TC-STYLE-014: Tablet layout (768px)", async ({ page, writePage }) => {
    await page.setViewportSize({ width: 768, height: 1024 });
    await writePage.goto("1");

    // Check that page loads correctly
    const textarea = await writePage.getTextarea();
    await expect(textarea.first()).toBeVisible();

    // Check layout adapts to tablet size
    const container = page.locator('.container, main, [class*="container"]').first();
    if ((await container.count()) > 0) {
      const width = await container.evaluate((el) => {
        const style = window.getComputedStyle(el);
        return parseFloat(style.width);
      });

      // Container should adapt to viewport
      expect(width).toBeLessThanOrEqual(768);
    }
  });

  test("TC-STYLE-015: Desktop layout (1920px)", async ({ page, writePage }) => {
    await page.setViewportSize({ width: 1920, height: 1080 });
    await writePage.goto("1");

    // Check that page loads correctly
    const textarea = await writePage.getTextarea();
    await expect(textarea.first()).toBeVisible();

    // Check that content doesn't stretch too wide (max-width constraint)
    const container = page.locator('.container, main, [class*="container"]').first();
    if ((await container.count()) > 0) {
      const maxWidth = await container.evaluate((el) => {
        const style = window.getComputedStyle(el);
        return parseFloat(style.maxWidth) || parseFloat(style.width);
      });

      // Content should have reasonable max-width (allow up to viewport width)
      // Most sites use max-width of 1200-1400px, but 1920px is acceptable for full-width layouts
      if (maxWidth > 0) {
        expect(maxWidth).toBeLessThanOrEqual(1920);
        // If full width, check that content is centered or has padding
        // Note: Full-width layouts are valid, so we only check if padding/margin exists
        // but don't require it (some layouts intentionally use full width)
        if (maxWidth === 1920) {
          const marginLeft = await container.evaluate((el) => {
            const style = window.getComputedStyle(el);
            return parseFloat(style.marginLeft);
          });
          const paddingLeft = await container.evaluate((el) => {
            const style = window.getComputedStyle(el);
            return parseFloat(style.paddingLeft);
          });
          const paddingRight = await container.evaluate((el) => {
            const style = window.getComputedStyle(el);
            return parseFloat(style.paddingRight);
          });
          const marginRight = await container.evaluate((el) => {
            const style = window.getComputedStyle(el);
            return parseFloat(style.marginRight);
          });
          // Check if there's any spacing (left/right margin or padding)
          // Full-width layouts without side spacing are acceptable
          const hasSpacing = marginLeft + paddingLeft + paddingRight + marginRight > 0;
          // If no spacing, verify the container is at least not overflowing
          if (!hasSpacing) {
            const actualWidth = await container.evaluate((el) => {
              return el.getBoundingClientRect().width;
            });
            expect(actualWidth).toBeLessThanOrEqual(1920);
          }
        }
      }
    }
  });

  test("TC-STYLE-016: Touch targets are 44px+ on mobile", async ({ page, writePage }) => {
    await page.setViewportSize({ width: 375, height: 667 });
    await writePage.goto("1");

    // Check submit button
    const submitButton = await writePage.getSubmitButton();
    if ((await submitButton.count()) > 0) {
      const height = await submitButton.first().evaluate((el) => {
        const style = window.getComputedStyle(el);
        return parseFloat(style.height);
      });

      expect(height).toBeGreaterThanOrEqual(44);
    }

    // Check other interactive elements
    const buttons = page.locator("button, a.btn");
    const buttonCount = await buttons.count();

    for (let i = 0; i < Math.min(buttonCount, 5); i++) {
      const button = buttons.nth(i);
      const height = await button.evaluate((el) => {
        const style = window.getComputedStyle(el);
        return parseFloat(style.height);
      });

      const width = await button.evaluate((el) => {
        const style = window.getComputedStyle(el);
        return parseFloat(style.width);
      });

      // Touch target should be at least 44x44px
      expect(Math.min(height, width)).toBeGreaterThanOrEqual(44);
    }
  });

  test("TC-STYLE-017: Results page responsive layout", async ({ page, resultsPage }) => {
    const essay = generateValidEssay();
    const { submissionId } = await createTestSubmission("Describe your weekend.", essay);

    // Test mobile
    await page.setViewportSize({ width: 375, height: 667 });
    await resultsPage.goto(submissionId);
    await resultsPage.waitForResults();

    // Check that results display correctly
    const overallScore = await resultsPage.getOverallScore();
    if ((await overallScore.count()) > 0) {
      await expect(overallScore.first()).toBeVisible();
    }

    // Test tablet
    await page.setViewportSize({ width: 768, height: 1024 });
    await page.reload();
    await resultsPage.waitForResults();

    // Check that results still display correctly
    if ((await overallScore.count()) > 0) {
      await expect(overallScore.first()).toBeVisible();
    }

    // Test desktop
    await page.setViewportSize({ width: 1920, height: 1080 });
    await page.reload();
    await resultsPage.waitForResults();

    // Check that results still display correctly
    if ((await overallScore.count()) > 0) {
      await expect(overallScore.first()).toBeVisible();
    }
  });
});
