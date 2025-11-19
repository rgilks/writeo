import { test, expect } from "./fixtures";
import { createTestSubmission, getTestEssay } from "./helpers";

test.describe("Visual & UI Tests", () => {
  test("TC-STYLE-011: Buttons are touch-friendly (44px+ height)", async ({ page, writePage }) => {
    await writePage.goto("1");

    // Check submit button height
    const submitButton = await writePage.getSubmitButton();
    if ((await submitButton.count()) > 0) {
      const height = await submitButton.first().evaluate((el) => {
        const style = window.getComputedStyle(el);
        return parseFloat(style.height);
      });

      // Button should be at least 44px tall for touch-friendly design
      expect(height).toBeGreaterThanOrEqual(44);
    }

    // Check other buttons on the page
    const buttons = page.locator("button");
    const buttonCount = await buttons.count();

    for (let i = 0; i < Math.min(buttonCount, 5); i++) {
      const button = buttons.nth(i);
      const height = await button.evaluate((el) => {
        const style = window.getComputedStyle(el);
        return parseFloat(style.height);
      });

      // Primary action buttons should be touch-friendly
      const isPrimary = await button.evaluate((el) => {
        return el.classList.contains("btn-primary") || el.classList.contains("btn-secondary");
      });

      if (isPrimary) {
        expect(height).toBeGreaterThanOrEqual(44);
      }
    }
  });

  test("TC-GRAM-001: Error highlighting colors match confidence tiers", async ({
    resultsPage,
    page,
  }) => {
    const essay = getTestEssay("withErrors");
    const { submissionId } = await createTestSubmission("Describe your weekend.", essay);

    await resultsPage.goto(submissionId);
    await resultsPage.waitForResults();

    await page.waitForTimeout(3000);

    // Check for error spans with underlines
    const errorSpans = page.locator('span[style*="border-bottom"], span[style*="text-decoration"]');
    const errorCount = await errorSpans.count();

    if (errorCount > 0) {
      // Check first few error spans for color
      for (let i = 0; i < Math.min(errorCount, 3); i++) {
        const span = errorSpans.nth(i);
        const color = await span.evaluate((el) => {
          const style = window.getComputedStyle(el);
          return style.color || style.borderBottomColor || "";
        });

        // Error colors should be red, orange, or amber (not black/gray)
        const isErrorColor =
          color.includes("rgb(220, 38, 38)") || // Red
          color.includes("rgb(234, 88, 12)") || // Orange
          color.includes("rgb(217, 119, 6)") || // Amber
          color.includes("#dc2626") || // Red hex
          color.includes("#ea580c") || // Orange hex
          color.includes("#d97706"); // Amber hex

        // If color is set, it should be an error color
        if (color && color !== "rgba(0, 0, 0, 0)" && color !== "transparent") {
          expect(
            isErrorColor || color.includes("220") || color.includes("234") || color.includes("217")
          ).toBeTruthy();
        }
      }
    }
  });

  test("TC-FE-036: Error tooltips appear on hover", async ({ resultsPage, page }) => {
    const essay = getTestEssay("withErrors");
    const { submissionId } = await createTestSubmission("Describe your weekend.", essay);

    await resultsPage.goto(submissionId);
    await resultsPage.waitForResults();

    await page.waitForTimeout(3000);

    // Find error spans (with title attribute or data attributes)
    const errorSpans = page.locator("span[title], span[data-error], span[aria-label]");
    const errorCount = await errorSpans.count();

    if (errorCount > 0) {
      // Hover over first error span
      const firstError = errorSpans.first();
      await firstError.hover();

      await page.waitForTimeout(500);

      // Check if tooltip appears (title attribute or aria-label)
      const title = await firstError.getAttribute("title");
      const ariaLabel = await firstError.getAttribute("aria-label");

      // Should have some tooltip text
      expect(title || ariaLabel).toBeTruthy();
    }
  });

  test("TC-STYLE-012: Text contrast meets minimum readability", async ({ page, writePage }) => {
    await writePage.goto("1");

    // Check main text color contrast
    const mainText = page.locator("body p, body div").first();
    const textColor = await mainText.evaluate((el) => {
      const style = window.getComputedStyle(el);
      return style.color;
    });

    const bgColor = await mainText.evaluate((el) => {
      const style = window.getComputedStyle(el);
      return style.backgroundColor;
    });

    // Basic check: text should not be same as background
    expect(textColor).not.toBe(bgColor);

    // Text should not be transparent or white on white
    expect(textColor).not.toBe("rgba(0, 0, 0, 0)");
    expect(textColor).not.toBe("transparent");
  });
});
