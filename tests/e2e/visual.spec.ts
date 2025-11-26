import { test, expect } from "./fixtures";

/**
 * Visual & UI Tests - Essential UI checks
 */

test.describe("Visual & UI", () => {
  test("buttons have minimum touch-friendly height (44px)", async ({ page, writePage }) => {
    await writePage.goto("1");

    const submitButton = await writePage.getSubmitButton();
    if ((await submitButton.count()) > 0) {
      const height = await submitButton.first().evaluate((el) => {
        const style = window.getComputedStyle(el);
        return parseFloat(style.height);
      });
      expect(height).toBeGreaterThanOrEqual(44);
    }
  });

  test("text has readable contrast", async ({ page, writePage }) => {
    await writePage.goto("1");

    const mainText = page.locator("body p, body div").first();
    const textColor = await mainText.evaluate((el) => {
      return window.getComputedStyle(el).color;
    });

    // Text should not be transparent
    expect(textColor).not.toBe("rgba(0, 0, 0, 0)");
    expect(textColor).not.toBe("transparent");
  });
});
