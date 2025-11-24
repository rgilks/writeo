import { test, expect } from "./fixtures";

/**
 * Draft Persistence & Navigation Tests
 * Tests for persistent draft management and version history functionality
 * Based on the Technical Architecture Report requirements
 */

test.describe("Draft Persistence & Navigation", () => {
  test.beforeEach(async ({ page }) => {
    // Clear localStorage before each test to ensure clean state
    await page.goto("/write/1");
    await page.evaluate(() => {
      localStorage.removeItem("writeo-draft-store");
    });
  });

  test("Critical Path: Data survives page reload", async ({ writePage, page }) => {
    await writePage.goto("1");

    const editor = page.getByRole("textbox", { name: /answer/i });
    const uniqueText = `Draft content ${Date.now()}`;

    // 1. User types content
    await editor.fill(uniqueText);

    // Wait for auto-save (2 seconds debounce)
    await page.waitForTimeout(2500);

    // 2. Verify draft appears in sidebar (visual confirmation)
    const sidebar = page.locator("aside.draft-sidebar");
    await expect(sidebar).toBeVisible({ timeout: 3000 });
    await expect(sidebar).toContainText(uniqueText.slice(0, 20), { timeout: 3000 });

    // 3. ACTION: Reload the page
    await page.reload();

    // Wait for hydration
    await page.waitForTimeout(1000);

    // 4. ASSERTION: Content is back in the editor
    const editorAfterReload = page.getByRole("textbox", { name: /answer/i });
    await expect(editorAfterReload).toHaveValue(uniqueText, { timeout: 3000 });

    // 5. ASSERTION: Draft is still in the sidebar
    const sidebarAfterReload = page.locator("aside.draft-sidebar");
    await expect(sidebarAfterReload).toBeVisible();
    await expect(sidebarAfterReload).toContainText(uniqueText.slice(0, 20));
  });

  test("Navigation: Switching between drafts works", async ({ writePage, page }) => {
    await writePage.goto("1");

    const editor = page.getByRole("textbox", { name: /answer/i });
    const newButton = page.getByRole("button", { name: "+ New" });

    // Create Draft A
    const draftA = "Draft Alpha content for testing navigation";
    await editor.fill(draftA);
    await page.waitForTimeout(2500); // Wait for auto-save

    // Create Draft B
    await newButton.click();
    await page.waitForTimeout(500);
    await expect(editor).toHaveValue(""); // Verify clear

    const draftB = "Draft Beta content for testing navigation";
    await editor.fill(draftB);
    await page.waitForTimeout(2500); // Wait for auto-save

    // Click Draft Alpha in sidebar
    const sidebar = page.locator("aside.draft-sidebar");
    await sidebar.getByText("Draft Alpha").click();
    await page.waitForTimeout(500);
    await expect(editor).toHaveValue(draftA);

    // Click Draft Beta in sidebar
    await sidebar.getByText("Draft Beta").click();
    await page.waitForTimeout(500);
    await expect(editor).toHaveValue(draftB);
  });

  test("Auto-save: Draft is created automatically after typing pause", async ({
    writePage,
    page,
  }) => {
    await writePage.goto("1");

    const editor = page.getByRole("textbox", { name: /answer/i });
    const sidebar = page.locator("aside.draft-sidebar");

    // Initially no drafts
    await expect(sidebar.getByText("No drafts yet")).toBeVisible();

    // Type content
    const testContent = "This is test content that should auto-save after a pause";
    await editor.fill(testContent);

    // Wait for auto-save debounce (2 seconds)
    await page.waitForTimeout(2500);

    // Draft should appear in sidebar
    await expect(sidebar.getByText("No drafts yet")).not.toBeVisible({ timeout: 1000 });
    await expect(sidebar).toContainText(testContent.slice(0, 20));
  });

  test("New Draft: Creating new draft clears editor", async ({ writePage, page }) => {
    await writePage.goto("1");

    const editor = page.getByRole("textbox", { name: /answer/i });
    const newButton = page.getByRole("button", { name: "+ New" });

    // Type some content
    await editor.fill("Some existing content");
    await page.waitForTimeout(2500); // Wait for auto-save

    // Click New button
    await newButton.click();
    await page.waitForTimeout(500);

    // Editor should be cleared
    await expect(editor).toHaveValue("");

    // Active draft indicator should be gone
    const autoSavedIndicator = page.getByText("✓ Auto-saved");
    const indicatorCount = await autoSavedIndicator.count();
    expect(indicatorCount).toBe(0);
  });

  test("Delete Draft: Deleting draft removes it from sidebar", async ({ writePage, page }) => {
    await writePage.goto("1");

    const editor = page.getByRole("textbox", { name: /answer/i });
    const sidebar = page.locator("aside.draft-sidebar");

    // Create a draft
    const testContent = "Content to be deleted";
    await editor.fill(testContent);
    await page.waitForTimeout(2500); // Wait for auto-save

    // Verify draft appears
    await expect(sidebar).toContainText(testContent.slice(0, 20));

    // Find and click delete button (×)
    const draftCard = sidebar
      .locator("div")
      .filter({ hasText: testContent.slice(0, 20) })
      .first();
    const deleteButton = draftCard.locator("button").filter({ hasText: "×" }).first();

    // Accept confirmation dialog
    page.once("dialog", (dialog) => {
      dialog.accept();
    });

    await deleteButton.click();
    await page.waitForTimeout(500);

    // Draft should be removed
    await expect(sidebar).not.toContainText(testContent.slice(0, 20), { timeout: 2000 });
  });

  test("Metadata Display: Draft sidebar shows timestamp and word count", async ({
    writePage,
    page,
  }) => {
    await writePage.goto("1");

    const editor = page.getByRole("textbox", { name: /answer/i });
    const sidebar = page.locator("aside.draft-sidebar");

    // Create a draft with known word count
    const testContent = "One two three four five six seven eight nine ten";
    await editor.fill(testContent);
    await page.waitForTimeout(2500); // Wait for auto-save

    // Check for word count display (should show "10 words")
    await expect(sidebar.getByText(/10 words/)).toBeVisible({ timeout: 3000 });

    // Check for timestamp (relative time like "Just now" or "Xm ago")
    const timePattern = /(Just now|\d+m ago|\d+h ago|\d+d ago)/;
    await expect(sidebar.getByText(timePattern)).toBeVisible({ timeout: 3000 });
  });

  test("Active Draft Indicator: Active draft is highlighted in sidebar", async ({
    writePage,
    page,
  }) => {
    await writePage.goto("1");

    const editor = page.getByRole("textbox", { name: /answer/i });
    const sidebar = page.locator("aside.draft-sidebar");
    const newButton = page.getByRole("button", { name: "+ New" });

    // Create Draft A
    await editor.fill("Draft A content");
    await page.waitForTimeout(2500);

    // Create Draft B
    await newButton.click();
    await page.waitForTimeout(500);
    await editor.fill("Draft B content");
    await page.waitForTimeout(2500);

    // Draft B should be active (highlighted)
    const draftBCard = sidebar.locator("div").filter({ hasText: "Draft B" }).first();
    const draftBStyle = await draftBCard.evaluate((el) => {
      const styles = window.getComputedStyle(el);
      return {
        backgroundColor: styles.backgroundColor,
        borderColor: styles.borderColor,
      };
    });

    // Active draft should have blue background or border
    const isHighlighted =
      draftBStyle.backgroundColor.includes("219") || // rgb(219, 234, 254) = #dbeafe
      draftBStyle.borderColor.includes("37") || // rgb(37, 99, 235) = #2563eb
      draftBStyle.borderColor.includes("59"); // Alternative blue

    expect(isHighlighted).toBe(true);
  });

  test("Resilience: Multiple drafts persist across reloads", async ({ writePage, page }) => {
    await writePage.goto("1");

    const editor = page.getByRole("textbox", { name: /answer/i });
    const sidebar = page.locator("aside.draft-sidebar");
    const newButton = page.getByRole("button", { name: "+ New" });

    // Create multiple drafts
    const drafts = ["First draft", "Second draft", "Third draft"];
    for (const draftContent of drafts) {
      await editor.fill(draftContent);
      await page.waitForTimeout(2500); // Wait for auto-save
      if (draftContent !== drafts[drafts.length - 1]) {
        await newButton.click();
        await page.waitForTimeout(500);
      }
    }

    // Verify all drafts are in sidebar
    for (const draftContent of drafts) {
      await expect(sidebar).toContainText(draftContent.slice(0, 10));
    }

    // Reload page
    await page.reload();
    await page.waitForTimeout(1000);

    // Verify all drafts still exist
    const sidebarAfterReload = page.locator("aside.draft-sidebar");
    for (const draftContent of drafts) {
      await expect(sidebarAfterReload).toContainText(draftContent.slice(0, 10), {
        timeout: 3000,
      });
    }
  });

  test("Keyboard Navigation: Drafts can be selected with keyboard", async ({ writePage, page }) => {
    await writePage.goto("1");

    const editor = page.getByRole("textbox", { name: /answer/i });
    const sidebar = page.locator("aside.draft-sidebar");

    // Create two drafts
    await editor.fill("First draft for keyboard test");
    await page.waitForTimeout(2500);

    await page.getByRole("button", { name: "+ New" }).click();
    await page.waitForTimeout(500);
    await editor.fill("Second draft for keyboard test");
    await page.waitForTimeout(2500);

    // Tab to first draft in sidebar
    await page.keyboard.press("Tab");
    await page.keyboard.press("Tab"); // May need multiple tabs to reach sidebar

    // Use arrow keys or Enter to select
    // Note: This test may need adjustment based on actual keyboard navigation implementation
    const firstDraft = sidebar.getByText("First draft").first();
    await firstDraft.focus();
    await page.keyboard.press("Enter");
    await page.waitForTimeout(500);

    // Content should load
    await expect(editor).toHaveValue("First draft for keyboard test");
  });
});
