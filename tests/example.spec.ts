import { test, expect } from '@playwright/test';

test('homepage has Writeo title and LanguageTool demo', async ({ page }) => {
  await page.goto('/');
  await expect(page.locator('h1')).toContainText('Writeo');
  await expect(page.locator('main')).toContainText('LanguageTool Demo');
});
