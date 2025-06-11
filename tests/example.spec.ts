import { test, expect } from '@playwright/test';

test('homepage has "Welcome to Writeo"', async ({ page }) => {
  await page.goto('/');
  await expect(page.locator('main')).toContainText('Welcome to Writeo');
});
