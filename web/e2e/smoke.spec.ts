import { expect, test } from '@playwright/test'

test.beforeEach(async ({ page }) => {
  await page.goto('/')
  await expect(page.getByRole('heading', { name: 'Agent Orchestrator' })).toBeVisible()
})

test('loads shell and core navigation', async ({ page }) => {
  await expect(page.getByRole('button', { name: 'Board' })).toBeVisible()
  await expect(page.getByRole('button', { name: 'Execution' })).toBeVisible()
  await expect(page.getByRole('button', { name: 'Review Queue' })).toBeVisible()
  await expect(page.getByRole('button', { name: 'Agents' })).toBeVisible()
  await expect(page.getByRole('button', { name: 'Settings' })).toBeVisible()
})

test('creates a task from the create-work modal', async ({ page }) => {
  const title = `E2E task ${Date.now()}`

  await page.getByRole('button', { name: 'Create Work' }).click()
  const modal = page.getByRole('dialog', { name: /Create Work modal/i })
  await expect(modal).toBeVisible()

  await modal.getByLabel('Title').fill(title)
  await modal.locator('button[form="create-task-form"]').click()

  await expect(modal).toBeHidden()
  await expect(page.locator('.task-card .task-title', { hasText: title }).first()).toBeVisible({ timeout: 15_000 })
})

test('saves settings through the real API', async ({ page }) => {
  await page.getByRole('button', { name: 'Settings' }).click()
  await expect(page.getByRole('heading', { name: 'Settings' })).toBeVisible()

  const concurrency = page.getByLabel('Orchestrator concurrency')
  await concurrency.fill('4')
  await page.getByRole('button', { name: 'Save settings' }).click()

  await expect(page.getByText('Settings saved.')).toBeVisible({ timeout: 10_000 })
})
