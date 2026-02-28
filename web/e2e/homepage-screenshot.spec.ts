import { expect, test, type APIRequestContext } from '@playwright/test'
import { execFileSync } from 'node:child_process'
import path from 'node:path'

type TaskResponse = { task: { id: string } }
type TaskPayload = TaskResponse['task'] & { [key: string]: unknown }
type BoardStage = 'backlog' | 'queued' | 'in_progress' | 'in_review' | 'blocked' | 'done'

async function createTask(
  request: APIRequestContext,
  {
    title,
    description,
    priority = 'P2',
    status = 'queued',
    hitlMode = 'autopilot',
    metadata = {},
  }: {
    title: string
    description: string
    priority?: 'P0' | 'P1' | 'P2' | 'P3'
    status?: BoardStage
    hitlMode?: 'autopilot' | 'supervised' | 'review_only'
    metadata?: Record<string, unknown>
  },
): Promise<TaskPayload> {
  const response = await request.post('/api/tasks', {
    data: {
      title,
      description,
      task_type: 'feature',
      priority,
      status,
      labels: ['readme', 'screenshot'],
      blocked_by: [],
      hitl_mode: hitlMode,
      metadata,
    },
  })
  expect(response.ok()).toBeTruthy()
  const payload = (await response.json()) as TaskResponse
  return payload.task
}

function forceSetPendingGate(projectDir: string, taskId: string, gate: string): void {
  const dbPath = path.join(projectDir, '.agent_orchestrator', 'runtime.db')
  const script = `
import datetime
import json
import sqlite3
import sys

db_path, task_id, gate = sys.argv[1:4]
conn = sqlite3.connect(db_path)
cur = conn.cursor()
row = cur.execute("SELECT payload FROM tasks WHERE id = ?", (task_id,)).fetchone()
if row is None:
    raise SystemExit(f"task not found: {task_id}")
payload = json.loads(row[0])
payload["pending_gate"] = gate
payload["status"] = "in_progress"
payload["current_step"] = "plan"
payload["hitl_mode"] = "supervised"
updated_at = datetime.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"
payload["updated_at"] = updated_at
cur.execute(
    "UPDATE tasks SET status = ?, pending_gate = ?, updated_at = ?, payload = ? WHERE id = ?",
    ("in_progress", gate, updated_at, json.dumps(payload, separators=(",", ":")), task_id),
)
conn.commit()
conn.close()
`.trim()
  execFileSync('python3', ['-c', script, dbPath, taskId, gate], {
    stdio: ['ignore', 'pipe', 'pipe'],
    encoding: 'utf-8',
  })
}

async function pauseOrchestrator(request: APIRequestContext): Promise<void> {
  const response = await request.post('/api/orchestrator/control', {
    data: { action: 'pause' },
  })
  expect(response.ok()).toBeTruthy()
}

test('captures homepage screenshot with task detail over board context', async ({ page, request }) => {
  test.setTimeout(90_000)

  await page.setViewportSize({ width: 1600, height: 960 })
  await pauseOrchestrator(request)
  const projectDir = path.resolve(process.cwd(), process.env.PLAYWRIGHT_PROJECT_DIR ?? '../.tmp/playwright-screenshot')

  await createTask(request, {
    title: 'Define tenant isolation model',
    description: 'Capture project-level boundaries before implementation.',
    priority: 'P1',
    status: 'backlog',
  })
  await createTask(request, {
    title: 'Prepare API contract for queue metrics',
    description: 'Document fields required by the execution panel.',
    status: 'queued',
  })
  await createTask(request, {
    title: 'Implement websocket burst coalescing',
    description: 'Avoid over-refreshing under sustained event throughput.',
    priority: 'P0',
    status: 'in_progress',
  })
  const planGateTask = await createTask(request, {
    title: 'Plan API pagination migration',
    description: 'Plan completed and waiting for implementation approval.',
    priority: 'P1',
    status: 'in_progress',
    hitlMode: 'supervised',
  })
  forceSetPendingGate(projectDir, String(planGateTask.id), 'before_implement')
  await createTask(request, {
    title: 'Implement dependency dropdown task form UX',
    description: [
      'Feature pipeline: plan -> implement -> verify -> review -> commit.',
      'Plan approved and implementation complete. Awaiting human review before commit.',
    ].join(' '),
    priority: 'P1',
    status: 'in_review',
    hitlMode: 'supervised',
    metadata: {
      human_review_actions: [
        {
          action: 'request_changes',
          ts: '2026-02-28T08:04:00Z',
          guidance: 'Please improve field-level helper copy and preserve keyboard accessibility semantics.',
        },
      ],
    },
  })
  await createTask(request, {
    title: 'Stabilize multi-project event filtering',
    description: 'Waiting for schema update from backend events payload.',
    priority: 'P1',
    status: 'blocked',
  })
  await createTask(request, {
    title: 'Fix board pane overflow behavior',
    description: 'Grid now avoids clipping at medium desktop widths.',
    status: 'done',
  })

  await page.goto('/')
  await expect(page.getByRole('heading', { name: 'Agent Orchestrator' })).toBeVisible()
  await expect(page.getByRole('heading', { name: 'Board' })).toBeVisible()
  const compactToggle = page.getByRole('switch', { name: 'Compact' })
  if (await compactToggle.isChecked()) {
    await compactToggle.click()
  }

  const inProgressColumn = page.locator('.board-col').filter({ has: page.getByRole('heading', { name: 'In Progress' }) })
  const inReviewColumn = page.locator('.board-col').filter({ has: page.getByRole('heading', { name: 'In Review' }) })
  const planGateCard = inProgressColumn.locator('.task-card').filter({ hasText: 'Plan API pagination migration' })
  await expect(planGateCard).toBeVisible()
  await expect(planGateCard.getByText('Awaiting Approval')).toBeVisible()
  await expect(inReviewColumn.getByText('Implement dependency dropdown task form UX')).toBeVisible()
  await inReviewColumn.getByText('Implement dependency dropdown task form UX').click()

  await expect(page.locator('.detail-card')).toBeVisible()
  await expect(page.locator('.detail-card').getByText('Feature pipeline: plan -> implement -> verify -> review -> commit.')).toBeVisible()
  await expect(page.locator('.detail-card').getByText('Review history')).toBeVisible()
  await expect(page.locator('.detail-card').getByText('Changes requested')).toBeVisible()
  await expect(page.getByRole('button', { name: 'Overview' })).toBeVisible()
  await expect(page.getByRole('button', { name: 'Workdoc' })).toBeVisible()
  await expect(page.getByRole('button', { name: 'Logs' })).toBeVisible()

  await page.screenshot({
    path: 'public/homepage-screenshot.png',
    fullPage: false,
  })
})
