import { expect, test, type APIRequestContext } from '@playwright/test'
import fs from 'node:fs/promises'
import path from 'node:path'

type TaskResponse = { task: { id: string } }
type TaskPayload = TaskResponse['task'] & { [key: string]: unknown }
type BoardStage = 'backlog' | 'queued' | 'in_progress' | 'in_review' | 'blocked' | 'done' | 'cancelled'

async function createTask(
  request: APIRequestContext,
  title: string,
  description: string,
  priority: 'P0' | 'P1' | 'P2' | 'P3' = 'P2',
  status: 'backlog' | 'queued' = 'queued',
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
      approval_mode: 'human_review',
      hitl_mode: 'autopilot',
    },
  })
  expect(response.ok()).toBeTruthy()
  const payload = (await response.json()) as TaskResponse
  return payload.task
}

async function pauseOrchestrator(request: APIRequestContext): Promise<void> {
  const response = await request.post('/api/orchestrator/control', {
    data: { action: 'pause' },
  })
  expect(response.ok()).toBeTruthy()
}

async function writeSeededBoardState(tasks: TaskPayload[], statuses: Record<string, BoardStage>): Promise<void> {
  const projectDir = process.env.PLAYWRIGHT_PROJECT_DIR ?? '../.tmp/playwright-project'
  const tasksPath = path.resolve(process.cwd(), projectDir, '.agent_orchestrator', 'tasks.yaml')
  const seeded = tasks.map((task) => ({
    ...task,
    status: statuses[task.id] ?? task.status,
    current_step: null,
    current_agent_id: null,
    pending_gate: null,
    run_ids: [],
    error: null,
    metadata: {},
  }))
  const payload = JSON.stringify({ version: 3, tasks: seeded }, null, 2)
  await fs.writeFile(tasksPath, payload, 'utf-8')
}

test('captures seeded homepage screenshot with varied board stages', async ({ page, request }) => {
  test.setTimeout(90_000)

  await page.setViewportSize({ width: 1600, height: 960 })
  await pauseOrchestrator(request)

  const backlogTaskA = await createTask(
    request,
    'Define tenant isolation model',
    'Capture project-level boundaries before implementation.',
    'P1',
    'backlog',
  )
  const backlogTaskB = await createTask(
    request,
    'Draft release notes outline',
    'Keep one backlog item for the screenshot narrative.',
    'P3',
    'backlog',
  )

  const queuedTask = await createTask(
    request,
    'Prepare API contract for queue metrics',
    'Document fields required by the execution panel.',
    'P2',
    'queued',
  )

  const inProgressTask = await createTask(
    request,
    'Implement websocket burst coalescing',
    'Avoid over-refreshing under sustained event throughput.',
    'P0',
    'queued',
  )

  const inReviewTask = await createTask(
    request,
    'Add keyboard support to mode selector',
    'Ensure listbox interactions are fully accessible.',
    'P1',
    'queued',
  )

  const blockedTask = await createTask(
    request,
    'Stabilize multi-project event filtering',
    'Waiting for schema update from backend events payload.',
    'P1',
    'queued',
  )

  const doneTask = await createTask(
    request,
    'Fix board pane overflow behavior',
    'Grid now avoids clipping at medium desktop widths.',
    'P2',
    'queued',
  )
  await writeSeededBoardState(
    [backlogTaskA, backlogTaskB, queuedTask, inProgressTask, inReviewTask, blockedTask, doneTask],
    {
      [backlogTaskA.id]: 'backlog',
      [backlogTaskB.id]: 'backlog',
      [queuedTask.id]: 'queued',
      [inProgressTask.id]: 'in_progress',
      [inReviewTask.id]: 'in_review',
      [blockedTask.id]: 'blocked',
      [doneTask.id]: 'done',
    },
  )

  await page.goto('/')
  await expect(page.getByRole('heading', { name: 'Agent Orchestrator' })).toBeVisible()
  await expect(page.getByRole('heading', { name: 'Board' })).toBeVisible()

  const backlogColumn = page.locator('.board-col').filter({ has: page.getByRole('heading', { name: 'Backlog' }) })
  const queuedColumn = page.locator('.board-col').filter({ has: page.getByRole('heading', { name: 'Queued' }) })
  const inProgressColumn = page.locator('.board-col').filter({ has: page.getByRole('heading', { name: 'In Progress' }) })
  const inReviewColumn = page.locator('.board-col').filter({ has: page.getByRole('heading', { name: 'In Review' }) })
  const blockedColumn = page.locator('.board-col').filter({ has: page.getByRole('heading', { name: 'Blocked' }) })
  const doneColumn = page.locator('.board-col').filter({ has: page.getByRole('heading', { name: 'Done' }) })

  await expect(backlogColumn.getByText('Define tenant isolation model')).toBeVisible()
  await expect(queuedColumn.getByText('Prepare API contract for queue metrics')).toBeVisible()
  await expect(inProgressColumn.getByText('Implement websocket burst coalescing')).toBeVisible()
  await expect(inReviewColumn.getByText('Add keyboard support to mode selector')).toBeVisible()
  await expect(blockedColumn.getByText('Stabilize multi-project event filtering')).toBeVisible()
  await expect(doneColumn.getByText('Fix board pane overflow behavior')).toBeVisible()

  await page.screenshot({
    path: 'public/homepage-screenshot.png',
    fullPage: false,
  })
})
