import { expect, test, type APIRequestContext } from '@playwright/test'

type TaskResponse = { task: { id: string } }
type BoardStage = 'ready' | 'in_progress' | 'in_review' | 'blocked' | 'done'

async function createTask(
  request: APIRequestContext,
  title: string,
  description: string,
  priority: 'P0' | 'P1' | 'P2' | 'P3' = 'P2',
): Promise<string> {
  const response = await request.post('/api/tasks', {
    data: {
      title,
      description,
      task_type: 'feature',
      priority,
      labels: ['readme', 'screenshot'],
      blocked_by: [],
      approval_mode: 'human_review',
      hitl_mode: 'autopilot',
    },
  })
  expect(response.ok()).toBeTruthy()
  const payload = (await response.json()) as TaskResponse
  return payload.task.id
}

async function transitionTask(
  request: APIRequestContext,
  taskId: string,
  status: BoardStage,
): Promise<void> {
  const response = await request.post(`/api/tasks/${taskId}/transition`, {
    data: { status },
  })
  expect(response.ok()).toBeTruthy()
}

async function moveTaskTo(
  request: APIRequestContext,
  taskId: string,
  target: BoardStage,
): Promise<void> {
  const pathToStatus: Record<BoardStage, BoardStage[]> = {
    ready: ['ready'],
    in_progress: ['ready', 'in_progress'],
    in_review: ['ready', 'in_progress', 'in_review'],
    blocked: ['ready', 'blocked'],
    done: ['ready', 'in_progress', 'in_review', 'done'],
  }

  for (const status of pathToStatus[target]) {
    await transitionTask(request, taskId, status)
  }
}

test('captures seeded homepage screenshot with varied board stages', async ({ page, request }) => {
  test.setTimeout(90_000)

  await page.setViewportSize({ width: 1600, height: 960 })

  await createTask(
    request,
    'Define tenant isolation model',
    'Capture project-level boundaries before implementation.',
    'P1',
  )
  await createTask(
    request,
    'Draft release notes outline',
    'Keep one backlog item for the screenshot narrative.',
    'P3',
  )

  const readyTaskId = await createTask(
    request,
    'Prepare API contract for queue metrics',
    'Document fields required by the execution panel.',
    'P2',
  )
  await moveTaskTo(request, readyTaskId, 'ready')

  const inProgressTaskId = await createTask(
    request,
    'Implement websocket burst coalescing',
    'Avoid over-refreshing under sustained event throughput.',
    'P0',
  )
  await moveTaskTo(request, inProgressTaskId, 'in_progress')

  const inReviewTaskId = await createTask(
    request,
    'Add keyboard support to mode selector',
    'Ensure listbox interactions are fully accessible.',
    'P1',
  )
  await moveTaskTo(request, inReviewTaskId, 'in_review')

  const blockedTaskId = await createTask(
    request,
    'Stabilize multi-project event filtering',
    'Waiting for schema update from backend events payload.',
    'P1',
  )
  await moveTaskTo(request, blockedTaskId, 'blocked')

  const doneTaskId = await createTask(
    request,
    'Fix board pane overflow behavior',
    'Grid now avoids clipping at medium desktop widths.',
    'P2',
  )
  await moveTaskTo(request, doneTaskId, 'done')

  await page.goto('/')
  await expect(page.getByRole('heading', { name: 'Agent Orchestrator' })).toBeVisible()
  await expect(page.getByRole('heading', { name: 'Board' })).toBeVisible()

  const backlogColumn = page.locator('.board-col').filter({ has: page.getByRole('heading', { name: 'Backlog' }) })
  const readyColumn = page.locator('.board-col').filter({ has: page.getByRole('heading', { name: 'Ready' }) })
  const inProgressColumn = page.locator('.board-col').filter({ has: page.getByRole('heading', { name: 'In Progress' }) })
  const inReviewColumn = page.locator('.board-col').filter({ has: page.getByRole('heading', { name: 'In Review' }) })
  const blockedColumn = page.locator('.board-col').filter({ has: page.getByRole('heading', { name: 'Blocked' }) })
  const doneColumn = page.locator('.board-col').filter({ has: page.getByRole('heading', { name: 'Done' }) })

  await expect(backlogColumn.getByText('Define tenant isolation model')).toBeVisible()
  await expect(readyColumn.getByText('Prepare API contract for queue metrics')).toBeVisible()
  await expect(inProgressColumn.getByText('Implement websocket burst coalescing')).toBeVisible()
  await expect(inReviewColumn.getByText('Add keyboard support to mode selector')).toBeVisible()
  await expect(blockedColumn.getByText('Stabilize multi-project event filtering')).toBeVisible()
  await expect(doneColumn.getByText('Fix board pane overflow behavior')).toBeVisible()

  await page.screenshot({
    path: 'public/homepage-screenshot.png',
    fullPage: false,
  })
})
