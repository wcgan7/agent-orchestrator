import { defineConfig } from '@playwright/test'

const backendProjectDir = process.env.PLAYWRIGHT_PROJECT_DIR ?? '../.tmp/playwright-project'
const backendPort = 19080
const frontendPort = 19030

export default defineConfig({
  testDir: './e2e',
  fullyParallel: false,
  workers: 1,
  retries: process.env.CI ? 1 : 0,
  timeout: 45_000,
  reporter: process.env.CI ? 'github' : 'list',
  use: {
    baseURL: `http://127.0.0.1:${frontendPort}`,
    trace: 'retain-on-failure',
    screenshot: 'only-on-failure',
    video: 'retain-on-failure',
  },
  webServer: [
    {
      command: `mkdir -p ${backendProjectDir} && PYTHONPATH=../src python -c "from agent_orchestrator.cli import main; import sys; raise SystemExit(main(sys.argv[1:]))" --project-dir ${backendProjectDir} server --host 127.0.0.1 --port ${backendPort}`,
      url: `http://127.0.0.1:${backendPort}/`,
      timeout: 120_000,
      reuseExistingServer: false,
    },
    {
      command: `VITE_API_PROXY_TARGET=http://127.0.0.1:${backendPort} VITE_WS_PROXY_TARGET=ws://127.0.0.1:${backendPort} npm run dev -- --host 127.0.0.1 --port ${frontendPort}`,
      url: `http://127.0.0.1:${frontendPort}`,
      timeout: 120_000,
      reuseExistingServer: false,
    },
  ],
})
