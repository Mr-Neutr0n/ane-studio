import { defineConfig, devices } from "@playwright/test";

export default defineConfig({
  testDir: "./tests",
  testMatch: "**/*.spec.ts",
  timeout: 20_000,
  expect: { timeout: 5_000 },
  fullyParallel: true,
  workers: 2,
  reporter: [["list"], ["html", { open: "never" }]],
  use: {
    baseURL: "http://127.0.0.1:4187",
    trace: "retain-on-failure",
    screenshot: "only-on-failure",
  },
  webServer: {
    command: "bun scripts/serve.ts",
    cwd: import.meta.dir,
    url: "http://127.0.0.1:4187",
    env: { PORT: "4187" },
    reuseExistingServer: false,
    timeout: 10_000,
  },
  projects: [
    {
      name: "desktop-chromium",
      use: { ...devices["Desktop Chrome"], viewport: { width: 1440, height: 1000 } },
    },
    {
      name: "mobile-chromium",
      use: { ...devices["Pixel 7"], viewport: { width: 390, height: 844 } },
    },
    {
      name: "small-mobile",
      use: {
        ...devices["Desktop Chrome"],
        viewport: { width: 320, height: 720 },
        isMobile: true,
        hasTouch: true,
      },
    },
    { name: "mobile-webkit", use: { ...devices["iPhone 13"] } },
  ],
});
