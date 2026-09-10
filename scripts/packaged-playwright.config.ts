import { defineConfig } from "@playwright/test";

export default defineConfig({
  testDir: "./packaged-tests",
  timeout: 90_000,
  workers: 1,
  outputDir: "../artifacts/packaged-test-results",
  reporter: "line",
  use: { trace: "retain-on-failure" },
});
