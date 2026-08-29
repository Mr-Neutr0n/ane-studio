import { mkdtemp, rm } from "node:fs/promises";
import { createRequire } from "node:module";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { _electron as electron, expect, test } from "@playwright/test";

const electronExecutable = createRequire(import.meta.url)("electron") as string;
let dataDirectory: string;

test.beforeEach(async () => {
  dataDirectory = await mkdtemp(join(tmpdir(), "ane-studio-e2e-"));
});

test.afterEach(async () => {
  await rm(dataDirectory, { recursive: true, force: true });
});

test("opens the local model lab on supported hardware", async () => {
  test.setTimeout(60_000);
  const application = await electron.launch({
    executablePath: electronExecutable,
    args: ["."],
    env: { ...process.env, ANE_STUDIO_HOME: dataDirectory },
  });
  try {
    const window = await application.firstWindow();
    await expect(window.getByText("LFM2.5 230M", { exact: true }).first()).toBeVisible({
      timeout: 30_000,
    });
    await expect(window.getByText("Local only", { exact: true })).toBeVisible();
    await expect(window.getByRole("button", { name: /Convert to Core ML/ })).toBeVisible();
    await expect(window.getByText(/Apple M5/)).toBeVisible();
  } finally {
    await application.close();
  }
});
