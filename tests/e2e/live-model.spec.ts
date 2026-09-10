import { mkdir } from "node:fs/promises";
import { resolve } from "node:path";
import { _electron as electron, expect, test } from "@playwright/test";
import { hasPassedParity } from "../../src/shared/contracts";
import { electronExecutable } from "./ui-fixtures";

// Opt-in integration evidence, never a fixture. Use an isolated, already validated
// model directory and pause other Core ML jobs before running this test. This test
// does not prepare models, remove data, or benchmark. Normal verify skips it.
// ANE_STUDIO_LIVE_HOME=/tmp/isolated-ready-models
// ANE_STUDIO_LIVE_MODEL=organization/model
// ANE_STUDIO_CAPTURE_ASSETS=1 writes current, real chat/details screenshots.
const liveHome = process.env.ANE_STUDIO_LIVE_HOME;
const modelId = process.env.ANE_STUDIO_LIVE_MODEL;

// biome-ignore lint/correctness/noEmptyPattern: Playwright requires a destructured fixture parameter.
test("opt-in real local model chat", async ({}, testInfo) => {
  test.skip(
    !liveHome || !modelId,
    "Set ANE_STUDIO_LIVE_HOME and ANE_STUDIO_LIVE_MODEL for real offline inference.",
  );
  test.setTimeout(300_000);
  if (!liveHome || !modelId) return;
  const application = await electron.launch({
    executablePath: electronExecutable,
    args: ["."],
    env: {
      ...process.env,
      ANE_STUDIO_HOME: resolve(liveHome),
      HF_TOKEN: "",
      HUGGING_FACE_HUB_TOKEN: "",
      HF_HUB_OFFLINE: "1",
      TRANSFORMERS_OFFLINE: "1",
    },
  });
  try {
    const page = await application.firstWindow();
    await expect(page.getByRole("button", { name: /Credentials/ })).toBeVisible({
      timeout: 45_000,
    });
    const bootstrap = await page.evaluate(async () => await window.studio.bootstrap());
    const record = bootstrap.models.find((item) => item.id === modelId);
    expect(record?.state).toBe("ready");
    expect(hasPassedParity(record?.parity)).toBe(true);
    expect(bootstrap.dataDirectory).toBe(resolve(liveHome));
    await page.getByTitle(modelId, { exact: true }).click();
    await page.getByLabel("Requested policy").selectOption("cpuOnly");
    const prompt = "Name one planet.";
    await page.getByLabel("Message the local model").fill(prompt);
    await page.getByRole("button", { name: "Send message" }).click();
    await expect(page.getByLabel("Message the local model")).toBeEnabled({ timeout: 240_000 });
    await expect(page.getByRole("alert")).not.toBeVisible();
    const reply = page.locator(".message.assistant p").last();
    await expect(reply).not.toHaveText("");
    const text = await reply.innerText();
    await expect(page.locator(".generation-note")).toBeVisible();
    const termination = await page.locator(".generation-note").innerText();
    await testInfo.attach("real-local-chat-evidence", {
      contentType: "application/json",
      body: JSON.stringify(
        {
          modelId,
          revision: record?.revision,
          parity: record?.parity,
          requestedPolicy: "cpuOnly",
          observedPlacement: "not captured",
          prompt,
          text,
          termination,
        },
        null,
        2,
      ),
    });
    if (process.env.ANE_STUDIO_CAPTURE_ASSETS === "1") {
      const assets = resolve("docs/assets");
      await mkdir(assets, { recursive: true });
      await page.screenshot({ path: resolve(assets, "ane-chat.png"), scale: "css" });
      await page.getByRole("button", { name: "Details", exact: true }).click();
      await page.screenshot({ path: resolve(assets, "ane-details.png"), scale: "css" });
    }
  } finally {
    await application.close();
  }
});
