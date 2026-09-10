import { mkdtemp, readFile, rm, stat } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { _electron as electron, expect, test as realTest } from "@playwright/test";
import type { BenchmarkReport, ComputePolicy } from "../../src/shared/contracts";
import {
  bootstrapFixture,
  configureFixture,
  electronExecutable,
  emitEvent,
  finishRequest,
  firstModelId,
  fixtureCalls,
  loadReadyModels,
  modelRecord,
  secondModelId,
  test,
} from "./ui-fixtures";

// Real Electron main/preload/renderer and real local Python health/catalog.
// No downloads, conversion, inference, or hardcoded chip generation.
realTest("real worker startup and renderer isolation", async () => {
  realTest.skip(
    process.platform !== "darwin" || process.arch !== "arm64",
    "Real Core ML worker needs Apple Silicon.",
  );
  realTest.setTimeout(60_000);
  const dataDirectory = await mkdtemp(join(tmpdir(), "ane-real-startup-"));
  const application = await electron.launch({
    executablePath: electronExecutable,
    args: ["."],
    env: {
      ...process.env,
      ANE_STUDIO_HOME: dataDirectory,
      HF_TOKEN: "",
      HUGGING_FACE_HUB_TOKEN: "",
    },
  });
  try {
    const page = await application.firstWindow();
    await expect(page.getByRole("heading", { name: "LFM2.5 230M", exact: true })).toBeVisible({
      timeout: 45_000,
    });
    await expect(page).toHaveTitle("ANE Studio");
    await expect(page.getByText("Local only", { exact: true })).toBeVisible();
    const hardware = await page.evaluate(async () => (await window.studio.bootstrap()).hardware);
    await expect(page.getByText(hardware.chip, { exact: true })).toBeVisible();
    await expect(page.getByRole("button", { name: /Convert (?:to Core ML|again)/ })).toBeVisible();
    expect(
      await page.evaluate(() => ({
        node: typeof Reflect.get(window, "require"),
        process: typeof Reflect.get(window, "process"),
      })),
    ).toEqual({ node: "undefined", process: "undefined" });
    expect(await page.evaluate(async () => (await window.studio.bootstrap()).dataDirectory)).toBe(
      dataDirectory,
    );
  } finally {
    await application.close();
    await rm(dataDirectory, { recursive: true, force: true });
  }
});

realTest("real credentials stay encrypted and are not returned to the renderer", async () => {
  realTest.skip(
    process.platform !== "darwin" || process.arch !== "arm64",
    "Real safeStorage/worker test needs Apple Silicon.",
  );
  realTest.setTimeout(60_000);
  const dataDirectory = await mkdtemp(join(tmpdir(), "ane-real-credentials-"));
  const application = await electron.launch({
    executablePath: electronExecutable,
    args: ["."],
    env: {
      ...process.env,
      ANE_STUDIO_HOME: dataDirectory,
      HF_TOKEN: "",
      HUGGING_FACE_HUB_TOKEN: "",
    },
  });
  try {
    realTest.skip(
      !(await application.evaluate(({ safeStorage }) => safeStorage.isEncryptionAvailable())),
      "OS credential encryption is unavailable.",
    );
    const page = await application.firstWindow();
    await expect(page.getByRole("button", { name: /Credentials/ })).toBeVisible({
      timeout: 45_000,
    });
    await page.getByRole("button", { name: /Credentials/ }).click();
    const fakeToken = `hf_${"AneE2EFixture".repeat(3)}`;
    await page.getByLabel("Access token").fill(fakeToken);
    await page.getByRole("button", { name: "Save token", exact: true }).click();
    await expect(page.getByRole("dialog")).not.toBeVisible();
    const credentialPath = join(dataDirectory, "credentials", "hf-token");
    const encrypted = await readFile(credentialPath);
    expect(encrypted.includes(Buffer.from(fakeToken))).toBe(false);
    expect((await stat(credentialPath)).mode & 0o777).toBe(0o600);
    const bootstrap = await page.evaluate(async () => await window.studio.bootstrap());
    expect(bootstrap.hasToken).toBe(true);
    expect(JSON.stringify(bootstrap).includes(fakeToken)).toBe(false);
    expect(await page.locator("body").innerText()).not.toContain(fakeToken);
    await page.getByRole("button", { name: /Credentials/ }).click();
    await expect(page.getByLabel("Access token")).toHaveValue("");
    await page.getByRole("button", { name: "Remove token" }).click();
    await expect(page.getByRole("dialog")).not.toBeVisible();
    expect((await page.evaluate(async () => await window.studio.bootstrap())).hasToken).toBe(false);
    await expect(stat(credentialPath)).rejects.toMatchObject({ code: "ENOENT" });
  } finally {
    await application.close();
    await rm(dataDirectory, { recursive: true, force: true });
  }
});

// The tests below drive the shipped Electron UI through deterministic main IPC
// fixtures. Their parity, throughput, and placement values are not model evidence.
test("offline UI starts, navigates, and fits the minimum window", async ({ studio }) => {
  const { page, application } = studio;
  await expect(page).toHaveTitle("ANE Studio");
  await expect(page.getByText("Apple Silicon UI fixture", { exact: true })).toBeVisible();
  await expect(page.getByLabel("Message the local model")).toBeDisabled();
  await page.getByRole("button", { name: "Tests", exact: true }).click();
  await expect(page.getByRole("button", { name: "Run all tests" })).toBeDisabled();
  await page.getByRole("button", { name: "Details", exact: true }).click();
  await expect(page.getByText("Lfm2ForCausalLM", { exact: true })).toBeVisible();
  await emitEvent(studio, { kind: "log", level: "info", message: "Deterministic fixture log" });
  await page.getByRole("button", { name: /Technical log/ }).click();
  await expect(page.locator(".logs")).toBeVisible();
  await expect(page.locator(".logs")).toContainText("Deterministic fixture log");
  await application.evaluate(({ BrowserWindow }) =>
    BrowserWindow.getAllWindows()[0]?.setSize(1080, 720),
  );
  await expect(page.getByRole("button", { name: /Credentials/ })).toBeInViewport();
  await page.getByRole("button", { name: "Chat", exact: true }).click();
  await expect(page.getByLabel("Message the local model")).toBeInViewport();
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth)).toBe(
    true,
  );
});

test("startup errors offer a working retry", async ({ studio }) => {
  await configureFixture(studio, { bootstrapError: "Fixture worker could not start." }, true);
  await expect(studio.page.getByRole("alert")).toContainText("Fixture worker could not start.");
  await configureFixture(studio, { bootstrapError: undefined });
  await studio.page.getByRole("button", { name: "Retry startup" }).click();
  await expect(
    studio.page.getByRole("heading", { name: "LFM2.5 230M", exact: true }),
  ).toBeVisible();
});

test("inspect rejects unsupported models without preparing and retains supported custom models", async ({
  studio,
}) => {
  const { page } = studio;
  await page.getByLabel("Add from Hugging Face").fill("not-a-model-id");
  await page.getByRole("button", { name: "Inspect model" }).click();
  await expect(page.getByRole("alert")).toContainText("organization/model");
  expect(await fixtureCalls(studio)).toHaveLength(0);
  await page.getByLabel("Add from Hugging Face").fill("fixture/unsupported");
  await page.getByLabel("Add from Hugging Face").press("Enter");
  await expect(page.getByRole("alert")).toContainText("has no registered adapter");
  expect((await fixtureCalls(studio)).map((call) => call.command)).toEqual(["inspect"]);
  await page.getByLabel("Add from Hugging Face").fill("fixture/network-error");
  await page.getByRole("button", { name: "Inspect model" }).click();
  await expect(page.getByRole("alert")).toContainText("Check your connection");
  await expect(page.getByRole("button", { name: "Inspect model" })).toBeEnabled();
  await page.getByLabel("Add from Hugging Face").fill("fixture/supported-custom");
  await page.getByRole("button", { name: "Inspect model" }).click();
  await expect(page.getByRole("heading", { name: "supported-custom", exact: true })).toBeVisible();
  await expect(page.getByRole("button", { name: /supported-custom/ })).toBeVisible();
  await expect(page.getByRole("status")).toContainText("parity are still required");
  await page.getByRole("button", { name: /LFM2.5 230M/ }).click();
  await page.getByRole("button", { name: /supported-custom/ }).click();
  await page.getByRole("button", { name: "Details", exact: true }).click();
  await expect(page.getByText("fixture-inspected-revision")).toBeVisible();
  await page.getByRole("button", { name: /Convert (?:to Core ML|again)/ }).click();
  await page.getByRole("button", { name: "Cancel operation" }).click();
  await expect(page.getByRole("status")).toContainText("Operation cancelled");
  await page.getByRole("button", { name: /LFM2.5 230M/ }).click();
  await page.getByRole("button", { name: /supported-custom/ }).click();
  await expect(page.getByRole("heading", { name: "supported-custom", exact: true })).toBeVisible();
  await expect(page.getByRole("button", { name: /Convert (?:to Core ML|again)/ })).toBeEnabled();
});

test("unsupported hardware blocks conversion and inference", async ({ studio }) => {
  const bootstrap = bootstrapFixture();
  bootstrap.hardware.supported = false;
  bootstrap.hardware.reason = "This fixture is not an Apple Silicon Mac.";
  bootstrap.models = [modelRecord()];
  await configureFixture(studio, { bootstrap }, true);
  await expect(studio.page.getByRole("alert")).toContainText("not an Apple Silicon Mac");
  await expect(
    studio.page.getByRole("button", { name: /Convert (?:to Core ML|again)/ }),
  ).toBeDisabled();
  await expect(studio.page.getByLabel("Message the local model")).toBeDisabled();
});

test("ready status alone never bypasses cosine and top-token parity", async ({ studio }) => {
  for (const parity of [
    { status: "passed" as const, cosineSimilarity: 0.9989, topTokenMatches: true },
    { status: "passed" as const, cosineSimilarity: 0.9999, topTokenMatches: false },
    { status: "passed" as const, topTokenMatches: true },
  ]) {
    await configureFixture(
      studio,
      { bootstrap: { ...bootstrapFixture(), models: [modelRecord({ parity })] } },
      true,
    );
    await expect(studio.page.getByLabel("Message the local model")).toBeDisabled();
    await expect(studio.page.getByText("Parity required", { exact: true }).first()).toBeVisible();
    await studio.page.getByRole("button", { name: "Tests", exact: true }).click();
    await expect(studio.page.getByRole("button", { name: "Run all tests" })).toBeDisabled();
  }
  expect(await fixtureCalls(studio)).toHaveLength(0);
});

test("conversion can cancel before the first progress event and retry", async ({ studio }) => {
  const { page } = studio;
  await page.getByRole("button", { name: /Convert (?:to Core ML|again)/ }).click();
  await expect(page.getByRole("button", { name: "Cancel operation" })).toBeEnabled();
  await expect(page.getByRole("button", { name: /LFM2.5 350M/ })).toBeDisabled();
  await expect(page.getByRole("button", { name: /Credentials/ })).toBeDisabled();
  await page.getByRole("button", { name: "Cancel operation" }).click();
  await expect(page.getByRole("status")).toContainText("Operation cancelled");
  await expect(page.getByRole("button", { name: /Convert (?:to Core ML|again)/ })).toBeEnabled();
  await page.getByRole("button", { name: /Convert (?:to Core ML|again)/ }).click();
  await emitEvent(studio, {
    kind: "progress",
    stage: "download",
    progress: 0.12,
    message: "Fixture download in progress",
  });
  await expect(page.getByRole("status")).toContainText("Fixture download in progress");
  await expect(page.getByRole("progressbar")).toHaveAttribute("value", "0.12");
  await page.getByRole("button", { name: "Cancel operation" }).click();
  await expect(page.getByRole("button", { name: /Convert (?:to Core ML|again)/ })).toBeEnabled();
  expect((await fixtureCalls(studio)).filter((call) => call.command === "cancel")).toHaveLength(2);
});

test("download and conversion errors recover, and failed parity leaves chat locked", async ({
  studio,
}) => {
  const { page } = studio;
  await page.getByRole("button", { name: /Convert (?:to Core ML|again)/ }).click();
  const downloading = modelRecord({ state: "downloading", parity: { status: "pending" } });
  await emitEvent(studio, { kind: "model", model: downloading });
  await emitEvent(studio, {
    kind: "progress",
    stage: "download",
    progress: 0.12,
    message: "Fixture downloading weights",
  });
  await expect(page.getByRole("status")).toContainText("Fixture downloading weights");
  await finishRequest(studio, "prepare", { error: "Fixture download failed. Disk is full." });
  await expect(page.getByRole("alert")).toContainText("Disk is full");
  await expect(page.getByRole("button", { name: /Convert (?:to Core ML|again)/ })).toBeEnabled();
  await page.getByRole("button", { name: /Convert (?:to Core ML|again)/ }).click();
  await emitEvent(studio, {
    kind: "model",
    model: modelRecord({ state: "converting", parity: { status: "pending" } }),
  });
  await emitEvent(studio, {
    kind: "progress",
    stage: "convert",
    progress: 0.7,
    message: "Fixture converting FP16 graph",
  });
  await expect(page.getByRole("status")).toContainText("Fixture converting FP16 graph");
  await finishRequest(studio, "prepare", { error: "Fixture Core ML conversion failed." });
  await expect(page.getByRole("alert")).toContainText("Core ML conversion failed");
  await page.getByRole("button", { name: /Convert (?:to Core ML|again)/ }).click();
  const failed = modelRecord({
    state: "failed",
    parity: { status: "failed", cosineSimilarity: 0.98, topTokenMatches: false },
    error: "Fixture source-to-Core-ML parity failed.",
  });
  await emitEvent(studio, { kind: "model", model: failed });
  await finishRequest(studio, "prepare", { result: failed });
  await expect(page.getByRole("alert")).toContainText("parity failed");
  await expect(page.getByRole("button", { name: "Try again" })).toBeEnabled();
  await expect(page.getByLabel("Message the local model")).toBeDisabled();
  await page.getByRole("button", { name: "Details", exact: true }).click();
  await expect(page.getByText("0.980000", { exact: true })).toBeVisible();
});

test("cancelled conversion stays interrupted after reload and can prepare again", async ({
  studio,
}) => {
  const { page } = studio;
  await page.getByRole("button", { name: "Convert to Core ML" }).click();
  await emitEvent(studio, {
    kind: "model",
    model: modelRecord({ state: "converting", parity: { status: "pending" } }),
  });
  await emitEvent(studio, {
    kind: "progress",
    stage: "convert",
    progress: 0.7,
    message: "Fixture conversion before cancellation",
  });
  await page.getByRole("button", { name: "Cancel operation" }).click();
  await expect(page.getByRole("button", { name: "Convert again" })).toBeEnabled();
  await page.reload();
  await expect(page.getByText("Interrupted", { exact: true }).first()).toBeVisible();
  await expect(page.getByRole("heading", { name: "Preparation was interrupted" })).toBeVisible();
  await expect(page.getByLabel("Message the local model")).toBeDisabled();
  await page.getByRole("button", { name: "Convert again" }).click();
  const ready = modelRecord();
  await emitEvent(studio, { kind: "model", model: ready });
  await finishRequest(studio, "prepare", { result: ready });
  await expect(page.getByLabel("Message the local model")).toBeEnabled();
});

test("successful fixture validation enables chat only after all conversion stages", async ({
  studio,
}) => {
  const { page } = studio;
  await page.getByRole("button", { name: /Convert (?:to Core ML|again)/ }).click();
  for (const [state, progress] of [
    ["downloading", 0.12],
    ["converting", 0.6],
    ["validating", 0.9],
  ] as const) {
    await emitEvent(studio, {
      kind: "model",
      model: modelRecord({ state, parity: { status: "pending" } }),
    });
    await emitEvent(studio, {
      kind: "progress",
      stage: state,
      progress,
      message: `Fixture ${state}`,
    });
    await expect(page.getByRole("status")).toContainText(`Fixture ${state}`);
    await expect(page.getByLabel("Message the local model")).toBeDisabled();
  }
  const ready = modelRecord();
  await emitEvent(studio, { kind: "model", model: ready });
  await finishRequest(studio, "prepare", { result: ready });
  await expect(page.getByLabel("Message the local model")).toBeEnabled();
  await expect(page.getByText("Parity passed", { exact: true })).toBeVisible();
  await expect(page.getByRole("button", { name: "Cancel operation" })).not.toBeVisible();
  await page.getByRole("button", { name: "Details", exact: true }).click();
  await expect(page.getByText("0.999800", { exact: true })).toBeVisible();
  await expect(page.getByText("e2e-fixture-not-a-real-digest", { exact: true })).toBeVisible();
  await expect(page.getByText("FP16 matrices + FP32 sensitive ops", { exact: true })).toBeVisible();
});

test("credential dialog clears unsaved secrets, traps focus, and recovers from errors", async ({
  studio,
}) => {
  const { page } = studio;
  const fakeToken = `hf_${"AneUIFixture".repeat(3)}`;
  await page.getByRole("button", { name: /Credentials/ }).click();
  const dialog = page.getByRole("dialog", { name: "Hugging Face token" });
  await expect(dialog).toBeVisible();
  await page.getByLabel("Access token").fill(fakeToken);
  await expect(page.getByLabel("Access token")).toHaveAttribute("type", "password");
  await page.getByRole("button", { name: "Save token", exact: true }).focus();
  await page.keyboard.press("Tab");
  expect(await dialog.evaluate((element) => element.contains(document.activeElement))).toBe(true);
  await page.keyboard.press("Escape");
  await expect(dialog).not.toBeVisible();
  await page.getByRole("button", { name: /Credentials/ }).click();
  await expect(page.getByLabel("Access token")).toHaveValue("");
  await expect(page.getByRole("button", { name: "Save token", exact: true })).toBeDisabled();
  await configureFixture(studio, { tokenError: "Fixture Keychain access failed." });
  await page.getByLabel("Access token").fill(fakeToken);
  await page.getByRole("button", { name: "Save token", exact: true }).click();
  await expect(dialog.getByRole("alert")).toContainText("Keychain access failed");
  await expect(page.getByRole("button", { name: "Save token", exact: true })).toBeEnabled();
  await configureFixture(studio, { tokenError: undefined });
  await page.getByRole("button", { name: "Save token", exact: true }).click();
  await expect(dialog).not.toBeVisible();
  expect(await page.locator("body").innerText()).not.toContain(fakeToken);
  await page.getByRole("button", { name: /Credentials/ }).click();
  await expect(page.getByLabel("Access token")).toHaveValue("");
  await expect(dialog).toContainText("Its value is never sent back");
  await configureFixture(studio, { tokenError: "Fixture token removal failed." });
  await page.getByRole("button", { name: "Remove token" }).click();
  await expect(dialog.getByRole("alert")).toContainText("token removal failed");
  await configureFixture(studio, { tokenError: undefined });
  await page.getByRole("button", { name: "Remove token" }).click();
  await expect(dialog).not.toBeVisible();
  await expect(page.getByRole("button", { name: /Credentials No token saved/ })).toBeVisible();
});

test("model removal requires confirmation and a failed removal does not lock the app", async ({
  studio,
}) => {
  const { page } = studio;
  await loadReadyModels(studio);
  await page.getByRole("button", { name: "Remove local model", exact: true }).click();
  const dialog = page.getByRole("dialog", { name: "Remove local model?" });
  await expect(dialog).toContainText("downloaded weights");
  await page.getByRole("button", { name: "Keep model" }).click();
  expect(await fixtureCalls(studio)).toHaveLength(0);
  await configureFixture(studio, { removeError: "Fixture files could not be removed." });
  await page.getByRole("button", { name: "Remove local model", exact: true }).click();
  await page.getByRole("button", { name: "Remove model", exact: true }).click();
  await expect(page.getByRole("alert")).toContainText("could not be removed");
  await expect(page.getByLabel("Message the local model")).toBeEnabled();
  await configureFixture(studio, { removeError: undefined });
  await page.getByRole("button", { name: "Remove local model", exact: true }).click();
  await page.getByRole("button", { name: "Remove model", exact: true }).click();
  await expect(page.getByRole("status")).toContainText("were removed");
  await expect(page.getByLabel("Message the local model")).toBeDisabled();
  await expect(page.getByRole("button", { name: /Convert (?:to Core ML|again)/ })).toBeEnabled();
});

test("chat uses final decoded text and never carries history or drafts across models", async ({
  studio,
}) => {
  const { page } = studio;
  await loadReadyModels(studio, [
    modelRecord(),
    modelRecord({ id: secondModelId, slug: "second-fixture" }),
  ]);
  await page.getByLabel("Requested policy").selectOption("cpuOnly");
  await page.getByLabel("Message the local model").fill("A short fixture prompt");
  await page.getByLabel("Message the local model").press("Enter");
  await emitEvent(studio, { kind: "token", token: "partial decode" });
  await expect(page.getByRole("log")).toContainText("partial decode");
  await expect(page.getByRole("button", { name: /LFM2.5 350M/ })).toBeDisabled();
  await finishRequest(studio, "chat", {
    result: {
      text: "Final decoded fixture reply.",
      tokens: 6,
      stopReason: "maxTokens",
      contextTokens: 20,
      sequenceLength: 64,
    },
  });
  await expect(page.locator(".generation-note")).toContainText("Final context used 20/64 tokens.");
  await expect(page.getByRole("log")).toContainText("Final decoded fixture reply.");
  await expect(page.getByRole("log")).not.toContainText("partial decode");
  const chat = (await fixtureCalls(studio)).find((call) => call.command === "chat");
  expect(chat?.payload).toMatchObject({
    modelId: firstModelId,
    policy: "cpuOnly",
    messages: [{ role: "user", content: "A short fixture prompt" }],
  });
  await page.getByLabel("Message the local model").fill("An unsent private draft");
  await page.getByRole("button", { name: /LFM2.5 350M/ }).click();
  await expect(page.getByLabel("Message the local model")).toHaveValue("");
  await expect(page.getByRole("log")).not.toContainText("fixture reply");
  await page.getByRole("button", { name: /LFM2.5 230M/ }).click();
  await expect(page.getByRole("log")).not.toContainText("fixture reply");
  await page.getByRole("button", { name: "Details", exact: true }).click();
  await page.getByRole("button", { name: /Technical log/ }).click();
  await expect(page.locator(".logs")).not.toContainText("partial decode");
});

test("chat errors, empty output, and cancellation preserve the draft but discard incomplete history", async ({
  studio,
}) => {
  const { page } = studio;
  await loadReadyModels(studio);
  await page.getByLabel("Message the local model").fill("A long fixture prompt");
  await page.getByRole("button", { name: "Send message" }).click();
  await finishRequest(studio, "chat", {
    error: "Prompt has 70 tokens; the fixed graph holds 64. Start a new chat.",
  });
  await expect(page.getByRole("alert")).toContainText("fixed graph holds 64");
  await expect(page.getByLabel("Message the local model")).toHaveValue("A long fixture prompt");
  await page.getByRole("button", { name: "Send message" }).click();
  await finishRequest(studio, "chat", { result: { text: "" } });
  await expect(page.getByRole("alert")).toContainText("The model returned no text");
  await expect(page.getByRole("log")).not.toContainText("A long fixture prompt");
  await page.getByRole("button", { name: "Send message" }).click();
  await emitEvent(studio, {
    kind: "progress",
    stage: "loading",
    message:
      "Loading the local Core ML runtime. First use of a compute policy can take several minutes. You can cancel.",
  });
  await expect(page.getByRole("status")).toContainText("First use of a compute policy");
  await expect(page.getByRole("progressbar")).not.toHaveAttribute("value");
  await expect(page.getByRole("button", { name: "Cancel operation" })).toBeEnabled();
  await emitEvent(studio, { kind: "token", token: "cancelled partial answer" });
  await page.getByRole("button", { name: "Cancel operation" }).click();
  await expect(page.getByRole("status")).toContainText("Operation cancelled");
  await emitEvent(studio, { kind: "token", token: "late stale token" });
  await expect(page.getByRole("log")).not.toContainText("cancelled partial answer");
  await expect(page.getByRole("log")).not.toContainText("late stale token");
  await expect(page.getByLabel("Message the local model")).toHaveValue("A long fixture prompt");
  await page.getByRole("button", { name: "Send message" }).click();
  await finishRequest(studio, "chat", { result: { text: "Recovered fixture answer." } });
  const calls = (await fixtureCalls(studio)).filter((call) => call.command === "chat");
  expect(calls.at(-1)?.payload.messages).toEqual([
    { role: "user", content: "A long fixture prompt" },
  ]);
  await page.getByRole("button", { name: "New chat" }).click();
  await expect(page.getByRole("log")).not.toContainText("Recovered fixture answer");
});

test("benchmark compares four policies without claiming observed accelerator activity", async ({
  studio,
}) => {
  const { page } = studio;
  await loadReadyModels(studio);
  await page.getByRole("button", { name: "Tests", exact: true }).click();
  await page.getByRole("button", { name: "Run all tests" }).click();
  await expect(page.getByRole("button", { name: "Cancel operation" })).toBeVisible();
  await page.getByRole("button", { name: "Cancel operation" }).click();
  await expect(page.getByRole("button", { name: "Run all tests" })).toBeEnabled();
  await page.getByRole("button", { name: "Run all tests" }).click();
  await finishRequest(studio, "benchmark", { error: "Fixture benchmark could not compile." });
  await expect(page.getByRole("alert")).toContainText("could not compile");
  await page.getByRole("button", { name: "Run all tests" }).click();
  const policies: ComputePolicy[] = ["cpuOnly", "cpuAndGPU", "cpuAndNeuralEngine", "all"];
  const report: BenchmarkReport = {
    modelId: firstModelId,
    prompt: "Fixture benchmark prompt",
    createdAt: new Date(0).toISOString(),
    policies: policies.map((policy, index) => ({
      policy,
      valid: index !== 1,
      firstTokenMs: index !== 1 ? 10 : undefined,
      medianTokenMs: index !== 1 ? 5 : undefined,
      tokensPerSecond: index !== 1 ? 200 : undefined,
      samples: index !== 1 ? [5, 5] : [],
      error: index === 1 ? "Fixture GPU policy failed" : undefined,
      placement: [
        { evidence: "requested", policy, observed: false, note: "Fixture requested policy" },
        {
          evidence: "planned",
          policy,
          observed: false,
          operationCounts: { ANE: index === 2 ? 7 : 0, CPU: 3 },
          note: "Fixture plan only",
        },
        { evidence: "observed", policy, observed: false, note: "No trace captured" },
      ],
    })),
  };
  await finishRequest(studio, "benchmark", { result: report });
  await expect(page.getByText("Not captured", { exact: true })).toBeVisible();
  await expect(page.getByText("Planned ANE 7/10", { exact: true })).toBeVisible();
  await expect(page.getByText("Fixture GPU policy failed", { exact: true })).toBeVisible();
  await expect(page.getByText(/not proof of ANE execution/)).toBeVisible();
  const request = (await fixtureCalls(studio)).find((call) => call.command === "benchmark");
  expect(request?.payload.policies).toEqual(policies);
});
