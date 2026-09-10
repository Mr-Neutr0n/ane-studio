import { mkdtemp, rm } from "node:fs/promises";
import { createRequire } from "node:module";
import { tmpdir } from "node:os";
import { join } from "node:path";
import {
  test as base,
  type ElectronApplication,
  _electron as electron,
  expect,
  type Page,
} from "@playwright/test";
import type { BootstrapData, ModelRecord, WorkerEvent } from "../../src/shared/contracts";

export const electronExecutable = createRequire(import.meta.url)("electron") as string;
export const firstModelId = "LiquidAI/LFM2.5-230M";
export const secondModelId = "LiquidAI/LFM2.5-350M";

export function modelRecord(overrides: Partial<ModelRecord> = {}): ModelRecord {
  return {
    id: firstModelId,
    slug: "lfm-fixture",
    state: "ready",
    architecture: "Lfm2ForCausalLM",
    parameterCount: 230_000_000,
    license: "apache-2.0",
    revision: "e2e-fixture-not-a-real-revision",
    maxSequenceLength: 64,
    artifactBytes: 1_024,
    artifactSha256: "e2e-fixture-not-a-real-digest",
    parity: {
      status: "passed",
      cosineSimilarity: 0.9998,
      topTokenMatches: true,
      computePrecision: "fp16-matrices-fp32-sensitive-ops",
    },
    updatedAt: new Date(0).toISOString(),
    ...overrides,
  };
}

export function bootstrapFixture(): BootstrapData {
  return {
    hardware: {
      supported: true,
      platform: "darwin",
      architecture: "arm64",
      osVersion: "15.0",
      chip: "Apple Silicon UI fixture",
      memoryBytes: 16 * 1024 ** 3,
    },
    curatedModels: [
      {
        id: firstModelId,
        label: "LFM2.5 230M",
        description: "Deterministic UI test fixture. No model is loaded.",
        parameters: 230_000_000,
        architecture: "Lfm2ForCausalLM",
        license: "apache-2.0",
        releasedAt: "2026-01-01",
        recommended: true,
      },
      {
        id: secondModelId,
        label: "LFM2.5 350M",
        description: "Second deterministic UI fixture.",
        parameters: 350_000_000,
        architecture: "Lfm2ForCausalLM",
        license: "apache-2.0",
        releasedAt: "2026-01-01",
        recommended: false,
      },
    ],
    models: [],
    hasToken: false,
    dataDirectory: "/fixture/not-user-data",
  };
}

interface FixtureState {
  bootstrap: BootstrapData;
  calls: { command: string; payload: Record<string, unknown> }[];
  pending?: {
    command: string;
    resolve(value: unknown): void;
    reject(reason: Error): void;
  };
  bootstrapError?: string;
  removeError?: string;
  tokenError?: string;
  tokenActions: string[];
}

export interface StudioFixture {
  application: ElectronApplication;
  page: Page;
  dataDirectory: string;
}

// These fixtures replace main-process IPC handlers from Playwright's privileged
// test driver. They do not expose a production test hook or run Python, download
// weights, convert a model, or claim real parity/placement evidence.
export const test = base.extend<{ studio: StudioFixture }>({
  // biome-ignore lint/correctness/noEmptyPattern: Playwright requires a destructured fixture parameter.
  studio: async ({}, use) => {
    const dataDirectory = await mkdtemp(join(tmpdir(), "ane-ui-fixture-"));
    let application: ElectronApplication | undefined;
    try {
      application = await electron.launch({
        executablePath: electronExecutable,
        args: ["."],
        env: {
          ...process.env,
          ANE_STUDIO_HOME: dataDirectory,
          // Even an early renderer bootstrap must not start a real worker.
          ANE_STUDIO_UV: process.execPath,
          HF_TOKEN: "",
          HUGGING_FACE_HUB_TOKEN: "",
        },
      });
      const page = await application.firstWindow();
      // firstWindow can still be about:blank. Reloading it is correctly blocked by
      // the app's navigation policy, so wait for the shipped document first.
      await page.waitForURL(/index\.html$/);
      await page.waitForLoadState("domcontentloaded");
      await application.evaluate(({ ipcMain }, bootstrap) => {
        const state: FixtureState = { bootstrap, calls: [], tokenActions: [] };
        Reflect.set(globalThis, "aneUiFixture", state);
        for (const channel of [
          "studio:bootstrap",
          "studio:request",
          "studio:cancel",
          "studio:token:save",
          "studio:token:delete",
        ])
          ipcMain.removeHandler(channel);
        ipcMain.handle("studio:bootstrap", () => {
          if (state.bootstrapError) throw new Error(state.bootstrapError);
          return state.bootstrap;
        });
        ipcMain.handle(
          "studio:request",
          (_event, command: string, payload: Record<string, unknown>) => {
            state.calls.push({ command, payload });
            if (command === "catalog")
              return {
                models: state.bootstrap.models,
                curatedModels: state.bootstrap.curatedModels,
              };
            if (command === "inspect") {
              if (payload.modelId === "fixture/network-error")
                throw new Error("Fixture metadata request failed. Check your connection.");
              return {
                id: payload.modelId,
                supported: payload.modelId !== "fixture/unsupported",
                reason:
                  "FixtureArchitecture has no registered adapter. No weights were downloaded.",
                architecture: "Lfm2ForCausalLM",
                parameterCount: 20_000_000,
                license: "apache-2.0",
                revision: "fixture-inspected-revision",
              };
            }
            if (command === "remove") {
              if (state.removeError) throw new Error(state.removeError);
              state.bootstrap.models = state.bootstrap.models.filter(
                (model) => model.id !== payload.modelId,
              );
              return { removed: true };
            }
            if (["prepare", "chat", "benchmark"].includes(command)) {
              if (state.pending) throw new Error("Fixture received concurrent worker operations.");
              return new Promise((resolve, reject) => {
                state.pending = { command, resolve, reject };
              });
            }
            throw new Error(`Unexpected fixture command: ${command}`);
          },
        );
        ipcMain.handle("studio:cancel", () => {
          state.calls.push({ command: "cancel", payload: {} });
          state.pending?.reject(new Error("Operation cancelled."));
          state.pending = undefined;
        });
        ipcMain.handle("studio:token:save", () => {
          if (state.tokenError) throw new Error(state.tokenError);
          state.tokenActions.push("save");
          state.bootstrap.hasToken = true;
        });
        ipcMain.handle("studio:token:delete", () => {
          if (state.tokenError) throw new Error(state.tokenError);
          state.tokenActions.push("delete");
          state.bootstrap.hasToken = false;
        });
      }, bootstrapFixture());
      await page.reload();
      await expect(page.getByText("ANE Studio", { exact: true }).first()).toBeVisible();
      await expect(page.getByRole("heading", { name: "LFM2.5 230M", exact: true })).toBeVisible();
      await use({ application, page, dataDirectory });
    } finally {
      await application?.close();
      await rm(dataDirectory, { recursive: true, force: true });
    }
  },
});

export async function configureFixture(
  studio: StudioFixture,
  values: Partial<
    Pick<FixtureState, "bootstrap" | "bootstrapError" | "removeError" | "tokenError">
  >,
  reload = false,
): Promise<void> {
  await studio.application.evaluate((_electron, values) => {
    Object.assign(Reflect.get(globalThis, "aneUiFixture") as FixtureState, values);
  }, values);
  if (reload) await studio.page.reload();
}

export async function emitEvent(
  studio: StudioFixture,
  event: Omit<WorkerEvent, "at" | "requestId">,
): Promise<void> {
  await studio.application.evaluate(({ BrowserWindow }, event) => {
    const state = Reflect.get(globalThis, "aneUiFixture") as FixtureState;
    if (event.model) {
      state.bootstrap.models = [
        ...state.bootstrap.models.filter((model) => model.id !== event.model?.id),
        event.model,
      ];
    }
    for (const window of BrowserWindow.getAllWindows()) {
      window.webContents.send("studio:worker-event", {
        ...event,
        at: new Date(0).toISOString(),
        requestId: "ui-fixture",
      });
    }
  }, event);
}

export async function finishRequest(
  studio: StudioFixture,
  command: string,
  outcome: { result?: unknown; error?: string },
): Promise<void> {
  await studio.application.evaluate(
    (_electron, { command, outcome }) => {
      const state = Reflect.get(globalThis, "aneUiFixture") as FixtureState;
      if (state.pending?.command !== command)
        throw new Error(`No pending ${command} fixture request.`);
      const pending = state.pending;
      state.pending = undefined;
      if (outcome.error) pending.reject(new Error(outcome.error));
      else pending.resolve(outcome.result);
    },
    { command, outcome },
  );
}

export async function fixtureCalls(studio: StudioFixture): Promise<FixtureState["calls"]> {
  return await studio.application.evaluate(
    () => (Reflect.get(globalThis, "aneUiFixture") as FixtureState).calls,
  );
}

export async function loadReadyModels(
  studio: StudioFixture,
  models = [modelRecord()],
): Promise<void> {
  await configureFixture(studio, { bootstrap: { ...bootstrapFixture(), models } }, true);
  await expect(studio.page.getByLabel("Message the local model")).toBeEnabled();
}
