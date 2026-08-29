import { join } from "node:path";
import { is } from "@electron-toolkit/utils";
import { app, BrowserWindow, ipcMain, shell } from "electron";
import type {
  BootstrapData,
  CuratedModel,
  HardwareInfo,
  ModelRecord,
  WorkerEvent,
} from "../shared/contracts";
import { CredentialStore } from "./credentials";
import { WorkerClient } from "./worker";

app.setName("ANE Studio");
app.setPath("userData", process.env.ANE_STUDIO_HOME || join(app.getPath("appData"), "ANE Studio"));

let mainWindow: BrowserWindow | undefined;
let worker: WorkerClient | undefined;
let credentials: CredentialStore;

function projectPath(): string {
  return app.isPackaged ? join(process.resourcesPath, "worker") : join(app.getAppPath(), "worker");
}

function sendEvent(event: WorkerEvent): void {
  if (!mainWindow?.isDestroyed()) mainWindow?.webContents.send("studio:worker-event", event);
}

async function createWorker(): Promise<WorkerClient> {
  if (worker) return worker;
  worker = new WorkerClient({
    projectDirectory: projectPath(),
    dataDirectory: app.getPath("userData"),
    environmentDirectory: app.isPackaged
      ? join(app.getPath("userData"), "worker-environment")
      : undefined,
    token: await credentials.readToken(),
    onEvent: sendEvent,
    onLog: (message) =>
      sendEvent({
        kind: "log",
        requestId: "worker",
        message,
        level: "info",
        at: new Date().toISOString(),
      }),
  });
  return worker;
}

async function createWindow(): Promise<void> {
  const window = new BrowserWindow({
    width: 1440,
    height: 940,
    minWidth: 1080,
    minHeight: 720,
    show: false,
    titleBarStyle: "hiddenInset",
    backgroundColor: "#101827",
    webPreferences: {
      preload: join(__dirname, "../preload/index.cjs"),
      sandbox: true,
      contextIsolation: true,
      nodeIntegration: false,
    },
  });
  mainWindow = window;
  window.once("ready-to-show", () => window.show());
  window.webContents.setWindowOpenHandler(({ url }) => {
    if (url.startsWith("https://")) void shell.openExternal(url);
    return { action: "deny" };
  });
  window.webContents.on("will-navigate", (event, url) => {
    const current = window.webContents.getURL();
    if (url !== current) event.preventDefault();
  });

  if (is.dev && process.env.ELECTRON_RENDERER_URL) {
    await window.loadURL(process.env.ELECTRON_RENDERER_URL);
  } else {
    await window.loadFile(join(__dirname, "../renderer/index.html"));
  }
}

function registerIpc(): void {
  ipcMain.handle("studio:bootstrap", async (): Promise<BootstrapData> => {
    const client = await createWorker();
    const [hardware, catalog, hasToken] = await Promise.all([
      client.request<HardwareInfo>("health"),
      client.request<{ curatedModels: CuratedModel[]; models: ModelRecord[] }>("catalog"),
      credentials.hasToken(),
    ]);
    return {
      hardware,
      curatedModels: catalog.curatedModels,
      models: catalog.models,
      hasToken,
      dataDirectory: app.getPath("userData"),
    };
  });

  ipcMain.handle(
    "studio:request",
    async (_event, command: string, payload: Record<string, unknown> = {}) => {
      const allowed = new Set([
        "catalog",
        "inspect",
        "prepare",
        "chat",
        "benchmark",
        "placement",
        "remove",
      ]);
      if (!allowed.has(command)) throw new Error(`Unsupported command: ${command}`);
      return await (await createWorker()).request(command, payload);
    },
  );

  ipcMain.handle("studio:cancel", async () => {
    if (!worker) return;
    const token = await credentials.readToken();
    await worker.restart(token);
  });

  ipcMain.handle("studio:token:save", async (_event, token: string) => {
    await credentials.saveToken(token);
    if (worker) await worker.restart(await credentials.readToken());
  });

  ipcMain.handle("studio:token:delete", async () => {
    await credentials.deleteToken();
    if (worker) await worker.restart();
  });
}

app.whenReady().then(async () => {
  credentials = new CredentialStore(join(app.getPath("userData"), "credentials", "hf-token"));
  registerIpc();
  await createWindow();
  app.on("activate", async () => {
    if (BrowserWindow.getAllWindows().length === 0) await createWindow();
  });
});

app.on("window-all-closed", () => {
  if (process.platform !== "darwin") app.quit();
});

app.on("before-quit", () => {
  void worker?.stop();
});
