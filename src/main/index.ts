import { dirname, join, sep } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import { is } from "@electron-toolkit/utils";
import { app, BrowserWindow, dialog, type IpcMainInvokeEvent, ipcMain, shell } from "electron";
import {
  type BootstrapData,
  type CuratedModel,
  type HardwareInfo,
  isWorkerEvent,
  type ModelRecord,
  normalizeHuggingFaceToken,
  validateStudioRequest,
  type WorkerEvent,
} from "../shared/contracts";
import { CredentialStore } from "./credentials";
import { WorkerClient } from "./worker";

app.setName("ANE Studio");
// Keep existing installations and exported worker commands on the same data directory.
app.setPath("userData", process.env.ANE_STUDIO_HOME || join(app.getPath("appData"), "ANE Studio"));

let mainWindow: BrowserWindow | undefined;
let rendererDocumentURL: string | undefined;
let worker: WorkerClient | undefined;
let credentials: CredentialStore;
let controlQueue: Promise<unknown> = Promise.resolve();
let queuedControls = 0;
let quitting = false;
let shutdownComplete = false;

function publicError(error: unknown): string {
  const message = error instanceof Error ? error.message : "The local operation failed. Try again.";
  return message.replace(/hf_[A-Za-z0-9]+/g, "[redacted token]").slice(0, 2_000);
}

/** Only serialize worker setup and control changes, never a long model operation. */
function control<T>(action: () => Promise<T>): Promise<T> {
  if (quitting) return Promise.reject(new Error("ANE Studio is shutting down."));
  if (queuedControls >= 32)
    return Promise.reject(new Error("Too many requests. Wait and try again."));
  queuedControls += 1;
  const result = controlQueue.then(() => {
    if (quitting) throw new Error("ANE Studio is shutting down.");
    return action();
  });
  controlQueue = result.then(
    () => {
      queuedControls -= 1;
    },
    () => {
      queuedControls -= 1;
    },
  );
  return result;
}

function projectPath(): string {
  return app.isPackaged ? join(process.resourcesPath, "worker") : join(app.getAppPath(), "worker");
}

function sendEvent(event: WorkerEvent): void {
  if (!isWorkerEvent(event) || !mainWindow || mainWindow.isDestroyed()) return;
  if (mainWindow.webContents.isDestroyed()) return;
  mainWindow.webContents.send("studio:worker-event", event);
}

// Call only under control() so an async credential read cannot create duplicate workers.
async function createWorker(): Promise<WorkerClient> {
  if (worker) return worker;
  const token = await credentials.readToken();
  if (quitting) throw new Error("ANE Studio is shutting down.");
  worker = new WorkerClient({
    projectDirectory: projectPath(),
    dataDirectory: app.getPath("userData"),
    packaged: app.isPackaged,
    token,
    onEvent: sendEvent,
    onLog: (message) =>
      sendEvent({
        kind: "log",
        requestId: "worker",
        message: publicError(new Error(message)),
        level: "info",
        at: new Date().toISOString(),
      }),
  });
  return worker;
}

async function resetWorker(): Promise<void> {
  await worker?.stop();
  worker = undefined;
  // Start lazily on the next request. Cancel and token changes must not wait for a new runtime.
}

function sameDocument(url: string, expected: string | undefined): boolean {
  if (!expected) return false;
  try {
    const actual = new URL(url);
    const trusted = new URL(expected);
    actual.hash = "";
    trusted.hash = "";
    return actual.href === trusted.href;
  } catch {
    return false;
  }
}

function assertTrustedSender(event: IpcMainInvokeEvent): void {
  if (
    !mainWindow ||
    mainWindow.isDestroyed() ||
    mainWindow.webContents.isDestroyed() ||
    event.sender !== mainWindow.webContents ||
    !event.senderFrame ||
    event.senderFrame !== mainWindow.webContents.mainFrame ||
    !sameDocument(event.senderFrame.url, rendererDocumentURL)
  ) {
    throw new Error("This page cannot access ANE Studio.");
  }
}

function rendererURL(): string {
  if (is.dev && process.env.ELECTRON_RENDERER_URL) {
    const url = new URL(process.env.ELECTRON_RENDERER_URL);
    if (
      !["http:", "https:"].includes(url.protocol) ||
      !["localhost", "127.0.0.1", "[::1]"].includes(url.hostname) ||
      url.username ||
      url.password
    )
      throw new Error("The development renderer must use a local loopback URL.");
    return url.href;
  }
  return pathToFileURL(join(__dirname, "../renderer/index.html")).href;
}

function allowedRendererRequest(rawURL: string, documentURL: string): boolean {
  try {
    const url = new URL(rawURL);
    const document = new URL(documentURL);
    if (url.protocol === "data:") return true;
    if (document.protocol === "file:") {
      if (url.protocol !== "file:") return false;
      const directory = dirname(fileURLToPath(document));
      return fileURLToPath(url).startsWith(`${directory}${sep}`);
    }
    if (url.username || url.password) return false;
    if (url.protocol === "ws:") url.protocol = "http:";
    if (url.protocol === "wss:") url.protocol = "https:";
    return url.origin === document.origin;
  } catch {
    return false;
  }
}

async function createWindow(): Promise<void> {
  const documentURL = rendererURL();
  const window = new BrowserWindow({
    width: 1440,
    height: 940,
    minWidth: 1080,
    minHeight: 720,
    show: false,
    title: "ANE Studio",
    titleBarStyle: "hiddenInset",
    backgroundColor: "#101827",
    webPreferences: {
      preload: join(__dirname, "../preload/index.cjs"),
      partition: "ane-renderer",
      sandbox: true,
      contextIsolation: true,
      nodeIntegration: false,
      webSecurity: true,
      webviewTag: false,
    },
  });
  mainWindow = window;
  rendererDocumentURL = documentURL;
  const session = window.webContents.session;
  session.setPermissionRequestHandler((_contents, _permission, callback) => callback(false));
  session.setPermissionCheckHandler(() => false);
  session.setDevicePermissionHandler(() => false);
  session.webRequest.onBeforeRequest((details, callback) => {
    callback({ cancel: !allowedRendererRequest(details.url, documentURL) });
  });
  window.once("ready-to-show", () => {
    if (!window.isDestroyed()) window.show();
  });
  let openingExternal = false;
  window.webContents.setWindowOpenHandler(({ url }) => {
    try {
      const target = new URL(url);
      if (
        openingExternal ||
        url.length > 2_048 ||
        target.protocol !== "https:" ||
        target.username ||
        target.password ||
        target.port ||
        !["huggingface.co", "github.com", "harikp.com"].includes(target.hostname)
      )
        return { action: "deny" };
      openingExternal = true;
      void dialog
        .showMessageBox(window, {
          type: "question",
          title: "Open external link?",
          message: "Open this link in your browser?",
          detail: target.href,
          buttons: ["Cancel", "Open browser"],
          defaultId: 0,
          cancelId: 0,
          noLink: true,
        })
        .then(async ({ response }) => {
          if (response === 1 && !window.isDestroyed()) await shell.openExternal(target.href);
        })
        .catch(() => {
          if (!window.isDestroyed())
            dialog.showErrorBox(
              "Could not open the browser",
              "Try opening the site in your browser.",
            );
        })
        .finally(() => {
          openingExternal = false;
        });
    } catch {
      // Reject malformed URLs without reflecting their contents into an error or log.
    }
    return { action: "deny" };
  });
  window.webContents.on("will-navigate", (event, url) => {
    if (!sameDocument(url, documentURL)) event.preventDefault();
  });
  window.webContents.on("will-redirect", (event, url) => {
    if (!sameDocument(url, documentURL)) event.preventDefault();
  });
  window.webContents.on("will-frame-navigate", (event) => {
    if (!event.isMainFrame || !sameDocument(event.url, documentURL)) event.preventDefault();
  });
  window.webContents.on("will-attach-webview", (event) => event.preventDefault());
  window.webContents.on("render-process-gone", () => {
    void control(resetWorker).catch(() => undefined);
  });
  window.on("closed", () => {
    if (mainWindow === window) {
      mainWindow = undefined;
      rendererDocumentURL = undefined;
      void control(resetWorker).catch(() => undefined);
    }
  });
  await window.loadURL(documentURL);
}

function handle(
  channel: string,
  argumentsCount: number,
  action: (...args: unknown[]) => Promise<unknown>,
): void {
  ipcMain.handle(channel, async (event, ...args: unknown[]) => {
    assertTrustedSender(event);
    if (args.length !== argumentsCount) throw new Error("Invalid IPC arguments.");
    try {
      return await action(...args);
    } catch (error) {
      throw new Error(publicError(error));
    }
  });
}

function registerIpc(): void {
  handle("studio:bootstrap", 0, async (): Promise<BootstrapData> => {
    const { result } = await control(async () => {
      const client = await createWorker();
      const hasToken = await credentials.hasToken();
      return {
        result: Promise.all([
          client.request<HardwareInfo>("health"),
          client.request<{ curatedModels: CuratedModel[]; models: ModelRecord[] }>("catalog"),
        ]).then(([hardware, catalog]) => ({
          hardware,
          curatedModels: catalog.curatedModels,
          models: catalog.models,
          hasToken,
          dataDirectory: app.getPath("userData"),
        })),
      };
    });
    return await result;
  });

  handle("studio:request", 2, async (command, payload) => {
    const validated = validateStudioRequest(command, payload);
    const { result } = await control(async () => ({
      result: (await createWorker()).request(validated.command, validated.payload),
    }));
    return await result;
  });

  handle("studio:cancel", 0, async () => await control(resetWorker));

  handle("studio:token:save", 1, async (token) => {
    const normalized = normalizeHuggingFaceToken(token);
    await control(async () => {
      await credentials.saveToken(normalized);
      await resetWorker();
    });
  });

  handle(
    "studio:token:delete",
    0,
    async () =>
      await control(async () => {
        await credentials.deleteToken();
        await resetWorker();
      }),
  );
}

if (!app.requestSingleInstanceLock()) {
  app.quit();
} else {
  app.on("second-instance", () => {
    void app
      .whenReady()
      .then(async () => {
        if (!mainWindow || mainWindow.isDestroyed()) {
          await createWindow();
        } else {
          if (mainWindow.isMinimized()) mainWindow.restore();
          mainWindow.show();
          mainWindow.focus();
        }
      })
      .catch((error) => dialog.showErrorBox("ANE Studio could not open", publicError(error)));
  });
  app
    .whenReady()
    .then(async () => {
      credentials = new CredentialStore(join(app.getPath("userData"), "credentials", "hf-token"));
      registerIpc();
      await createWindow();
      app.on("activate", () => {
        if (BrowserWindow.getAllWindows().length === 0) {
          void createWindow().catch((error) =>
            dialog.showErrorBox("ANE Studio could not open", publicError(error)),
          );
        }
      });
    })
    .catch((error) => {
      dialog.showErrorBox("ANE Studio could not start", publicError(error));
      app.quit();
    });
}

app.on("window-all-closed", () => {
  if (process.platform !== "darwin") app.quit();
});

app.on("before-quit", (event) => {
  if (shutdownComplete) return;
  event.preventDefault();
  if (quitting) return;
  quitting = true;
  // Electron must remain alive until the child process and its pending requests settle.
  void controlQueue
    .then(resetWorker)
    .catch(() => undefined)
    .finally(() => {
      shutdownComplete = true;
      app.quit();
    });
});
