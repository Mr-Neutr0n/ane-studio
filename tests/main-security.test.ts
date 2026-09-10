// @vitest-environment node
import { pathToFileURL } from "node:url";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { WorkerOptions } from "../src/main/worker";

type Listener = (...args: unknown[]) => void;
type IpcHandler = (event: unknown, ...args: unknown[]) => Promise<unknown>;
interface FakeWindow {
  destroyed: boolean;
  isMinimized: ReturnType<typeof vi.fn>;
  restore: ReturnType<typeof vi.fn>;
  focus: ReturnType<typeof vi.fn>;
  options: Record<string, unknown>;
  url: string;
  listeners: Map<string, Listener>;
  webContents: {
    mainFrame: { url: string };
    listeners: Map<string, Listener>;
    session: {
      setPermissionRequestHandler: ReturnType<typeof vi.fn>;
      setPermissionCheckHandler: ReturnType<typeof vi.fn>;
      setDevicePermissionHandler: ReturnType<typeof vi.fn>;
      webRequest: { onBeforeRequest: ReturnType<typeof vi.fn> };
    };
    isDestroyed: ReturnType<typeof vi.fn>;
    send: ReturnType<typeof vi.fn>;
    setWindowOpenHandler: ReturnType<typeof vi.fn>;
  };
}
const state = vi.hoisted(() => ({
  dev: false,
  windows: [] as FakeWindow[],
  handlers: new Map<string, IpcHandler>(),
  events: new Map<string, Listener>(),
  paths: new Map<string, string>(),
  workerOptions: [] as WorkerOptions[],
  setName: vi.fn(),
  requestLock: vi.fn(),
  setPath: vi.fn(),
  quit: vi.fn(),
  showError: vi.fn(),
  confirm: vi.fn(),
  openExternal: vi.fn(),
  request: vi.fn<(command: string, payload?: Record<string, unknown>) => Promise<unknown>>(),
  stop: vi.fn<() => Promise<void>>(),
  readToken: vi.fn<() => Promise<string | undefined>>(),
  hasToken: vi.fn<() => Promise<boolean>>(),
  saveToken: vi.fn<(token: string) => Promise<void>>(),
  deleteToken: vi.fn<() => Promise<void>>(),
}));
vi.mock("@electron-toolkit/utils", () => ({
  is: {
    get dev() {
      return state.dev;
    },
  },
}));
vi.mock("../src/main/worker", () => ({
  WorkerClient: class {
    constructor(options: WorkerOptions) {
      state.workerOptions.push(options);
    }
    request = state.request;
    stop = state.stop;
  },
}));
vi.mock("../src/main/credentials", () => ({
  CredentialStore: class {
    readToken = state.readToken;
    hasToken = state.hasToken;
    saveToken = state.saveToken;
    deleteToken = state.deleteToken;
  },
}));
vi.mock("electron", () => ({
  app: {
    isPackaged: false,
    requestSingleInstanceLock: state.requestLock,
    setName: state.setName,
    setPath: (name: string, path: string) => {
      state.setPath(name, path);
      state.paths.set(name, path);
    },
    getPath: (name: string) => state.paths.get(name),
    getAppPath: () => "/test/project",
    whenReady: () => Promise.resolve(),
    on: (name: string, callback: Listener) => state.events.set(name, callback),
    quit: state.quit,
  },
  BrowserWindow: class {
    static getAllWindows() {
      return state.windows.filter((window) => !window.destroyed);
    }
    destroyed = false;
    isMinimized = vi.fn(() => false);
    restore = vi.fn();
    focus = vi.fn();
    url = "";
    listeners = new Map<string, Listener>();
    webContents = {
      mainFrame: { url: "" },
      listeners: new Map<string, Listener>(),
      session: {
        setPermissionRequestHandler: vi.fn(),
        setPermissionCheckHandler: vi.fn(),
        setDevicePermissionHandler: vi.fn(),
        webRequest: { onBeforeRequest: vi.fn() },
      },
      isDestroyed: vi.fn(() => this.destroyed),
      send: vi.fn(),
      setWindowOpenHandler: vi.fn(),
      on: (name: string, callback: Listener) => this.webContents.listeners.set(name, callback),
    };
    constructor(public options: Record<string, unknown>) {
      state.windows.push(this);
    }
    isDestroyed() {
      return this.destroyed;
    }
    show() {}
    once(name: string, callback: Listener) {
      this.listeners.set(name, callback);
    }
    on(name: string, callback: Listener) {
      this.listeners.set(name, callback);
    }
    async loadURL(url: string) {
      this.url = url;
      this.webContents.mainFrame.url = url;
    }
  },
  ipcMain: {
    handle: (channel: string, handler: IpcHandler) => state.handlers.set(channel, handler),
  },
  dialog: { showMessageBox: state.confirm, showErrorBox: state.showError },
  shell: { openExternal: state.openExternal },
}));

function deferred<T>() {
  let resolve!: (value: T) => void;
  let reject!: (reason: Error) => void;
  const promise = new Promise<T>((yes, no) => {
    resolve = yes;
    reject = no;
  });
  return { promise, resolve, reject };
}
function window(): FakeWindow {
  const result = state.windows.at(-1);
  if (!result) throw new Error("No test window");
  return result;
}
function invoke(channel: string, ...args: unknown[]): Promise<unknown> {
  const handler = state.handlers.get(channel);
  if (!handler) throw new Error(`No handler: ${channel}`);
  return handler(
    { sender: window().webContents, senderFrame: window().webContents.mainFrame },
    ...args,
  );
}
async function start() {
  await import("../src/main/index");
}
function allowRequest(url: string): boolean {
  const callback = vi.fn();
  const hook = window().webContents.session.webRequest.onBeforeRequest.mock.calls[0]?.[0];
  hook({ url }, callback);
  return !callback.mock.calls[0]?.[0].cancel;
}
function openWindow(url: string) {
  const handler = window().webContents.setWindowOpenHandler.mock.calls[0]?.[0];
  return handler({ url });
}

beforeEach(() => {
  vi.resetModules();
  vi.clearAllMocks();
  vi.unstubAllEnvs();
  vi.stubEnv("ANE_STUDIO_HOME", "");
  vi.stubEnv("ELECTRON_RENDERER_URL", "");
  state.dev = false;
  state.requestLock.mockReturnValue(true);
  state.windows.length = 0;
  state.workerOptions.length = 0;
  state.handlers.clear();
  state.events.clear();
  state.paths.clear();
  state.paths.set("appData", "/test/appData");
  state.readToken.mockResolvedValue(undefined);
  state.hasToken.mockResolvedValue(false);
  state.saveToken.mockResolvedValue(undefined);
  state.deleteToken.mockResolvedValue(undefined);
  state.stop.mockResolvedValue(undefined);
  state.request.mockImplementation(async (command) =>
    command === "catalog" ? { curatedModels: [], models: [] } : {},
  );
  state.confirm.mockResolvedValue({ response: 0 });
  state.openExternal.mockResolvedValue(undefined);
});
afterEach(() => {
  vi.unstubAllEnvs();
});

describe("Electron process and renderer security", () => {
  it("exits a duplicate app before registering IPC or starting a worker", async () => {
    state.requestLock.mockReturnValue(false);
    await start();
    expect(state.quit).toHaveBeenCalledTimes(1);
    expect(state.windows).toHaveLength(0);
    expect(state.handlers.size).toBe(0);
    expect(state.workerOptions).toHaveLength(0);
  });

  it("restores and focuses the existing window for a duplicate Finder launch", async () => {
    await start();
    window().isMinimized.mockReturnValue(true);
    const focused = deferred<void>();
    window().focus.mockImplementationOnce(() => {
      focused.resolve();
    });
    state.events.get("second-instance")?.();
    await focused.promise;
    expect(window().restore).toHaveBeenCalledTimes(1);
    expect(window().focus).toHaveBeenCalledTimes(1);
    expect(state.windows).toHaveLength(1);
  });

  it("keeps the stable data path while sandboxing the app", async () => {
    await start();
    expect(state.setName).toHaveBeenCalledWith("ANE Studio");
    expect(state.setPath).toHaveBeenCalledWith("userData", "/test/appData/ANE Studio");
    expect(window().options.webPreferences).toMatchObject({
      sandbox: true,
      contextIsolation: true,
      nodeIntegration: false,
      webSecurity: true,
      webviewTag: false,
      partition: "ane-renderer",
    });
    await invoke("studio:bootstrap");
    expect(state.workerOptions[0]).toMatchObject({
      dataDirectory: "/test/appData/ANE Studio",
      packaged: false,
    });
    expect(state.workerOptions[0]).not.toHaveProperty("environmentDirectory");
  });

  it.each([
    "studio:bootstrap",
    "studio:request",
    "studio:cancel",
    "studio:token:save",
    "studio:token:delete",
  ])(
    "rejects foreign senders, subframes, missing frames, and navigated documents on %s",
    async (channel) => {
      await start();
      const handler = state.handlers.get(channel) as IpcHandler;
      const contents = window().webContents;
      const url = contents.mainFrame.url;
      for (const event of [
        { sender: {}, senderFrame: contents.mainFrame },
        { sender: contents, senderFrame: { url } },
        { sender: contents, senderFrame: null },
      ])
        await expect(handler(event)).rejects.toThrow("cannot access");
      contents.mainFrame.url = "https://example.com/";
      await expect(handler({ sender: contents, senderFrame: contents.mainFrame })).rejects.toThrow(
        "cannot access",
      );
      contents.mainFrame.url = url;
      window().destroyed = true;
      await expect(handler({ sender: contents, senderFrame: contents.mainFrame })).rejects.toThrow(
        "cannot access",
      );
      expect(state.workerOptions).toHaveLength(0);
      expect(state.saveToken).not.toHaveBeenCalled();
      expect(state.deleteToken).not.toHaveBeenCalled();
    },
  );

  it("validates command, argument count, and payload in main even if preload is bypassed", async () => {
    await start();
    await expect(invoke("studio:request", "health", {})).rejects.toThrow("Unsupported");
    await expect(invoke("studio:request", "remove", { modelId: "../outside" })).rejects.toThrow();
    await expect(
      invoke("studio:request", "inspect", { modelId: "org/model", path: "/tmp" }),
    ).rejects.toThrow();
    await expect(invoke("studio:request", "catalog")).rejects.toThrow("arguments");
    await expect(invoke("studio:cancel", "unexpected")).rejects.toThrow("arguments");
    await expect(invoke("studio:token:save", {})).rejects.toThrow("valid Hugging Face token");
    expect(state.workerOptions).toHaveLength(0);
    await invoke("studio:request", "inspect", { modelId: " org/model " });
    expect(state.request).toHaveBeenCalledWith("inspect", { modelId: "org/model" });
  });

  it("blocks renderer networking outside its own bundled files and denies permissions", async () => {
    await start();
    const document = window().url;
    expect(allowRequest(document)).toBe(true);
    expect(allowRequest(new URL("assets/app.js", document).href)).toBe(true);
    expect(allowRequest("data:image/png;base64,AA==")).toBe(true);
    for (const url of [
      "https://huggingface.co/",
      "http://127.0.0.1:8000/",
      "file:///etc/passwd",
      new URL("../main/index.js", document).href,
    ]) {
      expect(allowRequest(url)).toBe(false);
    }
    const session = window().webContents.session;
    expect(session.setPermissionCheckHandler.mock.calls[0]?.[0]()).toBe(false);
    expect(session.setDevicePermissionHandler.mock.calls[0]?.[0]()).toBe(false);
    const callback = vi.fn();
    session.setPermissionRequestHandler.mock.calls[0]?.[0](null, "media", callback);
    expect(callback).toHaveBeenCalledWith(false);
  });

  it("denies navigation, redirects, subframes, and webviews", async () => {
    await start();
    for (const eventName of ["will-navigate", "will-redirect"]) {
      const preventDefault = vi.fn();
      window().webContents.listeners.get(eventName)?.({ preventDefault }, "https://example.com");
      expect(preventDefault).toHaveBeenCalledTimes(1);
      window().webContents.listeners.get(eventName)?.(
        { preventDefault },
        `${window().url}#details`,
      );
      expect(preventDefault).toHaveBeenCalledTimes(1);
    }
    for (const [eventName, extra] of [
      ["will-frame-navigate", { isMainFrame: false, url: window().url }],
      ["will-attach-webview", {}],
    ] as const) {
      const preventDefault = vi.fn();
      window().webContents.listeners.get(eventName)?.({ preventDefault, ...extra });
      expect(preventDefault).toHaveBeenCalledTimes(1);
    }
  });

  it("rejects unsafe external URLs and requires native consent for allowed HTTPS sites", async () => {
    await start();
    for (const url of [
      "file:///tmp",
      "javascript:alert(1)",
      "https://example.com/",
      "https://huggingface.co.evil.test/",
      "https://huggingface.co@evil.test/",
      "https://user:pass@huggingface.co/",
      "https://huggingface.co:444/",
      "bad-url",
    ]) {
      expect(openWindow(url)).toEqual({ action: "deny" });
    }
    expect(state.confirm).not.toHaveBeenCalled();
    const consent = deferred<{ response: number }>();
    const opened = deferred<void>();
    state.confirm.mockReturnValueOnce(consent.promise);
    state.openExternal.mockImplementationOnce(async () => {
      opened.resolve();
    });
    expect(openWindow("https://huggingface.co/org/model")).toEqual({ action: "deny" });
    openWindow("https://github.com/");
    expect(state.confirm).toHaveBeenCalledTimes(1);
    expect(state.confirm.mock.calls[0]?.[1]).toMatchObject({ defaultId: 0, cancelId: 0 });
    expect(state.openExternal).not.toHaveBeenCalled();
    consent.resolve({ response: 1 });
    await opened.promise;
    expect(state.openExternal).toHaveBeenCalledWith("https://huggingface.co/org/model");
  });

  it("never opens an allowed link when consent is declined", async () => {
    await start();
    openWindow("https://harikp.com/");
    await state.confirm.mock.results[0]?.value;
    expect(state.openExternal).not.toHaveBeenCalled();
  });

  it("allows only the loopback Vite origin in development", async () => {
    state.dev = true;
    vi.stubEnv("ELECTRON_RENDERER_URL", "http://127.0.0.1:5173/");
    await start();
    expect(window().url).toBe("http://127.0.0.1:5173/");
    expect(allowRequest("http://127.0.0.1:5173/@vite/client")).toBe(true);
    expect(allowRequest("ws://127.0.0.1:5173/")).toBe(true);
    expect(allowRequest("http://127.0.0.1:8000/")).toBe(false);
    expect(allowRequest("https://example.com/")).toBe(false);
    expect(allowRequest(pathToFileURL("/etc/passwd").href)).toBe(false);
  });

  it("refuses to load a remote development renderer with the privileged preload", async () => {
    state.dev = true;
    vi.stubEnv("ELECTRON_RENDERER_URL", "https://example.com/");
    await start();
    expect(state.windows).toHaveLength(0);
    expect(state.showError).toHaveBeenCalledWith(
      "ANE Studio could not start",
      expect.stringContaining("loopback"),
    );
  });
});

describe("main worker control and recovery", () => {
  it("creates only one worker for concurrent bootstrap and requests", async () => {
    await start();
    const credentialRead = deferred<string | undefined>();
    const readStarted = deferred<void>();
    state.readToken.mockImplementationOnce(() => {
      readStarted.resolve();
      return credentialRead.promise;
    });
    const first = invoke("studio:bootstrap");
    const second = invoke("studio:request", "catalog", {});
    await readStarted.promise;
    expect(state.readToken).toHaveBeenCalledTimes(1);
    credentialRead.resolve(undefined);
    await Promise.all([first, second]);
    expect(state.workerOptions).toHaveLength(1);
  });

  it("cancels a pending model request without waiting for it and starts lazily afterward", async () => {
    await start();
    const operation = deferred<unknown>();
    const requested = deferred<void>();
    state.request.mockImplementationOnce(() => {
      requested.resolve();
      return operation.promise;
    });
    state.stop.mockImplementationOnce(async () => {
      operation.reject(new Error("Operation cancelled."));
    });
    const result = invoke("studio:request", "prepare", { modelId: "org/model" }).catch(
      (error: Error) => error.message,
    );
    await requested.promise;
    await invoke("studio:cancel");
    expect(await result).toBe("Operation cancelled.");
    expect(state.stop).toHaveBeenCalledTimes(1);
    expect(state.workerOptions).toHaveLength(1);
    await invoke("studio:request", "catalog", {});
    expect(state.workerOptions).toHaveLength(2);
  });

  it("does not start a worker just to cancel an idle app", async () => {
    await start();
    await invoke("studio:cancel");
    expect(state.workerOptions).toHaveLength(0);
    expect(state.readToken).not.toHaveBeenCalled();
  });

  it("serializes credential changes before the next worker can read the token", async () => {
    await start();
    await invoke("studio:request", "catalog", {});
    const token = `hf_${"a".repeat(34)}`;
    state.saveToken.mockImplementationOnce(async () => {
      state.readToken.mockResolvedValue(token);
    });
    const saved = invoke("studio:token:save", token);
    const next = invoke("studio:request", "catalog", {});
    await Promise.all([saved, next]);
    expect(state.stop).toHaveBeenCalledTimes(1);
    expect(state.workerOptions[1]?.token).toBe(token);
    state.deleteToken.mockImplementationOnce(async () => {
      state.readToken.mockResolvedValue(undefined);
    });
    await invoke("studio:token:delete");
    await invoke("studio:request", "catalog", {});
    expect(state.workerOptions[2]?.token).toBeUndefined();
  });

  it("recovers after a rejected control action and redacts tokens in IPC errors/logs", async () => {
    await start();
    const token = `hf_${"a".repeat(34)}`;
    state.saveToken.mockRejectedValueOnce(new Error(`Keychain failure ${token}`));
    await expect(invoke("studio:token:save", token)).rejects.toThrow(
      "Keychain failure [redacted token]",
    );
    await invoke("studio:bootstrap");
    state.workerOptions[0]?.onLog(`Authorization failed for ${token}`);
    const last = window().webContents.send.mock.calls.at(-1)?.[1];
    expect(last.message).toBe("Authorization failed for [redacted token]");
  });

  it("drops invalid events and stops sending to destroyed windows", async () => {
    await start();
    await invoke("studio:bootstrap");
    const onEvent = state.workerOptions[0]?.onEvent;
    onEvent?.(null as never);
    expect(window().webContents.send).not.toHaveBeenCalled();
    const event = {
      kind: "token" as const,
      requestId: "r",
      token: "hello",
      at: new Date(0).toISOString(),
    };
    onEvent?.(event);
    expect(window().webContents.send).toHaveBeenCalledExactlyOnceWith("studio:worker-event", event);
    window().destroyed = true;
    onEvent?.(event);
    expect(window().webContents.send).toHaveBeenCalledTimes(1);
  });

  it.each(["closed", "render-process-gone"])("stops hidden model work after %s", async (name) => {
    await start();
    await invoke("studio:bootstrap");
    const stopped = deferred<void>();
    state.stop.mockImplementationOnce(async () => {
      stopped.resolve();
    });
    const listeners = name === "closed" ? window().listeners : window().webContents.listeners;
    listeners.get(name)?.();
    await stopped.promise;
    expect(state.stop).toHaveBeenCalledTimes(1);
  });

  it("holds application quit until the worker exits and refuses new work", async () => {
    await start();
    await invoke("studio:bootstrap");
    const stopped = deferred<void>();
    const stopping = deferred<void>();
    const quit = deferred<void>();
    state.stop.mockImplementationOnce(() => {
      stopping.resolve();
      return stopped.promise;
    });
    state.quit.mockImplementationOnce(() => {
      quit.resolve();
    });
    const preventDefault = vi.fn();
    state.events.get("before-quit")?.({ preventDefault });
    await stopping.promise;
    expect(preventDefault).toHaveBeenCalledTimes(1);
    expect(state.quit).not.toHaveBeenCalled();
    await expect(invoke("studio:request", "catalog", {})).rejects.toThrow("shutting down");
    stopped.resolve();
    await quit.promise;
    const finalPrevent = vi.fn();
    state.events.get("before-quit")?.({ preventDefault: finalPrevent });
    expect(finalPrevent).not.toHaveBeenCalled();
  });
});
