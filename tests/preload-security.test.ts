// @vitest-environment node
import { EventEmitter } from "node:events";
import { beforeEach, describe, expect, it, vi } from "vitest";
import type { StudioAPI } from "../src/shared/contracts";

const mocks = vi.hoisted(() => ({ invoke: vi.fn(), expose: vi.fn() }));
const emitter = new EventEmitter();
vi.mock("electron", () => ({
  contextBridge: { exposeInMainWorld: mocks.expose },
  ipcRenderer: {
    invoke: mocks.invoke,
    on: (channel: string, callback: (...args: unknown[]) => void) => emitter.on(channel, callback),
    removeListener: (channel: string, callback: (...args: unknown[]) => void) =>
      emitter.removeListener(channel, callback),
  },
}));

let api: StudioAPI;
beforeEach(async () => {
  vi.resetModules();
  vi.clearAllMocks();
  emitter.removeAllListeners();
  mocks.invoke.mockResolvedValue({});
  await import("../src/preload/index");
  api = mocks.expose.mock.calls[0]?.[1] as StudioAPI;
});

describe("isolated preload bridge", () => {
  it("exposes only the fixed application API, never raw IPC", () => {
    expect(mocks.expose.mock.calls[0]?.[0]).toBe("studio");
    expect(Object.keys(api).sort()).toEqual([
      "bootstrap",
      "cancel",
      "deleteToken",
      "onWorkerEvent",
      "request",
      "saveToken",
    ]);
    expect(Object.isFrozen(api)).toBe(true);
  });

  it("uses only fixed IPC channels with normalized validated data", async () => {
    await api.bootstrap();
    await api.request("inspect", { modelId: "  org/model  " });
    await api.cancel();
    await api.saveToken(` hf_${"a".repeat(34)} `);
    await api.deleteToken();
    expect(mocks.invoke.mock.calls).toEqual([
      ["studio:bootstrap"],
      ["studio:request", "inspect", { modelId: "org/model" }],
      ["studio:cancel"],
      ["studio:token:save", `hf_${"a".repeat(34)}`],
      ["studio:token:delete"],
    ]);
  });

  it("rejects invalid input before crossing IPC", async () => {
    await expect(api.request("prepare", { modelId: "../outside" })).rejects.toThrow();
    await expect(api.request("catalog", { path: "/tmp" })).rejects.toThrow();
    await expect(api.saveToken("bad")).rejects.toThrow();
    expect(mocks.invoke).not.toHaveBeenCalled();
  });

  it("supports independent subscriptions with idempotent cleanup", () => {
    const first = vi.fn();
    const second = vi.fn();
    const stopFirst = api.onWorkerEvent(first);
    const stopSecond = api.onWorkerEvent(second);
    const payload = {
      kind: "token",
      requestId: "r",
      token: "hello",
      at: new Date(0).toISOString(),
    };
    const electronEvent = { sender: { invoke: vi.fn() } };
    emitter.emit("studio:worker-event", electronEvent, payload);
    expect(first).toHaveBeenCalledExactlyOnceWith(payload);
    expect(second).toHaveBeenCalledExactlyOnceWith(payload);
    stopFirst();
    stopFirst();
    emitter.emit("studio:worker-event", electronEvent, payload);
    expect(first).toHaveBeenCalledTimes(1);
    expect(second).toHaveBeenCalledTimes(2);
    stopSecond();
    expect(emitter.listenerCount("studio:worker-event")).toBe(0);
  });

  it("drops malformed events and does not pass Electron event objects", () => {
    const callback = vi.fn();
    api.onWorkerEvent(callback);
    for (const value of [null, { kind: "model", model: {} }, { kind: "token", token: {} }]) {
      emitter.emit("studio:worker-event", { privileged: true }, value);
    }
    expect(callback).not.toHaveBeenCalled();
    expect(() => api.onWorkerEvent(null as unknown as () => void)).toThrow(
      "callback must be a function",
    );
  });
});
