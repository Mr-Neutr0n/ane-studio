// @vitest-environment node
import { EventEmitter } from "node:events";
import { PassThrough, Writable } from "node:stream";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { WorkerClient } from "../src/main/worker";

const mocks = vi.hoisted(() => ({
  spawn: vi.fn(),
  mkdirSync: vi.fn(),
  mkdtempSync: vi.fn(),
  rm: vi.fn(),
}));
vi.mock("node:child_process", () => ({ spawn: mocks.spawn }));
vi.mock("node:fs", () => ({ mkdirSync: mocks.mkdirSync, mkdtempSync: mocks.mkdtempSync }));
vi.mock("node:fs/promises", () => ({ rm: mocks.rm }));

class FakeChild extends EventEmitter {
  pid: number;
  stdout = new PassThrough();
  stderr = new PassThrough();
  messages: { id: string; command: string; payload: unknown }[] = [];
  exitOnKill = true;
  kill = vi.fn((signal: string) => {
    if (this.exitOnKill) this.emit("exit", null, signal);
    return true;
  });
  stdin = new Writable({
    write: (chunk, _encoding, callback) => {
      const message = JSON.parse(chunk.toString());
      this.messages.push(message);
      if (autoReplyHealth && message.command === "health")
        this.reply(this.messages.length - 1, { supported: true });
      callback();
    },
  });
  constructor(pid: number) {
    super();
    this.pid = pid;
  }
  reply(index: number, result: unknown = {}) {
    this.stdout.write(`${JSON.stringify({ id: this.messages[index]?.id, ok: true, result })}\n`);
  }
}

let children: FakeChild[];
let clients: WorkerClient[];
let autoReplyHealth = false;
const logs = vi.fn();
const events = vi.fn();
function client(options: Partial<ConstructorParameters<typeof WorkerClient>[0]> = {}) {
  const value = new WorkerClient({
    projectDirectory: "/bundle/worker",
    dataDirectory: "/tmp/ane-worker-test",
    onLog: logs,
    onEvent: events,
    ...options,
  });
  clients.push(value);
  return value;
}
async function tick() {
  for (let index = 0; index < 8; index += 1) await Promise.resolve();
}
async function ready(value: WorkerClient) {
  autoReplyHealth = true;
  try {
    await value.start();
    const child = children.at(-1);
    if (!child) throw new Error("Worker was not spawned.");
    return child;
  } finally {
    autoReplyHealth = false;
  }
}

beforeEach(() => {
  autoReplyHealth = false;
  children = [];
  clients = [];
  logs.mockReset();
  events.mockReset();
  mocks.spawn.mockReset();
  mocks.mkdirSync.mockReset();
  mocks.mkdtempSync.mockReset();
  mocks.rm.mockReset();
  mocks.rm.mockResolvedValue(undefined);
  mocks.mkdtempSync.mockImplementation(
    (prefix) => `${prefix}${mocks.mkdtempSync.mock.calls.length}`,
  );
  mocks.spawn.mockImplementation(() => {
    const child = new FakeChild(100_000 + children.length);
    children.push(child);
    return child;
  });
  vi.spyOn(process, "kill").mockImplementation((pid, signal) => {
    children.find((child) => child.pid === -pid)?.kill(String(signal));
    return true;
  });
});
afterEach(async () => {
  for (const child of children) child.exitOnKill = true;
  for (const value of clients) await value.stop();
  vi.useRealTimers();
  vi.unstubAllEnvs();
  vi.restoreAllMocks();
});

describe("WorkerClient", () => {
  it("shares one startup and waits for health before concurrent requests", async () => {
    const value = client();
    const first = value.request("catalog");
    const second = value.request("catalog");
    const child = children[0];
    expect(mocks.spawn).toHaveBeenCalledTimes(1);
    expect(child?.messages.map((message) => message.command)).toEqual(["health"]);
    child?.reply(0, { supported: true });
    await tick();
    expect(child?.messages.map((message) => message.command)).toEqual([
      "health",
      "catalog",
      "catalog",
    ]);
    child?.reply(1, { one: true });
    child?.reply(2, { two: true });
    await expect(first).resolves.toEqual({ one: true });
    await expect(second).resolves.toEqual({ two: true });
  });

  it("uses bundled isolated Python with no uv or inherited login on PATH", async () => {
    vi.stubEnv("PATH", "/usr/bin:/bin");
    vi.stubEnv("ANE_STUDIO_UV", "/do/not/run/uv");
    vi.stubEnv("HF_TOKEN", "outside-login");
    vi.stubEnv("HF_HOME", "/other/account");
    vi.stubEnv("HUGGING_FACE_HUB_TOKEN", "outside-login");
    vi.stubEnv("PYTHONPATH", "/other/code");
    const value = client({ packaged: true });
    await ready(value);
    const [executable, args, options] = mocks.spawn.mock.calls[0] ?? [];
    expect(executable).toBe("/bundle/runtime/bin/python3.12");
    expect(args).toEqual(["-I", "-B", "-u", "-m", "ane_studio_worker.cli"]);
    expect(options.env.PATH).toBe("/usr/bin:/bin");
    expect(options.env.HF_TOKEN).toBe("");
    expect(options.env.HF_HOME).toBe("/tmp/ane-worker-test/huggingface");
    expect(options.env.HF_HUB_DISABLE_IMPLICIT_TOKEN).toBe("1");
    expect(options.env.HF_HUB_DISABLE_TELEMETRY).toBe("1");
    expect(options.env.HUGGING_FACE_HUB_TOKEN).toBeUndefined();
    expect(options.env.PYTHONPATH).toBeUndefined();
    expect(mocks.mkdirSync).toHaveBeenCalledWith("/tmp/ane-worker-test", {
      recursive: true,
      mode: 0o700,
    });
    expect(mocks.mkdtempSync).toHaveBeenCalledWith("/tmp/ane-worker-test/.worker-tmp-");
    expect(options.env.TMPDIR).toBe("/tmp/ane-worker-test/.worker-tmp-1/");
    expect(mocks.mkdtempSync.mock.invocationCallOrder[0]).toBeLessThan(
      mocks.spawn.mock.invocationCallOrder[0] ?? 0,
    );
  });

  it("recovers after spawn failure without exposing raw error text", async () => {
    const value = client({ packaged: true });
    const start = value.start();
    const check = expect(start).rejects.toThrow("bundled Python runtime");
    children[0]?.emit("error", new Error("private path and credentials"));
    await check;
    expect(mocks.rm).toHaveBeenCalledWith("/tmp/ane-worker-test/.worker-tmp-1", {
      recursive: true,
      force: true,
    });
    await ready(value);
    expect(mocks.spawn).toHaveBeenCalledTimes(2);
    expect(logs).not.toHaveBeenCalled();
  });

  it("times out startup, terminates the process group, and permits retry", async () => {
    vi.useFakeTimers();
    const value = client({ startupTimeoutMs: 100 });
    const check = expect(value.start()).rejects.toThrow("timed out");
    await vi.advanceTimersByTimeAsync(100);
    await check;
    expect(process.kill).toHaveBeenCalledWith(-100_000, "SIGTERM");
    await ready(value);
  });

  it("removes only its allocated temp directory when spawn throws synchronously", async () => {
    mocks.spawn.mockImplementationOnce(() => {
      throw new Error("private spawn detail");
    });
    const value = client({ packaged: true });
    await expect(value.start()).rejects.toThrow("bundled Python runtime");
    expect(mocks.rm).toHaveBeenCalledExactlyOnceWith("/tmp/ane-worker-test/.worker-tmp-1", {
      recursive: true,
      force: true,
    });
    await ready(value);
    expect(mocks.spawn.mock.calls[1]?.[2].env.TMPDIR).toBe("/tmp/ane-worker-test/.worker-tmp-2/");
  });

  it("keeps native temp files until cancellation terminates the worker group", async () => {
    const value = client();
    const child = await ready(value);
    child.exitOnKill = false;
    const stopping = value.stop();
    await tick();
    expect(process.kill).toHaveBeenCalledWith(-child.pid, "SIGTERM");
    expect(mocks.rm).not.toHaveBeenCalled();
    child.emit("exit", null, "SIGTERM");
    await stopping;
    expect(process.kill).toHaveBeenCalledWith(-child.pid, "SIGKILL");
    expect(mocks.rm).toHaveBeenCalledExactlyOnceWith("/tmp/ane-worker-test/.worker-tmp-1", {
      recursive: true,
      force: true,
    });
    expect(mocks.rm.mock.invocationCallOrder[0]).toBeGreaterThan(
      vi.mocked(process.kill).mock.invocationCallOrder.at(-1) ?? 0,
    );
  });

  it("waits for crashed-session temp cleanup before starting a fresh session", async () => {
    const value = client();
    const child = await ready(value);
    let finishCleanup!: () => void;
    mocks.rm.mockImplementationOnce(
      () =>
        new Promise<void>((resolve) => {
          finishCleanup = resolve;
        }),
    );
    child.emit("exit", 1, null);
    await tick();
    expect(process.kill).toHaveBeenCalledWith(-child.pid, "SIGKILL");
    expect(mocks.rm).toHaveBeenCalledExactlyOnceWith("/tmp/ane-worker-test/.worker-tmp-1", {
      recursive: true,
      force: true,
    });
    const replacement = value.start();
    await tick();
    expect(mocks.spawn).toHaveBeenCalledTimes(1);
    finishCleanup();
    await tick();
    expect(mocks.spawn).toHaveBeenCalledTimes(2);
    expect(mocks.spawn.mock.calls[1]?.[2].env.TMPDIR).toBe("/tmp/ane-worker-test/.worker-tmp-2/");
    children[1]?.reply(0, { supported: true });
    await replacement;
  });

  it("rejects pending work on cancel and ignores an old worker's events and exit", async () => {
    const value = client();
    const old = await ready(value);
    const request = value.request("chat");
    const check = expect(request).rejects.toThrow("cancelled");
    await tick();
    const oldId = old.messages[1]?.id;
    await value.stop();
    await check;
    const replacement = await ready(value);
    old.stdout.write(
      `${JSON.stringify({ kind: "token", requestId: oldId, token: "stale", at: new Date().toISOString() })}\n`,
    );
    old.emit("exit", 1, null);
    const current = value.request("catalog");
    await tick();
    replacement.reply(1, { live: true });
    await expect(current).resolves.toEqual({ live: true });
    expect(events).not.toHaveBeenCalled();
    expect(mocks.spawn).toHaveBeenCalledTimes(2);
  });

  it("redacts explicit and token-shaped secrets in logs, events, and errors", async () => {
    const secret = `hf_${"a".repeat(30)}`;
    const value = client({ token: secret });
    const child = await ready(value);
    const request = value.request("chat");
    await tick();
    const id = child.messages[1]?.id;
    child.stderr.write(`warning ${secret}\n`);
    child.stdout.write(
      `${JSON.stringify({ kind: "token", requestId: id, token: secret, at: new Date().toISOString() })}\n`,
    );
    child.stdout.write(
      `${JSON.stringify({ id, ok: false, error: { message: `bad ${secret}`, detail: secret } })}\n`,
    );
    await expect(request).rejects.toThrow("bad [redacted]");
    expect(logs).toHaveBeenCalledWith("warning [redacted]");
    expect(events).toHaveBeenCalledWith(expect.objectContaining({ token: "[redacted]" }));
  });

  it("rejects malformed protocol without logging raw data", async () => {
    const value = client();
    const child = await ready(value);
    const request = value.request("catalog");
    await tick();
    const check = expect(request).rejects.toThrow("protocol error");
    child.stdout.write("not JSON with private content\n");
    await check;
    expect(logs).toHaveBeenCalledWith("Model worker emitted an invalid response.");
    expect(JSON.stringify(logs.mock.calls)).not.toContain("private content");
    await ready(value);
  });

  it("waits for automatic termination before spawning a replacement", async () => {
    vi.useFakeTimers();
    const value = client({ requestTimeoutMs: 100 });
    const old = await ready(value);
    old.exitOnKill = false;
    const request = value.request("catalog");
    const rejected = expect(request).rejects.toThrow("timed out");
    await tick();
    await vi.advanceTimersByTimeAsync(100);
    await rejected;
    const replacement = value.start();
    await tick();
    expect(mocks.spawn).toHaveBeenCalledTimes(1);
    old.exitOnKill = true;
    old.emit("exit", null, "SIGTERM");
    await tick();
    expect(mocks.spawn).toHaveBeenCalledTimes(2);
    children[1]?.reply(0, { supported: true });
    await replacement;
  });

  it("kills surviving descendants when the worker wrapper exits unexpectedly", async () => {
    const value = client();
    const child = await ready(value);
    const request = value.request("catalog");
    await tick();
    const rejected = expect(request).rejects.toThrow("unexpectedly");
    child.emit("exit", 1, null);
    await rejected;
    expect(process.kill).toHaveBeenCalledWith(-child.pid, "SIGKILL");
    await ready(value);
  });

  it("accepts many short carriage-return progress lines without killing the worker", async () => {
    const value = client();
    const child = await ready(value);
    child.stderr.write("progress\r".repeat(10_000));
    expect(process.kill).not.toHaveBeenCalled();
    const request = value.request("catalog");
    await tick();
    child.reply(1, { live: true });
    await expect(request).resolves.toEqual({ live: true });
  });

  it("recycles a worker after a broken input stream", async () => {
    const value = client();
    const old = await ready(value);
    const request = value.request("catalog");
    await tick();
    const rejected = expect(request).rejects.toThrow("Cannot write");
    old.stdin.emit("error", new Error("private EPIPE detail"));
    await rejected;
    await ready(value);
    expect(mocks.spawn).toHaveBeenCalledTimes(2);
  });

  it("rejects oversized unterminated lines before readline grows without a bound", async () => {
    const value = client();
    const child = await ready(value);
    const request = value.request("catalog");
    await tick();
    const rejected = expect(request).rejects.toThrow("safety limit");
    child.exitOnKill = false;
    child.stderr.write("x".repeat(64_001));
    await rejected;
    expect(child.stderr.destroyed).toBe(true);
    expect(child.stdout.destroyed).toBe(true);
    expect(logs).not.toHaveBeenCalled();
    child.exitOnKill = true;
    child.emit("exit", null, "SIGKILL");
    await ready(value);
  });

  it("bounds pending requests and times out stuck operations", async () => {
    vi.useFakeTimers();
    const value = client({ requestTimeoutMs: 100 });
    await ready(value);
    const requests = Array.from({ length: 32 }, () => value.request("catalog"));
    const settled = Promise.allSettled(requests);
    await tick();
    await expect(value.request("catalog")).rejects.toThrow("Too many worker requests");
    await vi.advanceTimersByTimeAsync(100);
    expect((await settled).every((result) => result.status === "rejected")).toBe(true);
    await ready(value);
  });
});
