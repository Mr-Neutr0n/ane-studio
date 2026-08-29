import { type ChildProcessWithoutNullStreams, spawn } from "node:child_process";
import { createInterface } from "node:readline";
import type { WorkerEvent, WorkerResponse } from "../shared/contracts";

interface PendingRequest {
  resolve(value: unknown): void;
  reject(reason: Error): void;
}

export interface WorkerOptions {
  projectDirectory: string;
  dataDirectory: string;
  token?: string;
  environmentDirectory?: string;
  onEvent(event: WorkerEvent): void;
  onLog(line: string): void;
}

export class WorkerClient {
  private child?: ChildProcessWithoutNullStreams;
  private pending = new Map<string, PendingRequest>();
  private sequence = 0;
  private stopping = false;

  constructor(private readonly options: WorkerOptions) {}

  async start(): Promise<void> {
    if (this.child) return;
    this.stopping = false;
    const executable = process.env.ANE_STUDIO_UV || "uv";
    const child = spawn(
      executable,
      [
        "run",
        "--project",
        this.options.projectDirectory,
        "--extra",
        "coreml",
        "--locked",
        "ane-studio-worker",
      ],
      {
        cwd: this.options.projectDirectory,
        env: {
          ...process.env,
          ANE_STUDIO_HOME: this.options.dataDirectory,
          HF_TOKEN: this.options.token || "",
          PYTHONUNBUFFERED: "1",
          ...(this.options.environmentDirectory
            ? { UV_PROJECT_ENVIRONMENT: this.options.environmentDirectory }
            : {}),
        },
        stdio: ["pipe", "pipe", "pipe"],
      },
    );
    this.child = child;

    createInterface({ input: child.stdout }).on("line", (line) => this.handleLine(line));
    createInterface({ input: child.stderr }).on("line", (line) => this.options.onLog(line));
    child.on("error", (error) => this.failAll(error));
    child.on("exit", (code, signal) => {
      this.child = undefined;
      if (!this.stopping) {
        this.failAll(
          new Error(`Model worker stopped unexpectedly (${signal || code || "unknown"}).`),
        );
      }
    });

    await this.request("health", {});
  }

  async request<T>(command: string, payload: Record<string, unknown> = {}): Promise<T> {
    if (!this.child) await this.start();
    const id = `request-${Date.now()}-${++this.sequence}`;
    const message = JSON.stringify({ id, command, payload });
    return await new Promise<T>((resolve, reject) => {
      this.pending.set(id, { resolve: resolve as (value: unknown) => void, reject });
      this.child?.stdin.write(`${message}\n`, (error) => {
        if (!error) return;
        this.pending.delete(id);
        reject(error);
      });
    });
  }

  async restart(token?: string): Promise<void> {
    await this.stop();
    this.options.token = token;
    await this.start();
  }

  async stop(): Promise<void> {
    this.stopping = true;
    const child = this.child;
    this.child = undefined;
    if (!child) return;
    child.kill("SIGTERM");
    await new Promise<void>((resolve) => {
      const timer = setTimeout(() => {
        child.kill("SIGKILL");
        resolve();
      }, 2_000);
      child.once("exit", () => {
        clearTimeout(timer);
        resolve();
      });
    });
    this.failAll(new Error("Operation cancelled."));
  }

  private handleLine(line: string): void {
    let message: WorkerResponse | WorkerEvent;
    try {
      message = JSON.parse(line) as WorkerResponse | WorkerEvent;
    } catch {
      this.options.onLog(`worker emitted invalid JSON: ${line}`);
      return;
    }
    if ("kind" in message) {
      this.options.onEvent(message);
      return;
    }
    const pending = this.pending.get(message.id);
    if (!pending) return;
    this.pending.delete(message.id);
    if (message.ok) {
      pending.resolve(message.result);
    } else {
      const error = new Error(message.error?.message || "Worker request failed.");
      if (message.error?.detail) error.stack = message.error.detail;
      pending.reject(error);
    }
  }

  private failAll(error: Error): void {
    for (const request of this.pending.values()) request.reject(error);
    this.pending.clear();
  }
}
