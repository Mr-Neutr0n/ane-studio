import { type ChildProcessWithoutNullStreams, spawn } from "node:child_process";
import { dirname, join } from "node:path";
import { createInterface } from "node:readline";
import { isWorkerEvent, type WorkerEvent, type WorkerResponse } from "../shared/contracts";

interface PendingRequest {
  resolve(value: unknown): void;
  reject(reason: Error): void;
  timer: ReturnType<typeof setTimeout>;
}

interface WorkerSession {
  child: ChildProcessWithoutNullStreams;
  pending: Map<string, PendingRequest>;
  ready: boolean;
  stopped: boolean;
  exited: Promise<void>;
  termination?: Promise<void>;
}

export interface WorkerOptions {
  projectDirectory: string;
  dataDirectory: string;
  token?: string;
  packaged?: boolean;
  environmentDirectory?: string;
  startupTimeoutMs?: number;
  requestTimeoutMs?: number;
  onEvent(event: WorkerEvent): void;
  onLog(line: string): void;
}

const REQUEST_TIMEOUTS: Record<string, number> = {
  health: 30_000,
  catalog: 30_000,
  inspect: 120_000,
  remove: 120_000,
  placement: 300_000,
  chat: 600_000,
  prepare: 1_800_000,
  benchmark: 1_800_000,
};

export class WorkerClient {
  private session?: WorkerSession;
  private starting?: Promise<void>;
  private stopping?: Promise<void>;
  private sequence = 0;

  constructor(private readonly options: WorkerOptions) {}

  async start(): Promise<void> {
    if (this.stopping) await this.stopping;
    if (this.starting) return await this.starting;
    if (this.session?.ready) return;
    const starting = this.launch();
    this.starting = starting;
    try {
      await starting;
    } finally {
      if (this.starting === starting) this.starting = undefined;
    }
  }

  private async launch(): Promise<void> {
    const packaged = this.options.packaged === true;
    const executable = packaged
      ? join(dirname(this.options.projectDirectory), "runtime", "bin", "python3.12")
      : process.env.ANE_STUDIO_UV || "uv";
    const args = packaged
      ? ["-I", "-B", "-u", "-m", "ane_studio_worker.cli"]
      : [
          "run",
          "--project",
          this.options.projectDirectory,
          "--extra",
          "coreml",
          "--locked",
          "ane-studio-worker",
        ];
    const env = { ...process.env };
    // Do not inherit another tool's Hugging Face login or Python import path.
    for (const key of Object.keys(env)) {
      if (key.startsWith("HF_") || key.startsWith("HUGGING_FACE_") || key.startsWith("PYTHON")) {
        delete env[key];
      }
    }
    const child = spawn(executable, args, {
      cwd: this.options.projectDirectory,
      detached: process.platform !== "win32",
      env: {
        ...env,
        ANE_STUDIO_HOME: this.options.dataDirectory,
        HF_HOME: join(this.options.dataDirectory, "huggingface"),
        HF_TOKEN: this.options.token || "",
        HF_HUB_DISABLE_IMPLICIT_TOKEN: "1",
        HF_HUB_DISABLE_TELEMETRY: "1",
        DO_NOT_TRACK: "1",
        PYTHONUNBUFFERED: "1",
        PYTHONDONTWRITEBYTECODE: "1",
        ...(this.options.environmentDirectory && !packaged
          ? { UV_PROJECT_ENVIRONMENT: this.options.environmentDirectory }
          : {}),
      },
      stdio: ["pipe", "pipe", "pipe"],
    });
    let exit!: () => void;
    const session: WorkerSession = {
      child,
      pending: new Map(),
      ready: false,
      stopped: false,
      exited: new Promise<void>((resolve) => {
        exit = resolve;
      }),
    };
    this.session = session;
    const boundLines = (stream: NodeJS.ReadableStream, maximumBytes: number) => {
      let bytes = 0;
      stream.on("data", (chunk: Buffer) => {
        if (session.stopped) return;
        for (const byte of chunk) {
          bytes = byte === 10 || byte === 13 ? 0 : bytes + 1;
          if (bytes > maximumBytes) break;
        }
        if (bytes > maximumBytes) {
          this.failAll(session, new Error("Model worker response exceeded the safety limit."));
          stdout.close();
          stderr.close();
          child.stdout.destroy();
          child.stderr.destroy();
          void this.terminate(session);
        }
      });
    };
    // Limit unterminated lines before readline can accumulate an unbounded buffer.
    boundLines(child.stdout, 8_000_000);
    boundLines(child.stderr, 64_000);
    const stdout = createInterface({ input: child.stdout });
    const stderr = createInterface({ input: child.stderr });
    stdout.on("line", (line) => this.handleLine(session, line));
    stderr.on("line", (line) => {
      if (this.session === session && !session.stopped) this.options.onLog(this.redact(line));
    });
    child.stdin.on("error", () => {
      if (session.stopped) return;
      this.failAll(session, new Error("Cannot write to the model worker."));
      void this.terminate(session);
    });
    child.once("error", () => {
      const hint = packaged
        ? "The bundled Python runtime could not start. Download a fresh copy of ANE Studio."
        : "The model worker could not start. Install uv, then run bun run worker:setup. Set ANE_STUDIO_UV to its absolute path if needed.";
      this.failAll(session, new Error(hint));
      session.stopped = true;
      if (this.session === session) this.session = undefined;
      stdout.close();
      stderr.close();
      exit();
    });
    child.once("exit", (code, signal) => {
      stdout.close();
      stderr.close();
      // An unexpectedly exiting uv wrapper must not leave its Python child alive.
      if (!session.stopped && process.platform !== "win32" && child.pid) {
        try {
          process.kill(-child.pid, "SIGKILL");
        } catch {
          /* The process group already exited. */
        }
      }
      if (this.session === session) this.session = undefined;
      this.failAll(
        session,
        new Error(
          session.stopped
            ? "Operation cancelled."
            : `Model worker stopped unexpectedly (${signal || code || "unknown"}).`,
        ),
      );
      session.stopped = true;
      exit();
    });
    try {
      await this.send(session, "health", {}, this.options.startupTimeoutMs ?? 120_000);
      if (session.stopped || this.session !== session) throw new Error("Operation cancelled.");
      session.ready = true;
    } catch (error) {
      await this.terminate(session);
      throw error;
    }
  }

  async request<T>(command: string, payload: Record<string, unknown> = {}): Promise<T> {
    await this.start();
    const session = this.session;
    if (!session?.ready || session.stopped) throw new Error("Model worker is not available.");
    return await this.send<T>(
      session,
      command,
      payload,
      this.options.requestTimeoutMs ?? REQUEST_TIMEOUTS[command] ?? 120_000,
    );
  }

  private send<T>(
    session: WorkerSession,
    command: string,
    payload: Record<string, unknown>,
    timeout: number,
  ): Promise<T> {
    if (session.stopped || this.session !== session)
      return Promise.reject(new Error("Operation cancelled."));
    if (session.pending.size >= 32)
      return Promise.reject(new Error("Too many worker requests. Wait for the current operation."));
    const id = `request-${Date.now()}-${++this.sequence}`;
    const message = JSON.stringify({ id, command, payload });
    return new Promise<T>((resolve, reject) => {
      const timer = setTimeout(() => {
        this.failAll(session, new Error(`Model worker ${command} timed out. Please retry.`));
        void this.terminate(session);
      }, timeout);
      session.pending.set(id, { resolve: resolve as (value: unknown) => void, reject, timer });
      session.child.stdin.write(`${message}\n`, (error) => {
        if (!error) return;
        const pending = session.pending.get(id);
        if (!pending) return;
        clearTimeout(pending.timer);
        session.pending.delete(id);
        pending.reject(new Error("Cannot write to the model worker."));
        void this.terminate(session);
      });
    });
  }

  async restart(token?: string): Promise<void> {
    await this.stop();
    this.options.token = token;
    await this.start();
  }

  async stop(): Promise<void> {
    const session = this.session;
    if (session) await this.terminate(session);
    else if (this.stopping) await this.stopping;
    // Do not let a cancelled startup survive a public stop/restart boundary.
    await this.starting?.catch(() => {});
  }

  private terminate(session: WorkerSession): Promise<void> {
    if (session.termination) return session.termination;
    const stopping = this.endSession(session);
    session.termination = stopping;
    this.stopping = stopping;
    void stopping
      .finally(() => {
        if (this.stopping === stopping) this.stopping = undefined;
      })
      .catch(() => {});
    return stopping;
  }

  private async endSession(session: WorkerSession): Promise<void> {
    if (this.session === session) this.session = undefined;
    this.failAll(session, new Error("Operation cancelled."));
    if (session.stopped) return await session.exited;
    session.stopped = true;
    const kill = (signal: NodeJS.Signals) => {
      try {
        if (process.platform !== "win32" && session.child.pid)
          process.kill(-session.child.pid, signal);
        else session.child.kill(signal);
      } catch {
        session.child.kill(signal);
      }
    };
    kill("SIGTERM");
    const timer = setTimeout(() => kill("SIGKILL"), 2_000);
    try {
      await session.exited;
      // uv may exit before a descendant. Do not leave the old process group alive.
      if (process.platform !== "win32") kill("SIGKILL");
    } finally {
      clearTimeout(timer);
    }
  }

  private handleLine(session: WorkerSession, line: string): void {
    if (this.session !== session || session.stopped) return;
    let message: WorkerResponse | WorkerEvent;
    try {
      if (line.length > 8_000_000) throw new Error("Worker response is too large.");
      const value: unknown = JSON.parse(this.redact(line));
      if (!value || typeof value !== "object" || Array.isArray(value))
        throw new Error("Invalid object.");
      message = value as WorkerResponse | WorkerEvent;
      if ("kind" in message) {
        if (!isWorkerEvent(message)) throw new Error("Invalid event.");
        if (session.pending.has(message.requestId)) this.options.onEvent(message);
        return;
      }
      if (typeof message.id !== "string" || typeof message.ok !== "boolean")
        throw new Error("Invalid response.");
    } catch {
      this.options.onLog("Model worker emitted an invalid response.");
      this.failAll(session, new Error("Model worker protocol error. Please retry."));
      void this.terminate(session);
      return;
    }
    const pending = session.pending.get(message.id);
    if (!pending) return;
    clearTimeout(pending.timer);
    session.pending.delete(message.id);
    if (message.ok) pending.resolve(message.result);
    else
      pending.reject(
        new Error(
          typeof message.error?.message === "string"
            ? this.redact(message.error.message)
            : "Worker request failed.",
        ),
      );
  }

  private redact(value: string): string {
    let result = value;
    if (this.options.token) result = result.split(this.options.token).join("[redacted]");
    return result.replace(/hf_[A-Za-z0-9]{10,}/g, "[redacted]");
  }

  private failAll(session: WorkerSession, error: Error): void {
    for (const request of session.pending.values()) {
      clearTimeout(request.timer);
      request.reject(error);
    }
    session.pending.clear();
  }
}
