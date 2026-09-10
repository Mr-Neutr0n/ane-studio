import {
  Activity,
  AlertTriangle,
  ArrowUp,
  Box,
  Check,
  ChevronDown,
  CircleGauge,
  CircuitBoard,
  Cpu,
  Download,
  ExternalLink,
  FileCode2,
  FlaskConical,
  Gauge,
  HardDrive,
  KeyRound,
  LoaderCircle,
  MemoryStick,
  MessageSquare,
  Plus,
  Settings2,
  ShieldCheck,
  Sparkles,
  TerminalSquare,
  Trash2,
  X,
  Zap,
} from "lucide-react";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type {
  BenchmarkReport,
  BootstrapData,
  ChatMessage,
  ComputePolicy,
  CuratedModel,
  ModelRecord,
  PlacementSummary,
  WorkerEvent,
} from "../shared/contracts";
import { hasPassedParity } from "../shared/contracts";

type View = "chat" | "tests" | "details";
type Operation = "prepare" | "inspect" | "chat" | "benchmark" | "remove";
type DisplayMessage = ChatMessage & { id: string };
type DisplayEvent = WorkerEvent & { id: string };
type ChatResult = {
  text: string;
  stopReason?: "eos" | "contextLimit" | "maxTokens";
  contextTokens?: number;
  sequenceLength?: number;
  tokens?: number;
};

const policies: { id: ComputePolicy; label: string }[] = [
  { id: "cpuOnly", label: "CPU only" },
  { id: "cpuAndGPU", label: "CPU + GPU" },
  { id: "cpuAndNeuralEngine", label: "CPU + Neural Engine" },
  { id: "all", label: "Automatic" },
];

const operationLabels: Record<Operation, string> = {
  prepare: "Inspecting the model before download...",
  inspect: "Reading Hugging Face metadata...",
  chat: "Generating on this Mac...",
  benchmark: "Comparing Core ML compute policies...",
  remove: "Removing local model files...",
};

function bytes(value?: number): string {
  if (!value) return "Not available";
  const units = ["B", "KB", "MB", "GB"];
  let amount = value;
  let unit = 0;
  while (amount >= 1024 && unit < units.length - 1) {
    amount /= 1024;
    unit += 1;
  }
  return `${amount.toFixed(unit > 1 ? 1 : 0)} ${units[unit]}`;
}

function parameterLabel(value?: number): string {
  if (!value) return "Unknown size";
  if (value >= 1_000_000_000) return `${(value / 1_000_000_000).toFixed(1)}B parameters`;
  return `${Math.round(value / 1_000_000)}M parameters`;
}

function modelCanRun(record?: ModelRecord): boolean {
  return record?.state === "ready" && hasPassedParity(record.parity);
}

function isIncomplete(record?: ModelRecord): boolean {
  return (
    !!record && ["inspecting", "downloading", "converting", "validating"].includes(record.state)
  );
}

function stateLabel(record?: ModelRecord, active = false): string {
  if (!record) return "Not converted";
  if (!active && isIncomplete(record)) return "Interrupted";
  if (record.state === "ready" && !modelCanRun(record)) return "Parity required";
  const labels: Record<ModelRecord["state"], string> = {
    available: "Available",
    inspecting: "Inspecting",
    downloading: "Downloading",
    converting: "Converting",
    validating: "Validating",
    ready: "Ready",
    failed: "Failed",
  };
  return labels[record.state];
}

function precisionLabel(value?: string): string {
  if (value === "fp16-matrices-fp32-sensitive-ops") return "FP16 matrices + FP32 sensitive ops";
  if (value === "fp32") return "FP32 compute";
  if (value === "fp16") return "FP16 compute";
  return value || "Not recorded";
}

function errorMessage(reason: unknown): string {
  return reason instanceof Error ? reason.message : String(reason);
}

function App(): React.JSX.Element {
  const [bootstrap, setBootstrap] = useState<BootstrapData>();
  const [bootAttempt, setBootAttempt] = useState(0);
  const [error, setError] = useState<string>();
  const [notice, setNotice] = useState<string>();
  const [selectedId, setSelectedId] = useState("LiquidAI/LFM2.5-230M");
  const [customId, setCustomId] = useState("");
  const [view, setView] = useState<View>("chat");
  const [operation, setOperation] = useState<Operation>();
  const activeOperation = useRef<Operation | undefined>(undefined);
  const [cancelling, setCancelling] = useState(false);
  const cancellingRef = useRef(false);
  const [events, setEvents] = useState<DisplayEvent[]>([]);
  const [progress, setProgress] = useState<WorkerEvent>();
  const [messages, setMessages] = useState<DisplayMessage[]>([]);
  const [generation, setGeneration] = useState<ChatResult>();
  const [prompt, setPrompt] = useState("");
  const [policy, setPolicy] = useState<ComputePolicy>("cpuAndNeuralEngine");
  const [benchmark, setBenchmark] = useState<BenchmarkReport>();
  const [showLogs, setShowLogs] = useState(false);
  const [showToken, setShowToken] = useState(false);
  const [token, setToken] = useState("");
  const [tokenBusy, setTokenBusy] = useState(false);
  const [tokenError, setTokenError] = useState<string>();
  const [showRemove, setShowRemove] = useState(false);
  const endRef = useRef<HTMLDivElement>(null);
  const busy = operation !== undefined || cancelling;

  const updateModel = useCallback((model: ModelRecord): void => {
    setBootstrap((current) => {
      if (!current) return current;
      const models = current.models.filter((item) => item.id !== model.id);
      return { ...current, models: [...models, model] };
    });
  }, []);

  // biome-ignore lint/correctness/useExhaustiveDependencies: Changing bootAttempt explicitly retries a failed bootstrap.
  useEffect(() => {
    let active = true;
    setError(undefined);
    const unsubscribe = window.studio.onWorkerEvent((workerEvent) => {
      // Streamed output belongs in the chat, not in the technical log.
      if (workerEvent.kind !== "token") {
        setEvents((current) => [
          ...current.slice(-299),
          { ...workerEvent, id: crypto.randomUUID() },
        ]);
      }
      if (workerEvent.kind === "progress" && activeOperation.current) setProgress(workerEvent);
      if (workerEvent.kind === "model" && workerEvent.model) updateModel(workerEvent.model);
      if (workerEvent.kind === "token" && workerEvent.token && activeOperation.current === "chat") {
        setMessages((current) => {
          const next = [...current];
          const last = next.at(-1);
          if (last?.role === "assistant") {
            next[next.length - 1] = { ...last, content: last.content + workerEvent.token };
          }
          return next;
        });
      }
    });
    void window.studio
      .bootstrap()
      .then((data) => {
        if (!active) return;
        setBootstrap(data);
        setSelectedId((current) =>
          [...data.curatedModels, ...data.models].some((model) => model.id === current)
            ? current
            : data.curatedModels[0]?.id || data.models[0]?.id || current,
        );
      })
      .catch((reason: unknown) => {
        if (active) setError(errorMessage(reason));
      });
    return () => {
      active = false;
      unsubscribe();
    };
  }, [bootAttempt, updateModel]);

  useEffect(() => {
    if (messages.length) endRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const selectedRecord = bootstrap?.models.find((model) => model.id === selectedId);
  const selectedCurated = bootstrap?.curatedModels.find((model) => model.id === selectedId);
  const modelName = selectedCurated?.label || selectedId.split("/").at(-1) || selectedId;
  const isReady = modelCanRun(selectedRecord) && bootstrap?.hardware.supported === true;

  const modelList = useMemo(() => {
    if (!bootstrap) return [];
    const known = new Map<string, CuratedModel | ModelRecord>();
    for (const model of bootstrap.curatedModels) known.set(model.id, model);
    for (const model of bootstrap.models) if (!known.has(model.id)) known.set(model.id, model);
    return [...known.values()];
  }, [bootstrap]);

  function selectModel(id: string): void {
    if (busy || id === selectedId) return;
    setSelectedId(id);
    setGeneration(undefined);
    setMessages([]);
    setPrompt("");
    setBenchmark(undefined);
    setProgress(undefined);
    setError(undefined);
    setNotice(undefined);
  }

  function beginOperation(next: Operation): boolean {
    if (activeOperation.current || cancellingRef.current) return false;
    activeOperation.current = next;
    setOperation(next);
    setProgress(undefined);
    setError(undefined);
    setNotice(undefined);
    return true;
  }

  function finishOperation(): void {
    activeOperation.current = undefined;
    setOperation(undefined);
    setProgress(undefined);
  }

  function reportError(reason: unknown): void {
    if (!cancellingRef.current) setError(errorMessage(reason));
  }

  async function prepare(): Promise<void> {
    if (!bootstrap?.hardware.supported || !beginOperation("prepare")) return;
    try {
      const record = await window.studio.request<ModelRecord>("prepare", {
        modelId: selectedId,
        sequenceLength: 64,
      });
      updateModel(record);
      if (!modelCanRun(record)) {
        setError(
          record.error ||
            "This model did not pass source-to-Core-ML parity. Chat and tests stay disabled.",
        );
      }
    } catch (reason) {
      reportError(reason);
    } finally {
      finishOperation();
    }
  }

  async function sendMessage(): Promise<void> {
    const content = prompt.trim();
    if (!content || !isReady || !beginOperation("chat")) return;
    const conversation: DisplayMessage[] = [
      ...messages,
      { id: crypto.randomUUID(), role: "user", content },
    ];
    const reply: DisplayMessage = { id: crypto.randomUUID(), role: "assistant", content: "" };
    const requestMessages = conversation.map(({ role, content }) => ({ role, content }));
    setMessages([...conversation, reply]);
    setGeneration(undefined);
    setPrompt("");
    try {
      const result = await window.studio.request<ChatResult>("chat", {
        modelId: selectedId,
        messages: requestMessages,
        policy,
        maxNewTokens: 48,
        temperature: 0.2,
      });
      // Decoding the full token sequence can differ from decoding one token at a time.
      if (!result.text.trim()) {
        setMessages(messages);
        setPrompt(content);
        setError(
          "The model returned no text. The fixed context may be full. Try a shorter prompt or start a new chat.",
        );
      } else {
        setMessages([...conversation, { ...reply, content: result.text }]);
        setGeneration(result);
        if (result.stopReason === "contextLimit") {
          setNotice(
            `Reply stopped at the ${result.sequenceLength || selectedRecord?.maxSequenceLength || 64}-token context limit. Start a new chat or use a shorter prompt for more output.`,
          );
        }
      }
    } catch (reason) {
      reportError(reason);
      // A cancelled or failed reply must not become context for another request.
      setMessages(messages);
      setPrompt(content);
    } finally {
      finishOperation();
    }
  }

  async function runTests(): Promise<void> {
    if (!isReady || !beginOperation("benchmark")) return;
    try {
      const report = await window.studio.request<BenchmarkReport>("benchmark", {
        modelId: selectedId,
        prompt: "Explain why the sky appears blue in one sentence.",
        policies: policies.map((item) => item.id),
      });
      setBenchmark(report);
    } catch (reason) {
      reportError(reason);
    } finally {
      finishOperation();
    }
  }

  async function addCustomModel(): Promise<void> {
    if (busy) return;
    const id = customId.trim();
    if (id.length > 200 || !/^[A-Za-z0-9][A-Za-z0-9._-]*\/[A-Za-z0-9][A-Za-z0-9._-]*$/.test(id)) {
      setError("Enter a Hugging Face model ID such as organization/model.");
      return;
    }
    if (!beginOperation("inspect")) return;
    try {
      const inspected = await window.studio.request<{
        id: string;
        supported: boolean;
        reason?: string;
        architecture: string;
        parameterCount?: number;
        license?: string;
        revision?: string;
      }>("inspect", { modelId: id });
      if (!inspected.supported)
        throw new Error(inspected.reason || "This architecture is not supported yet.");
      if (!bootstrap?.models.some((model) => model.id === id)) {
        updateModel({
          id,
          slug: "",
          state: "available",
          architecture: inspected.architecture,
          parameterCount: inspected.parameterCount,
          license: inspected.license,
          revision: inspected.revision,
          updatedAt: new Date().toISOString(),
        });
      }
      setSelectedId(id);
      setGeneration(undefined);
      setMessages([]);
      setPrompt("");
      setBenchmark(undefined);
      setCustomId("");
      setNotice("Metadata checked. Conversion and source-to-Core-ML parity are still required.");
    } catch (reason) {
      reportError(reason);
    } finally {
      finishOperation();
    }
  }

  async function cancelOperation(): Promise<void> {
    if (!activeOperation.current || cancellingRef.current) return;
    cancellingRef.current = true;
    setCancelling(true);
    try {
      await window.studio.cancel();
      const catalog = await window.studio.request<{ models: ModelRecord[] }>("catalog");
      setBootstrap((current) => {
        if (!current) return current;
        // A cancelled metadata request may not have written a manifest yet.
        // Keep models inspected in this window until they are prepared or removed.
        const inspected = current.models.filter(
          (model) => !model.slug && !catalog.models.some((saved) => saved.id === model.id),
        );
        return { ...current, models: [...catalog.models, ...inspected] };
      });
      setNotice(
        "Operation cancelled. Files already downloaded stay on this Mac. You can try again.",
      );
    } catch (reason) {
      setError(errorMessage(reason));
    } finally {
      cancellingRef.current = false;
      setCancelling(false);
    }
  }

  async function remove(): Promise<void> {
    if (!selectedRecord || !beginOperation("remove")) return;
    try {
      await window.studio.request("remove", { modelId: selectedId });
      setBootstrap((current) =>
        current
          ? {
              ...current,
              models: current.models.filter((model) => model.id !== selectedId),
            }
          : current,
      );
      if (!selectedCurated)
        setSelectedId(bootstrap?.curatedModels[0]?.id || "LiquidAI/LFM2.5-230M");
      setMessages([]);
      setPrompt("");
      setBenchmark(undefined);
      setGeneration(undefined);
      setNotice("The local model and its results were removed.");
    } catch (reason) {
      reportError(reason);
    } finally {
      setShowRemove(false);
      finishOperation();
    }
  }

  function closeToken(): void {
    if (tokenBusy) return;
    setToken("");
    setTokenError(undefined);
    setShowToken(false);
  }

  async function saveToken(): Promise<void> {
    if (tokenBusy || !token.trim()) return;
    setTokenBusy(true);
    setTokenError(undefined);
    try {
      await window.studio.saveToken(token.trim());
      setBootstrap((current) => (current ? { ...current, hasToken: true } : current));
      setToken("");
      setShowToken(false);
      setNotice("Hugging Face token saved in encrypted local storage.");
    } catch (reason) {
      setTokenError(errorMessage(reason));
    } finally {
      setTokenBusy(false);
    }
  }

  async function deleteToken(): Promise<void> {
    if (tokenBusy) return;
    setTokenBusy(true);
    setTokenError(undefined);
    try {
      await window.studio.deleteToken();
      setBootstrap((current) => (current ? { ...current, hasToken: false } : current));
      setToken("");
      setShowToken(false);
      setNotice("The saved Hugging Face token was removed.");
    } catch (reason) {
      setTokenError(errorMessage(reason));
    } finally {
      setTokenBusy(false);
    }
  }

  if (!bootstrap) {
    return (
      <main className="boot-screen">
        <div className="brand-mark large">
          <CircuitBoard size={28} />
        </div>
        <h1>ANE Studio</h1>
        <p role={error ? "alert" : "status"}>{error || "Starting the local model worker..."}</p>
        {error ? (
          <button
            className="primary-button"
            onClick={() => setBootAttempt((current) => current + 1)}
          >
            Retry startup
          </button>
        ) : (
          <LoaderCircle className="spin" size={22} />
        )}
      </main>
    );
  }

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="window-drag" />
        <div className="brand">
          <div className="brand-mark">
            <CircuitBoard size={17} />
          </div>
          <div>
            <strong>ANE Studio</strong>
            <span>Apple Silicon model lab</span>
          </div>
        </div>
        <div className="sidebar-label">
          <span>Model bench</span>
          <small>{modelList.length}</small>
        </div>
        <div className="model-list">
          {modelList.map((model) => {
            const curated = "description" in model;
            const record = bootstrap.models.find((item) => item.id === model.id);
            const recordActive = operation === "prepare" && record?.id === selectedId;
            const recordState =
              record?.state === "ready" && !modelCanRun(record)
                ? "failed"
                : isIncomplete(record) && !recordActive
                  ? "interrupted"
                  : record?.state || "available";
            return (
              <button
                className={`model-row ${selectedId === model.id ? "selected" : ""}`}
                key={model.id}
                aria-pressed={selectedId === model.id}
                disabled={busy}
                title={model.id}
                onClick={() => selectModel(model.id)}
              >
                <span className="model-icon">
                  <Box size={15} />
                </span>
                <span className="model-row-copy">
                  <strong>{curated ? model.label : model.id.split("/").at(-1)}</strong>
                  <small>
                    {record
                      ? `${stateLabel(record, recordActive)} · ${parameterLabel(record.parameterCount || (curated ? model.parameters : undefined))}`
                      : curated
                        ? parameterLabel(model.parameters)
                        : "Not converted"}
                  </small>
                </span>
                <span className={`status-dot ${recordState}`} />
              </button>
            );
          })}
        </div>
        <div className="add-model">
          <label htmlFor="model-id">
            <Plus size={13} /> Add from Hugging Face
          </label>
          <div className="inline-input">
            <input
              id="model-id"
              disabled={busy}
              maxLength={200}
              value={customId}
              onChange={(event) => setCustomId(event.target.value)}
              onKeyDown={(event) => event.key === "Enter" && void addCustomModel()}
              placeholder="organization/model"
            />
            <button
              aria-label="Inspect model"
              onClick={() => void addCustomModel()}
              disabled={busy || !customId.trim()}
            >
              <ArrowUp size={14} />
            </button>
          </div>
        </div>
        <div className="sidebar-footer">
          <div className="machine-card">
            <Cpu size={16} />
            <div>
              <strong>{bootstrap.hardware.chip}</strong>
              <span>{bytes(bootstrap.hardware.memoryBytes)} unified memory</span>
            </div>
            <span className="online-dot" />
          </div>
          <button className="settings-button" disabled={busy} onClick={() => setShowToken(true)}>
            <Settings2 size={15} /> Credentials
            <span className="sr-only">{bootstrap.hasToken ? "Token saved" : "No token saved"}</span>
            {bootstrap.hasToken && <Check size={14} className="green" />}
          </button>
        </div>
      </aside>

      <main className="workspace">
        <header className="topbar">
          <div className="model-heading">
            <div className="eyebrow">{selectedId}</div>
            <div className="model-title-row">
              <h1>{modelName}</h1>
              <span className={`header-state ${isReady ? "ready" : "idle"}`}>
                {isReady ? "Parity passed" : stateLabel(selectedRecord, operation === "prepare")}
              </span>
            </div>
            <p>
              {selectedCurated?.description ||
                "A local Hugging Face checkpoint prepared for Core ML."}
            </p>
          </div>
          <div className="top-actions">
            <span className="local-pill">
              <ShieldCheck size={14} />
              <span>
                <small>Privacy</small>
                <strong>Local only</strong>
              </span>
            </span>
            {selectedRecord && (
              <button
                className="icon-button danger"
                title="Remove local model"
                aria-label="Remove local model"
                disabled={busy}
                onClick={() => setShowRemove(true)}
              >
                <Trash2 size={16} />
              </button>
            )}
          </div>
        </header>

        <nav className="tabs" aria-label="Model views">
          <button className={view === "chat" ? "active" : ""} onClick={() => setView("chat")}>
            <MessageSquare size={15} /> Chat
          </button>
          <button className={view === "tests" ? "active" : ""} onClick={() => setView("tests")}>
            <FlaskConical size={15} /> Tests
          </button>
          <button className={view === "details" ? "active" : ""} onClick={() => setView("details")}>
            <FileCode2 size={15} /> Details
          </button>
        </nav>

        {error && (
          <div className="error-banner" role="alert">
            <AlertTriangle size={16} />
            <span>{error}</span>
            <button aria-label="Dismiss error" onClick={() => setError(undefined)}>
              <X size={15} />
            </button>
          </div>
        )}

        {notice && (
          <div className="notice-banner" role="status">
            {notice}
          </div>
        )}
        {!bootstrap.hardware.supported && (
          <div className="error-banner" role="alert">
            <AlertTriangle size={16} />
            <span>
              {bootstrap.hardware.reason ||
                "Conversion and inference need Apple Silicon and macOS 15 or later."}
            </span>
          </div>
        )}
        {(operation || cancelling) && (
          <section className="operation-status" aria-label="Worker operation">
            <div className="progress-copy" role="status">
              <LoaderCircle className="spin" size={14} />
              <span>
                {cancelling
                  ? "Stopping the worker..."
                  : progress?.message || (operation && operationLabels[operation])}
              </span>
            </div>
            <progress
              aria-label="Operation progress"
              max={1}
              value={
                typeof progress?.progress === "number"
                  ? Math.min(1, Math.max(0, progress.progress))
                  : undefined
              }
            />
            {operation !== "remove" && (
              <button
                className="secondary-button"
                disabled={cancelling}
                onClick={() => void cancelOperation()}
              >
                {cancelling ? "Cancelling..." : "Cancel operation"}
              </button>
            )}
          </section>
        )}

        {!isReady && (
          <section className="conversion-card">
            <div className="conversion-icon">
              <Download size={24} />
            </div>
            <div className="conversion-copy">
              <span className="section-kicker">Core ML conversion</span>
              <h2>
                {selectedRecord?.state === "failed"
                  ? "Conversion needs attention"
                  : isIncomplete(selectedRecord) && !busy
                    ? "Preparation was interrupted"
                    : "Prepare this model for Apple Silicon"}
              </h2>
              <p>
                Download the pinned Hugging Face revision, convert it to Core ML, then compare its
                logits with the source model before chat is enabled.
              </p>
              {selectedCurated?.caution && (
                <p className="caution">
                  <AlertTriangle size={14} /> {selectedCurated.caution}
                </p>
              )}
            </div>
            <div className="conversion-action">
              <span className={`state-badge ${selectedRecord?.state || "available"}`}>
                {stateLabel(selectedRecord, operation === "prepare")}
              </span>
              <button
                className="primary-button"
                disabled={busy || !bootstrap.hardware.supported}
                onClick={() => void prepare()}
              >
                {operation === "prepare" ? (
                  <LoaderCircle className="spin" size={16} />
                ) : (
                  <Sparkles size={16} />
                )}
                {selectedRecord?.state === "failed"
                  ? "Try again"
                  : isIncomplete(selectedRecord) && !busy
                    ? "Convert again"
                    : "Convert to Core ML"}
              </button>
            </div>
            {selectedRecord?.error && <p className="model-error">{selectedRecord.error}</p>}
          </section>
        )}

        {view === "chat" && (
          <ChatView
            ready={isReady}
            modelName={modelName}
            sequenceLength={selectedRecord?.maxSequenceLength || 64}
            generation={generation}
            newChat={() => {
              setGeneration(undefined);
              setNotice(undefined);
              setMessages([]);
              setPrompt("");
              setError(undefined);
            }}
            messages={messages}
            prompt={prompt}
            setPrompt={setPrompt}
            send={() => void sendMessage()}
            busy={busy}
            policy={policy}
            setPolicy={setPolicy}
            endRef={endRef}
          />
        )}
        {view === "tests" && (
          <TestsView
            ready={isReady}
            busy={busy}
            running={operation === "benchmark"}
            report={benchmark}
            run={() => void runTests()}
          />
        )}
        {view === "details" && (
          <DetailsView
            record={selectedRecord}
            curated={selectedCurated}
            events={events}
            showLogs={showLogs}
            setShowLogs={setShowLogs}
          />
        )}
      </main>

      {showToken && (
        <Modal labelId="token-title" close={closeToken} busy={tokenBusy}>
          <div className="modal-header">
            <div className="modal-icon">
              <KeyRound size={19} />
            </div>
            <div>
              <h2 id="token-title">Hugging Face token</h2>
              <p>Only needed for gated or private models.</p>
            </div>
            <button aria-label="Close credentials" disabled={tokenBusy} onClick={closeToken}>
              <X size={17} />
            </button>
          </div>
          <p className="token-state">
            {bootstrap.hasToken
              ? "A token is saved. Its value is never sent back to this window."
              : "No token saved."}
          </p>
          <label htmlFor="hf-token">Access token</label>
          <input
            id="hf-token"
            type="password"
            value={token}
            disabled={tokenBusy}
            maxLength={512}
            onChange={(event) => setToken(event.target.value)}
            placeholder="hf_..."
            autoComplete="off"
            spellCheck={false}
          />
          <p className="privacy-note">
            <ShieldCheck size={14} /> Encrypted locally with Electron safeStorage and macOS
            Keychain. Used only for Hugging Face access. Never written to logs.
          </p>
          {tokenError && (
            <p className="modal-error" role="alert">
              {tokenError}
            </p>
          )}
          <div className="modal-actions">
            {bootstrap.hasToken && (
              <button
                className="text-button danger"
                disabled={tokenBusy}
                onClick={() => void deleteToken()}
              >
                Remove token
              </button>
            )}
            <span />
            <button className="secondary-button" disabled={tokenBusy} onClick={closeToken}>
              Cancel
            </button>
            <button
              className="primary-button"
              disabled={!token.trim() || tokenBusy}
              onClick={() => void saveToken()}
            >
              {tokenBusy ? "Saving changes..." : "Save token"}
            </button>
          </div>
        </Modal>
      )}
      {showRemove && (
        <Modal labelId="remove-title" busy={busy} close={() => setShowRemove(false)}>
          <div className="modal-header">
            <div className="modal-icon">
              <Trash2 size={19} />
            </div>
            <div>
              <h2 id="remove-title">Remove local model?</h2>
              <p>{modelName}</p>
            </div>
            <button
              aria-label="Close removal dialog"
              disabled={busy}
              onClick={() => setShowRemove(false)}
            >
              <X size={17} />
            </button>
          </div>
          <p className="removal-note">
            This deletes this model's downloaded weights, Core ML package, and benchmark results
            from this Mac. You will need to download and convert it again.
          </p>
          <div className="modal-actions">
            <span />
            <button
              className="secondary-button"
              disabled={busy}
              onClick={() => setShowRemove(false)}
            >
              Keep model
            </button>
            <button className="primary-button" disabled={busy} onClick={() => void remove()}>
              {busy ? "Removing..." : "Remove model"}
            </button>
          </div>
        </Modal>
      )}
    </div>
  );
}

function Modal(props: {
  labelId: string;
  close(): void;
  busy: boolean;
  children: React.ReactNode;
}): React.JSX.Element {
  const dialogRef = useRef<HTMLDialogElement>(null);
  useEffect(() => {
    const dialog = dialogRef.current;
    dialog?.showModal();
    return () => dialog?.close();
  }, []);
  return (
    <dialog
      ref={dialogRef}
      className="modal"
      aria-labelledby={props.labelId}
      onKeyDown={(event) => {
        if (event.key === "Escape") {
          event.preventDefault();
          if (!props.busy) props.close();
        }
      }}
      onCancel={(event) => {
        event.preventDefault();
        if (!props.busy) props.close();
      }}
      onClick={(event) => {
        if (event.target !== event.currentTarget || props.busy) return;
        const bounds = event.currentTarget.getBoundingClientRect();
        if (
          event.clientX < bounds.left ||
          event.clientX > bounds.right ||
          event.clientY < bounds.top ||
          event.clientY > bounds.bottom
        )
          props.close();
      }}
    >
      {props.children}
    </dialog>
  );
}

function ChatView(props: {
  ready: boolean;
  modelName: string;
  sequenceLength: number;
  newChat(): void;
  messages: DisplayMessage[];
  generation?: ChatResult;
  prompt: string;
  setPrompt(value: string): void;
  send(): void;
  busy: boolean;
  policy: ComputePolicy;
  setPolicy(value: ComputePolicy): void;
  endRef: React.RefObject<HTMLDivElement | null>;
}): React.JSX.Element {
  return (
    <section className="chat-view">
      <div className="chat-toolbar">
        <div className="policy-select">
          <label className="toolbar-label" htmlFor="compute-policy">
            Requested policy
          </label>
          <Zap size={14} />
          <select
            id="compute-policy"
            disabled={props.busy}
            value={props.policy}
            onChange={(event) => props.setPolicy(event.target.value as ComputePolicy)}
          >
            {policies.map((item) => (
              <option key={item.id} value={item.id}>
                {item.label}
              </option>
            ))}
          </select>
          <ChevronDown size={13} />
        </div>
        <span className={`session-state ${props.ready ? "ready" : ""}`}>
          <i /> {props.ready ? "Parity verified" : "Waiting for conversion"}
        </span>
        <button
          className="text-button"
          disabled={props.busy || (!props.messages.length && !props.prompt)}
          onClick={props.newChat}
        >
          New chat
        </button>
      </div>
      <p className="context-note">
        Fixed {props.sequenceLength}-token context, including the reply. Long conversations leave
        less room for output. Requested policy is not proof of accelerator use.
      </p>
      <div className="messages" role="log" aria-label="Conversation" aria-live="polite">
        {!props.messages.length && (
          <article className="message assistant">
            <div className="avatar">
              <CircuitBoard size={16} />
            </div>
            <div>
              <span>ANE Studio</span>
              <p>
                {props.ready
                  ? `${props.modelName} passed source-to-Core-ML parity. Send a message to start a local session.`
                  : "Convert a model, then send a message. Every prompt and output stays on this Mac."}
              </p>
            </div>
          </article>
        )}
        {props.messages.map((message) => (
          <article className={`message ${message.role}`} key={message.id}>
            <div className="avatar">
              {message.role === "assistant" ? <CircuitBoard size={16} /> : "You"}
            </div>
            <div>
              <span>{message.role === "assistant" ? "Model" : "You"}</span>
              <p>{message.content || <LoaderCircle className="spin" size={16} />}</p>
            </div>
          </article>
        ))}
        {props.generation?.stopReason && (
          <p className="generation-note" role="status">
            {props.generation.tokens ?? "Unknown"} output tokens.{" "}
            {props.generation.stopReason === "eos"
              ? "Stopped at the end-of-sequence token."
              : props.generation.stopReason === "contextLimit"
                ? "Stopped at the fixed context limit."
                : "Stopped at the requested output limit."}
            {props.generation.contextTokens !== undefined &&
              ` Final context used ${props.generation.contextTokens}/${props.generation.sequenceLength || props.sequenceLength} tokens.`}
          </p>
        )}
        <div ref={props.endRef} />
      </div>
      <div className={`composer ${!props.ready ? "disabled" : ""}`}>
        <textarea
          aria-label="Message the local model"
          maxLength={32768}
          value={props.prompt}
          onChange={(event) => props.setPrompt(event.target.value)}
          onKeyDown={(event) => {
            if (event.key === "Enter" && !event.shiftKey) {
              event.preventDefault();
              props.send();
            }
          }}
          disabled={!props.ready || props.busy}
          placeholder={
            props.ready ? "Message the local model..." : "Convert the model to enable chat"
          }
          rows={2}
        />
        <button
          aria-label="Send message"
          disabled={!props.ready || !props.prompt.trim() || props.busy}
          onClick={props.send}
        >
          <ArrowUp size={17} />
        </button>
        <small>
          <span>Return to send · Shift + Return for a new line</span>
          <span>
            <ShieldCheck size={10} /> Stays on this Mac
          </span>
        </small>
      </div>
    </section>
  );
}

function TestsView(props: {
  ready: boolean;
  busy: boolean;
  running: boolean;
  report?: BenchmarkReport;
  run(): void;
}): React.JSX.Element {
  const fastest = props.report?.policies
    .filter((item) => item.valid && item.tokensPerSecond)
    .sort((a, b) => (b.tokensPerSecond || 0) - (a.tokensPerSecond || 0))[0];
  const maxThroughput = Math.max(
    1,
    ...(props.report?.policies.map((item) => item.tokensPerSecond || 0) || []),
  );
  return (
    <section className="test-view">
      <div className="test-intro">
        <div>
          <span className="section-kicker">Policy comparison</span>
          <h2>Measure, do not assume</h2>
          <p>
            Run the same greedy generation through each public Core ML compute policy. Raw per-token
            samples stay in the local result file.
          </p>
        </div>
        <button
          className="primary-button"
          disabled={!props.ready || props.busy}
          onClick={props.run}
        >
          {props.running ? <LoaderCircle className="spin" size={16} /> : <Activity size={16} />} Run
          all tests
        </button>
      </div>
      {!props.report ? (
        <div className="empty-tests">
          <CircleGauge size={36} />
          <h3>No benchmark yet</h3>
          <p>
            Convert the model, then run all four policies under the same prompt and sequence length.
          </p>
        </div>
      ) : (
        <>
          <div className="metric-grid">
            <Metric
              icon={<Gauge size={17} />}
              label="Fastest policy"
              value={
                fastest
                  ? policies.find((item) => item.id === fastest.policy)?.label || fastest.policy
                  : "No valid run"
              }
            />
            <Metric
              icon={<Zap size={17} />}
              label="Decode speed"
              value={
                fastest?.tokensPerSecond
                  ? `${fastest.tokensPerSecond.toFixed(1)} tok/s`
                  : "Unavailable"
              }
            />
            <Metric
              icon={<Activity size={17} />}
              label="Median token"
              value={
                fastest?.medianTokenMs ? `${fastest.medianTokenMs.toFixed(1)} ms` : "Unavailable"
              }
            />
            <Metric
              icon={<ShieldCheck size={17} />}
              label="Observed placement"
              value="Not captured"
              warn
            />
          </div>
          <div className="policy-table">
            <div className="table-head">
              <span>Policy</span>
              <span>First token</span>
              <span>Median token</span>
              <span>Throughput</span>
              <span>Placement evidence</span>
            </div>
            {props.report.policies.map((result) => {
              const planned = result.placement.find((item) => item.evidence === "planned");
              return (
                <div className="table-row" key={result.policy}>
                  <span>
                    <strong>{policies.find((item) => item.id === result.policy)?.label}</strong>
                    <small>
                      {result.valid ? `${result.samples.length} raw samples` : result.error}
                    </small>
                  </span>
                  <span>{result.firstTokenMs ? `${result.firstTokenMs.toFixed(1)} ms` : "-"}</span>
                  <span>
                    {result.medianTokenMs ? `${result.medianTokenMs.toFixed(1)} ms` : "-"}
                  </span>
                  <span className="throughput-cell">
                    <strong>
                      {result.tokensPerSecond ? `${result.tokensPerSecond.toFixed(1)} tok/s` : "-"}
                    </strong>
                    <i>
                      <b
                        style={{
                          width: `${((result.tokensPerSecond || 0) / maxThroughput) * 100}%`,
                        }}
                      />
                    </i>
                  </span>
                  <span>
                    <Evidence placement={planned} />
                  </span>
                </div>
              );
            })}
          </div>
          <div className="evidence-note">
            <AlertTriangle size={15} />
            <p>
              Requested policy and planned operations are not proof of ANE execution. Capture an
              Instruments trace before making an observed-placement claim.
            </p>
          </div>
        </>
      )}
    </section>
  );
}

function Metric(props: {
  icon: React.ReactNode;
  label: string;
  value: string;
  warn?: boolean;
}): React.JSX.Element {
  return (
    <div className={`metric-card ${props.warn ? "warn" : ""}`}>
      <div>
        {props.icon}
        <span>{props.label}</span>
      </div>
      <strong>{props.value}</strong>
    </div>
  );
}

function Evidence({ placement }: { placement?: PlacementSummary }): React.JSX.Element {
  if (!placement?.operationCounts) return <span className="evidence-chip muted">Unavailable</span>;
  const ane = placement.operationCounts.ANE || 0;
  const total = Object.values(placement.operationCounts).reduce((sum, value) => sum + value, 0);
  return (
    <span className="evidence-chip">
      Planned ANE {ane}/{total}
    </span>
  );
}

function DetailsView(props: {
  record?: ModelRecord;
  curated?: CuratedModel;
  events: DisplayEvent[];
  showLogs: boolean;
  setShowLogs(value: boolean): void;
}): React.JSX.Element {
  const parity = props.record?.parity;
  return (
    <section className="details-view">
      <div className="detail-grid">
        <Detail
          icon={<FileCode2 size={17} />}
          label="Architecture"
          value={
            props.record?.architecture || props.curated?.architecture || "Inspect before conversion"
          }
        />
        <Detail
          icon={<MemoryStick size={17} />}
          label="Parameters"
          value={parameterLabel(props.record?.parameterCount || props.curated?.parameters)}
        />
        <Detail
          icon={<HardDrive size={17} />}
          label="Core ML artifact"
          value={bytes(props.record?.artifactBytes)}
        />
        <Detail
          icon={<ShieldCheck size={17} />}
          label="Model license"
          value={props.record?.license || props.curated?.license || "Unknown"}
        />
      </div>
      <div className="detail-panels">
        <section className="detail-panel">
          <div className="panel-heading">
            <div>
              <span className="section-kicker">Correctness gate</span>
              <h3>Source versus Core ML</h3>
            </div>
            <span
              className={`state-badge ${hasPassedParity(parity) ? "passed" : parity?.status === "passed" ? "failed" : parity?.status || "pending"}`}
            >
              {parity?.status === "passed" && !hasPassedParity(parity)
                ? "Invalid evidence"
                : parity?.status || "Not run"}
            </span>
          </div>
          <dl>
            <div>
              <dt>Cosine similarity</dt>
              <dd>{parity?.cosineSimilarity?.toFixed(6) || "-"}</dd>
            </div>
            <div>
              <dt>Maximum absolute error</dt>
              <dd>{parity?.maximumAbsoluteError?.toFixed(6) || "-"}</dd>
            </div>
            <div>
              <dt>Top token matches</dt>
              <dd>
                {parity?.topTokenMatches === undefined
                  ? "-"
                  : parity.topTokenMatches
                    ? "Yes"
                    : "No"}
              </dd>
            </div>
            <div>
              <dt>Artifact SHA-256</dt>
              <dd className="mono">{props.record?.artifactSha256 || "-"}</dd>
            </div>
          </dl>
        </section>
        <section className="detail-panel">
          <div className="panel-heading">
            <div>
              <span className="section-kicker">Provenance</span>
              <h3>Pinned source</h3>
            </div>
            <ExternalLink size={15} />
          </div>
          <dl>
            <div>
              <dt>Model</dt>
              <dd>{props.record?.id || props.curated?.id}</dd>
            </div>
            <div>
              <dt>Revision</dt>
              <dd className="mono">{props.record?.revision || "Resolved during download"}</dd>
            </div>
            <div>
              <dt>Precision</dt>
              <dd title={precisionLabel(parity?.computePrecision)}>
                {precisionLabel(parity?.computePrecision)}
              </dd>
            </div>
            <div>
              <dt>Sequence length</dt>
              <dd>{props.record?.maxSequenceLength || 64} tokens</dd>
            </div>
          </dl>
        </section>
      </div>
      <section className="logs-panel">
        <button aria-expanded={props.showLogs} onClick={() => props.setShowLogs(!props.showLogs)}>
          <TerminalSquare size={15} />
          <span>Technical log</span>
          <small>{props.events.length} events</small>
          <ChevronDown size={15} className={props.showLogs ? "open" : ""} />
        </button>
        {props.showLogs && (
          <div className="logs">
            {props.events.length ? (
              props.events.map((item) => (
                <p key={item.id}>
                  <time>{new Date(item.at).toLocaleTimeString()}</time>
                  <span className={item.level}>{item.stage || item.kind}</span>
                  {item.message || item.token || JSON.stringify(item.model)}
                </p>
              ))
            ) : (
              <p>No worker events yet.</p>
            )}
          </div>
        )}
      </section>
    </section>
  );
}

function Detail(props: { icon: React.ReactNode; label: string; value: string }): React.JSX.Element {
  return (
    <div className="detail-card">
      <span>{props.icon}</span>
      <div>
        <small>{props.label}</small>
        <strong>{props.value}</strong>
      </div>
    </div>
  );
}

export default App;
