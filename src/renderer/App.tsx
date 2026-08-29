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
import { useEffect, useMemo, useRef, useState } from "react";
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

type View = "chat" | "tests" | "details";

const policies: { id: ComputePolicy; label: string; short: string }[] = [
  { id: "cpuOnly", label: "CPU only", short: "CPU" },
  { id: "cpuAndGPU", label: "CPU + GPU", short: "GPU" },
  { id: "cpuAndNeuralEngine", label: "CPU + Neural Engine", short: "ANE" },
  { id: "all", label: "Automatic", short: "All" },
];

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

function stateLabel(record?: ModelRecord): string {
  if (!record) return "Not converted";
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

function App(): React.JSX.Element {
  const [bootstrap, setBootstrap] = useState<BootstrapData>();
  const [error, setError] = useState<string>();
  const [selectedId, setSelectedId] = useState("LiquidAI/LFM2.5-230M");
  const [customId, setCustomId] = useState("");
  const [view, setView] = useState<View>("chat");
  const [busy, setBusy] = useState(false);
  const [events, setEvents] = useState<WorkerEvent[]>([]);
  const [progress, setProgress] = useState<WorkerEvent>();
  const [messages, setMessages] = useState<ChatMessage[]>([
    {
      role: "assistant",
      content: "Convert a model, then send a message. Every prompt and output stays on this Mac.",
    },
  ]);
  const [prompt, setPrompt] = useState("");
  const [policy, setPolicy] = useState<ComputePolicy>("cpuAndNeuralEngine");
  const [benchmark, setBenchmark] = useState<BenchmarkReport>();
  const [showLogs, setShowLogs] = useState(false);
  const [showToken, setShowToken] = useState(false);
  const [token, setToken] = useState("");
  const endRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    void window.studio
      .bootstrap()
      .then(setBootstrap)
      .catch((reason: Error) => setError(reason.message));
    window.studio.onWorkerEvent((workerEvent) => {
      setEvents((current) => [...current.slice(-299), workerEvent]);
      if (workerEvent.kind === "progress") setProgress(workerEvent);
      if (workerEvent.kind === "model" && workerEvent.model) {
        const model = workerEvent.model;
        setBootstrap((current) => {
          if (!current) return current;
          const models = current.models.filter((item) => item.id !== model.id);
          return { ...current, models: [...models, model] };
        });
      }
      if (workerEvent.kind === "token" && workerEvent.token) {
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
  }, []);

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const selectedRecord = bootstrap?.models.find((model) => model.id === selectedId);
  const selectedCurated = bootstrap?.curatedModels.find((model) => model.id === selectedId);
  const modelName = selectedCurated?.label || selectedId.split("/").at(-1) || selectedId;
  const isReady = selectedRecord?.state === "ready" && selectedRecord.parity?.status === "passed";

  useEffect(() => {
    if (!isReady) return;
    setMessages((current) => {
      if (
        current.length === 1 &&
        current[0]?.role === "assistant" &&
        current[0].content.startsWith("Convert a model")
      ) {
        return [
          {
            role: "assistant",
            content: `${modelName} is loaded through Core ML. What would you like to ask?`,
          },
        ];
      }
      return current;
    });
  }, [isReady, modelName]);

  const modelList = useMemo(() => {
    if (!bootstrap) return [];
    const known = new Map<string, CuratedModel | ModelRecord>();
    for (const model of bootstrap.curatedModels) known.set(model.id, model);
    for (const model of bootstrap.models) if (!known.has(model.id)) known.set(model.id, model);
    return [...known.values()];
  }, [bootstrap]);

  async function prepare(): Promise<void> {
    setBusy(true);
    setError(undefined);
    try {
      await window.studio.request("prepare", { modelId: selectedId, sequenceLength: 64 });
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : String(reason));
    } finally {
      setBusy(false);
    }
  }

  async function sendMessage(): Promise<void> {
    const content = prompt.trim();
    if (!content || !isReady || busy) return;
    const requestMessages: ChatMessage[] = [
      ...messages.filter((message, index) => !(index === 0 && message.role === "assistant")),
      { role: "user", content },
    ];
    setMessages([...requestMessages, { role: "assistant", content: "" }]);
    setPrompt("");
    setBusy(true);
    setError(undefined);
    try {
      const result = await window.studio.request<{ text: string }>("chat", {
        modelId: selectedId,
        messages: requestMessages,
        policy,
        maxNewTokens: 48,
        temperature: 0.2,
      });
      setMessages((current) => {
        const next = [...current];
        const last = next.at(-1);
        if (last?.role === "assistant" && !last.content)
          next[next.length - 1] = { ...last, content: result.text };
        return next;
      });
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : String(reason));
      setMessages((current) => current.slice(0, -1));
    } finally {
      setBusy(false);
    }
  }

  async function runTests(): Promise<void> {
    if (!isReady || busy) return;
    setBusy(true);
    setError(undefined);
    try {
      const report = await window.studio.request<BenchmarkReport>("benchmark", {
        modelId: selectedId,
        prompt: "Explain why the sky appears blue in one sentence.",
        policies: policies.map((item) => item.id),
      });
      setBenchmark(report);
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : String(reason));
    } finally {
      setBusy(false);
    }
  }

  async function addCustomModel(): Promise<void> {
    const id = customId.trim();
    if (!id.includes("/")) {
      setError("Enter a Hugging Face model ID such as organization/model.");
      return;
    }
    setBusy(true);
    try {
      const inspected = await window.studio.request<{
        id: string;
        supported: boolean;
        reason?: string;
        architecture: string;
        parameterCount?: number;
        license?: string;
      }>("inspect", { modelId: id });
      if (!inspected.supported)
        throw new Error(inspected.reason || "This architecture is not supported yet.");
      setSelectedId(id);
      setCustomId("");
      setError(undefined);
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : String(reason));
    } finally {
      setBusy(false);
    }
  }

  async function remove(): Promise<void> {
    if (!selectedRecord || busy) return;
    setBusy(true);
    await window.studio.request("remove", { modelId: selectedId });
    setBootstrap((current) =>
      current
        ? { ...current, models: current.models.filter((model) => model.id !== selectedId) }
        : current,
    );
    setMessages([{ role: "assistant", content: "The local model and its results were removed." }]);
    setBenchmark(undefined);
    setBusy(false);
  }

  async function saveToken(): Promise<void> {
    try {
      await window.studio.saveToken(token);
      setBootstrap((current) => (current ? { ...current, hasToken: true } : current));
      setToken("");
      setShowToken(false);
      setError(undefined);
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : String(reason));
    }
  }

  if (!bootstrap) {
    return (
      <main className="boot-screen">
        <div className="brand-mark large">
          <CircuitBoard size={28} />
        </div>
        <h1>ANE Studio</h1>
        <p>{error || "Starting the local model worker..."}</p>
        {!error && <LoaderCircle className="spin" size={22} />}
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
            return (
              <button
                className={`model-row ${selectedId === model.id ? "selected" : ""}`}
                key={model.id}
                onClick={() => {
                  setSelectedId(model.id);
                  setBenchmark(undefined);
                }}
              >
                <span className="model-icon">
                  <Box size={15} />
                </span>
                <span className="model-row-copy">
                  <strong>{curated ? model.label : model.id.split("/").at(-1)}</strong>
                  <small>
                    {record?.state === "ready"
                      ? `Ready · ${parameterLabel(record.parameterCount || (curated ? model.parameters : undefined))}`
                      : curated
                        ? parameterLabel(model.parameters)
                        : stateLabel(record)}
                  </small>
                </span>
                <span className={`status-dot ${record?.state || "available"}`} />
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
              value={customId}
              onChange={(event) => setCustomId(event.target.value)}
              onKeyDown={(event) => event.key === "Enter" && void addCustomModel()}
              placeholder="organization/model"
            />
            <button onClick={() => void addCustomModel()} disabled={busy}>
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
          <button className="settings-button" onClick={() => setShowToken(true)}>
            <Settings2 size={15} /> Credentials
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
                {isReady ? "Parity passed" : stateLabel(selectedRecord)}
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
                onClick={() => void remove()}
              >
                <Trash2 size={16} />
              </button>
            )}
          </div>
        </header>

        <nav className="tabs">
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
          <div className="error-banner">
            <AlertTriangle size={16} />
            <span>{error}</span>
            <button onClick={() => setError(undefined)}>
              <X size={15} />
            </button>
          </div>
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
                  : "Prepare this model for Apple Silicon"}
              </h2>
              <p>
                Download the pinned Hugging Face revision, convert it to FP16, then compare its
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
                {stateLabel(selectedRecord)}
              </span>
              <button className="primary-button" disabled={busy} onClick={() => void prepare()}>
                {busy ? <LoaderCircle className="spin" size={16} /> : <Sparkles size={16} />}
                {selectedRecord?.state === "failed" ? "Try again" : "Convert to Core ML"}
              </button>
            </div>
            {busy && progress && (
              <div className="progress-row">
                <div className="progress-track">
                  <span style={{ width: `${Math.max(3, (progress.progress || 0) * 100)}%` }} />
                </div>
                <span>{progress.message}</span>
                <button onClick={() => void window.studio.cancel()}>Cancel</button>
              </div>
            )}
          </section>
        )}

        {view === "chat" && (
          <ChatView
            ready={isReady}
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
          <TestsView ready={isReady} busy={busy} report={benchmark} run={() => void runTests()} />
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
        <div className="modal-backdrop" onMouseDown={() => setShowToken(false)}>
          <section className="modal" onMouseDown={(event) => event.stopPropagation()}>
            <div className="modal-header">
              <div className="modal-icon">
                <KeyRound size={19} />
              </div>
              <div>
                <h2>Hugging Face token</h2>
                <p>Only needed for gated or private models.</p>
              </div>
              <button onClick={() => setShowToken(false)}>
                <X size={17} />
              </button>
            </div>
            <label htmlFor="hf-token">Access token</label>
            <input
              id="hf-token"
              type="password"
              value={token}
              onChange={(event) => setToken(event.target.value)}
              placeholder="hf_..."
              autoComplete="off"
            />
            <p className="privacy-note">
              <ShieldCheck size={14} /> Encrypted with macOS Keychain through Electron safeStorage.
              Never written to logs.
            </p>
            <div className="modal-actions">
              {bootstrap.hasToken && (
                <button
                  className="text-button danger"
                  onClick={() =>
                    void window.studio
                      .deleteToken()
                      .then(() => setBootstrap({ ...bootstrap, hasToken: false }))
                  }
                >
                  Remove token
                </button>
              )}
              <span />
              <button className="secondary-button" onClick={() => setShowToken(false)}>
                Cancel
              </button>
              <button
                className="primary-button"
                disabled={!token.trim()}
                onClick={() => void saveToken()}
              >
                Save token
              </button>
            </div>
          </section>
        </div>
      )}
    </div>
  );
}

function ChatView(props: {
  ready: boolean;
  messages: ChatMessage[];
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
          <span className="toolbar-label">Compute route</span>
          <Zap size={14} />
          <select
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
          <i /> {props.ready ? "Core ML session ready" : "Waiting for conversion"}
        </span>
      </div>
      <div className="messages">
        {props.messages.map((message, index) => (
          <article className={`message ${message.role}`} key={`${message.role}-${index}`}>
            <div className="avatar">
              {message.role === "assistant" ? <CircuitBoard size={16} /> : "H"}
            </div>
            <div>
              <span>{message.role === "assistant" ? "Model" : "You"}</span>
              <p>{message.content || <LoaderCircle className="spin" size={16} />}</p>
            </div>
          </article>
        ))}
        <div ref={props.endRef} />
      </div>
      <div className={`composer ${!props.ready ? "disabled" : ""}`}>
        <textarea
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
        <button disabled={!props.ready || !props.prompt.trim() || props.busy} onClick={props.send}>
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
          {props.busy ? <LoaderCircle className="spin" size={16} /> : <Activity size={16} />} Run
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
  events: WorkerEvent[];
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
            <span className={`state-badge ${parity?.status || "pending"}`}>
              {parity?.status || "Not run"}
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
              <dd>FP16 compute</dd>
            </div>
            <div>
              <dt>Sequence length</dt>
              <dd>{props.record?.maxSequenceLength || 64} tokens</dd>
            </div>
          </dl>
        </section>
      </div>
      <section className="logs-panel">
        <button onClick={() => props.setShowLogs(!props.showLogs)}>
          <TerminalSquare size={15} />
          <span>Technical log</span>
          <small>{props.events.length} events</small>
          <ChevronDown size={15} className={props.showLogs ? "open" : ""} />
        </button>
        {props.showLogs && (
          <div className="logs">
            {props.events.length ? (
              props.events.map((item, index) => (
                <p key={`${item.at}-${index}`}>
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
