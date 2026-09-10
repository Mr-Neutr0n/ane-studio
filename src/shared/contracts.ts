export const COMPUTE_POLICIES = ["cpuOnly", "cpuAndGPU", "cpuAndNeuralEngine", "all"] as const;
export type ComputePolicy = (typeof COMPUTE_POLICIES)[number];
export type StudioCommand =
  | "catalog"
  | "inspect"
  | "prepare"
  | "chat"
  | "benchmark"
  | "placement"
  | "remove";

export const MAX_PROMPT_CHARACTERS = 32_768;
export const MAX_CHAT_MESSAGES = 128;

const commandFields: Record<StudioCommand, readonly string[]> = {
  catalog: [],
  inspect: ["modelId"],
  prepare: ["modelId", "sequenceLength"],
  chat: ["modelId", "messages", "policy", "maxNewTokens", "temperature"],
  benchmark: ["modelId", "prompt", "policies"],
  placement: ["modelId", "policy"],
  remove: ["modelId"],
};

function isRecord(value: unknown): value is Record<string, unknown> {
  if (value === null || typeof value !== "object" || Array.isArray(value)) return false;
  const prototype = Object.getPrototypeOf(value);
  return prototype === Object.prototype || prototype === null;
}

export function normalizeModelId(value: unknown): string {
  if (typeof value !== "string" || value.length > 200) {
    throw new Error("Enter a Hugging Face model ID such as organization/model.");
  }
  const id = value.trim();
  const component = "[A-Za-z0-9_](?:[A-Za-z0-9_.-]{0,94}[A-Za-z0-9_])?";
  if (
    !new RegExp(`^${component}/${component}$`).test(id) ||
    id.includes("..") ||
    id.includes("--") ||
    id.endsWith(".git")
  ) {
    throw new Error("Enter a Hugging Face model ID such as organization/model.");
  }
  return id;
}

export function normalizeHuggingFaceToken(value: unknown): string {
  if (typeof value !== "string" || value.length > 512) {
    throw new Error("Enter a valid Hugging Face token beginning with hf_.");
  }
  const token = value.trim();
  if (!/^hf_[A-Za-z0-9]{17,253}$/.test(token)) {
    throw new Error("Enter a valid Hugging Face token beginning with hf_.");
  }
  return token;
}

function computePolicy(value: unknown): ComputePolicy {
  if (!COMPUTE_POLICIES.includes(value as ComputePolicy)) {
    throw new Error("Select a supported Core ML compute policy.");
  }
  return value as ComputePolicy;
}

function boundedNumber(value: unknown, minimum: number, maximum: number, integer: boolean): number {
  if (
    typeof value !== "number" ||
    !Number.isFinite(value) ||
    (integer && !Number.isInteger(value)) ||
    value < minimum ||
    value > maximum
  ) {
    throw new Error(
      `Expected ${integer ? "an integer" : "a number"} between ${minimum} and ${maximum}.`,
    );
  }
  return value;
}

/** Validate again in main. TypeScript and the preload are not a trust boundary. */
export function validateStudioRequest(
  command: unknown,
  payload: unknown = {},
): { command: StudioCommand; payload: Record<string, unknown> } {
  if (typeof command !== "string" || !Object.hasOwn(commandFields, command)) {
    throw new Error("Unsupported worker command.");
  }
  const validatedCommand = command as StudioCommand;
  const fields = commandFields[validatedCommand];
  if (!isRecord(payload) || Object.keys(payload).some((key) => !fields.includes(key))) {
    throw new Error("Invalid worker request payload.");
  }
  const result: Record<string, unknown> = {};
  if (command !== "catalog") result.modelId = normalizeModelId(payload.modelId);
  if (payload.policy !== undefined) result.policy = computePolicy(payload.policy);
  if (payload.sequenceLength !== undefined) {
    result.sequenceLength = boundedNumber(payload.sequenceLength, 16, 256, true);
  }
  if (payload.maxNewTokens !== undefined) {
    result.maxNewTokens = boundedNumber(payload.maxNewTokens, 1, 128, true);
  }
  if (payload.temperature !== undefined) {
    result.temperature = boundedNumber(payload.temperature, 0, 2, false);
  }
  if (command === "chat") {
    if (
      !Array.isArray(payload.messages) ||
      payload.messages.length === 0 ||
      payload.messages.length > MAX_CHAT_MESSAGES
    ) {
      throw new Error(`Chat needs between 1 and ${MAX_CHAT_MESSAGES} messages.`);
    }
    let characters = 0;
    result.messages = payload.messages.map((message: unknown): ChatMessage => {
      if (
        !isRecord(message) ||
        Object.keys(message).some((key) => key !== "role" && key !== "content") ||
        !["system", "user", "assistant"].includes(message.role as string) ||
        typeof message.content !== "string"
      ) {
        throw new Error("Each chat message needs a valid role and text content.");
      }
      characters += message.content.length;
      if (characters > MAX_PROMPT_CHARACTERS) {
        throw new Error(`Chat text must not exceed ${MAX_PROMPT_CHARACTERS} characters.`);
      }
      return { role: message.role as ChatMessage["role"], content: message.content };
    });
    if (!(result.messages as ChatMessage[]).some((message) => message.content.trim())) {
      throw new Error("Chat needs at least one non-empty message.");
    }
  }
  if (payload.prompt !== undefined) {
    if (
      typeof payload.prompt !== "string" ||
      !payload.prompt.trim() ||
      payload.prompt.length > MAX_PROMPT_CHARACTERS
    ) {
      throw new Error(`Benchmark prompt must contain 1 to ${MAX_PROMPT_CHARACTERS} characters.`);
    }
    result.prompt = payload.prompt;
  }
  if (payload.policies !== undefined) {
    if (
      !Array.isArray(payload.policies) ||
      payload.policies.length === 0 ||
      payload.policies.length > COMPUTE_POLICIES.length ||
      new Set(payload.policies).size !== payload.policies.length
    ) {
      throw new Error("Choose each benchmark compute policy at most once.");
    }
    result.policies = payload.policies.map(computePolicy);
  }
  return { command: validatedCommand, payload: result };
}

export type ModelState =
  | "available"
  | "inspecting"
  | "downloading"
  | "converting"
  | "validating"
  | "ready"
  | "failed";

export interface CuratedModel {
  id: string;
  label: string;
  description: string;
  parameters: number;
  architecture: string;
  license: string;
  releasedAt: string;
  recommended: boolean;
  caution?: string;
}

export interface ParityResult {
  status: "pending" | "passed" | "failed";
  computePrecision?: string;
  cosineSimilarity?: number;
  maximumAbsoluteError?: number;
  topTokenMatches?: boolean;
  sourceToken?: number;
  coremlToken?: number;
}

export interface PlacementSummary {
  evidence: "requested" | "planned" | "observed";
  policy: ComputePolicy;
  observed: boolean;
  operationCounts?: Record<string, number>;
  note: string;
}

export interface ModelRecord {
  id: string;
  slug: string;
  state: ModelState;
  architecture?: string;
  parameterCount?: number;
  license?: string;
  revision?: string;
  maxSequenceLength?: number;
  artifactBytes?: number;
  artifactSha256?: string;
  parity?: ParityResult;
  placement?: PlacementSummary[];
  error?: string;
  updatedAt: string;
}

export interface HardwareInfo {
  supported: boolean;
  platform: string;
  architecture: string;
  osVersion: string;
  chip: string;
  memoryBytes: number;
  powerSource?: string;
  reason?: string;
}

export interface BenchmarkPolicyResult {
  policy: ComputePolicy;
  valid: boolean;
  firstTokenMs?: number;
  medianTokenMs?: number;
  tokensPerSecond?: number;
  samples: number[];
  placement: PlacementSummary[];
  error?: string;
}

export interface BenchmarkReport {
  modelId: string;
  prompt: string;
  createdAt: string;
  policies: BenchmarkPolicyResult[];
}

export interface ChatMessage {
  role: "user" | "assistant" | "system";
  content: string;
}

export interface WorkerEvent {
  kind: "progress" | "log" | "token" | "model";
  requestId: string;
  stage?: string;
  progress?: number;
  message?: string;
  token?: string;
  level?: "debug" | "info" | "warning" | "error";
  model?: ModelRecord;
  at: string;
}

export interface WorkerResponse<T = unknown> {
  id: string;
  ok: boolean;
  result?: T;
  error?: {
    code: string;
    message: string;
    detail?: string;
  };
}

export interface BootstrapData {
  hardware: HardwareInfo;
  curatedModels: CuratedModel[];
  models: ModelRecord[];
  hasToken: boolean;
  dataDirectory: string;
}

export interface StudioAPI {
  bootstrap(): Promise<BootstrapData>;
  request<T>(command: StudioCommand, payload?: Record<string, unknown>): Promise<T>;
  cancel(): Promise<void>;
  saveToken(token: string): Promise<void>;
  deleteToken(): Promise<void>;
  onWorkerEvent(callback: (event: WorkerEvent) => void): () => void;
}

export function hasPassedParity(parity: ParityResult | null | undefined): boolean {
  return (
    parity?.status === "passed" &&
    typeof parity.cosineSimilarity === "number" &&
    Number.isFinite(parity.cosineSimilarity) &&
    parity.cosineSimilarity >= 0.999 &&
    parity.topTokenMatches === true
  );
}

function isPlacement(value: unknown): value is PlacementSummary {
  return (
    isRecord(value) &&
    ["requested", "planned", "observed"].includes(value.evidence as string) &&
    COMPUTE_POLICIES.includes(value.policy as ComputePolicy) &&
    typeof value.observed === "boolean" &&
    (!value.observed || value.evidence === "observed") &&
    typeof value.note === "string" &&
    (value.operationCounts === undefined ||
      (isRecord(value.operationCounts) &&
        Object.values(value.operationCounts).every(
          (count) => typeof count === "number" && Number.isSafeInteger(count) && count >= 0,
        )))
  );
}

function isModelRecord(value: unknown): value is ModelRecord {
  if (
    !isRecord(value) ||
    typeof value.id !== "string" ||
    typeof value.slug !== "string" ||
    typeof value.updatedAt !== "string" ||
    ![
      "available",
      "inspecting",
      "downloading",
      "converting",
      "validating",
      "ready",
      "failed",
    ].includes(value.state as string)
  )
    return false;
  for (const key of ["architecture", "license", "revision", "artifactSha256", "error"]) {
    if (value[key] !== undefined && typeof value[key] !== "string") return false;
  }
  for (const key of ["parameterCount", "maxSequenceLength", "artifactBytes"]) {
    if (
      value[key] !== undefined &&
      (typeof value[key] !== "number" || !Number.isFinite(value[key]))
    ) {
      return false;
    }
  }
  if (value.parity !== undefined) {
    const parity = value.parity;
    if (!isRecord(parity) || !["pending", "passed", "failed"].includes(parity.status as string)) {
      return false;
    }
    for (const key of ["cosineSimilarity", "maximumAbsoluteError", "sourceToken", "coremlToken"]) {
      if (
        parity[key] !== undefined &&
        (typeof parity[key] !== "number" || !Number.isFinite(parity[key]))
      ) {
        return false;
      }
    }
    if (parity.topTokenMatches !== undefined && typeof parity.topTokenMatches !== "boolean")
      return false;
    if (parity.computePrecision !== undefined && typeof parity.computePrecision !== "string")
      return false;
  }
  return (
    value.placement === undefined ||
    (Array.isArray(value.placement) && value.placement.every(isPlacement))
  );
}

/** Never forward Electron's event object or malformed worker data into the page. */
export function isWorkerEvent(value: unknown): value is WorkerEvent {
  if (
    !isRecord(value) ||
    !["progress", "log", "token", "model"].includes(value.kind as string) ||
    typeof value.requestId !== "string" ||
    !value.requestId ||
    typeof value.at !== "string" ||
    !Number.isFinite(Date.parse(value.at))
  )
    return false;
  for (const key of ["stage", "message", "token"]) {
    if (value[key] !== undefined && typeof value[key] !== "string") return false;
  }
  if (
    value.level !== undefined &&
    !["debug", "info", "warning", "error"].includes(value.level as string)
  ) {
    return false;
  }
  if (
    value.progress !== undefined &&
    (typeof value.progress !== "number" ||
      !Number.isFinite(value.progress) ||
      value.progress < 0 ||
      value.progress > 1)
  ) {
    return false;
  }
  if (value.model !== undefined && !isModelRecord(value.model)) return false;
  return (
    (value.kind !== "model" || value.model !== undefined) &&
    (value.kind !== "token" || value.token !== undefined)
  );
}
