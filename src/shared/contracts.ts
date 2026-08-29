export type ComputePolicy = "cpuOnly" | "cpuAndGPU" | "cpuAndNeuralEngine" | "all";

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
  request<T>(command: string, payload?: Record<string, unknown>): Promise<T>;
  cancel(): Promise<void>;
  saveToken(token: string): Promise<void>;
  deleteToken(): Promise<void>;
  onWorkerEvent(callback: (event: WorkerEvent) => void): void;
}
