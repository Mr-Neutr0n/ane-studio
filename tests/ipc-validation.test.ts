import { describe, expect, it } from "vitest";
import {
  COMPUTE_POLICIES,
  hasPassedParity,
  isWorkerEvent,
  MAX_CHAT_MESSAGES,
  MAX_PROMPT_CHARACTERS,
  normalizeHuggingFaceToken,
  normalizeModelId,
  validateStudioRequest,
} from "../src/shared/contracts";

const modelId = "LiquidAI/LFM2.5-230M";
const messages = [{ role: "user", content: "Explain gravity." }];

describe("runtime IPC request validation", () => {
  it.each(["health", "__proto__", "constructor", "toString", "shell", "", null, 5])(
    "rejects commands outside the renderer allowlist: %s",
    (command) => {
      expect(() => validateStudioRequest(command)).toThrow("Unsupported worker command");
    },
  );

  it.each([null, [], "payload", 7, true, new Date(), { path: "/tmp/model" }, { token: "secret" }])(
    "rejects non-object and extra payload fields",
    (payload) => {
      expect(() => validateStudioRequest("catalog", payload)).toThrow();
    },
  );

  it("rejects prototype keys and copies only validated own data", () => {
    expect(() => validateStudioRequest("catalog", JSON.parse('{"__proto__":{}}'))).toThrow();
    expect(() => validateStudioRequest("inspect", Object.create({ modelId }))).toThrow();
    const input = { modelId: `  ${modelId}  `, messages };
    const validated = validateStudioRequest("chat", input);
    expect(validated.payload).toEqual({ modelId, messages });
    expect(validated.payload).not.toBe(input);
    expect(validated.payload.messages).not.toBe(messages);
    expect((validated.payload.messages as unknown[])[0]).not.toBe(messages[0]);
  });

  it.each([
    "../model",
    "org/../model",
    "org/model/extra",
    "https://huggingface.co/org/model",
    "/tmp/model",
    "org\\model",
    "org/model?token=x",
    "org/model#main",
    "org/%2e%2e",
    "org/model.git",
    "org/model--name",
    "org/model..name",
    "org/-model",
    "org/model-",
    "org/.model",
    "org/model.",
    "org/a b",
    "org/model\nother",
    `org/${"x".repeat(97)}`,
  ])("rejects URL, local-path, and malformed model identifiers: %s", (id) => {
    expect(() => normalizeModelId(id)).toThrow();
  });

  it.each([modelId, "Qwen/Qwen2.5-0.5B-Instruct", "org_name/model.name_v2"])(
    "accepts exact Hugging Face model identifiers: %s",
    (id) => expect(normalizeModelId(id)).toBe(id),
  );

  it.each([NaN, Infinity, -Infinity, true, "64", 15, 257, 64.5])(
    "requires bounded integer sequence lengths",
    (sequenceLength) => {
      expect(() => validateStudioRequest("prepare", { modelId, sequenceLength })).toThrow();
    },
  );

  it.each([NaN, Infinity, true, "12", 0, 129, 1.5])(
    "requires bounded integer output lengths",
    (maxNewTokens) => {
      expect(() => validateStudioRequest("chat", { modelId, messages, maxNewTokens })).toThrow();
    },
  );

  it.each([NaN, Infinity, true, "0.2", -0.1, 2.01])(
    "requires finite bounded temperatures",
    (temperature) => {
      expect(() => validateStudioRequest("chat", { modelId, messages, temperature })).toThrow();
    },
  );

  it("accepts the numeric boundary values without coercion", () => {
    for (const sequenceLength of [16, 256]) {
      expect(
        validateStudioRequest("prepare", { modelId, sequenceLength }).payload.sequenceLength,
      ).toBe(sequenceLength);
    }
    for (const maxNewTokens of [1, 128]) {
      for (const temperature of [0, 2]) {
        expect(
          validateStudioRequest("chat", { modelId, messages, maxNewTokens, temperature }).payload,
        ).toEqual({ modelId, messages, maxNewTokens, temperature });
      }
    }
  });

  it.each([
    undefined,
    [],
    "text",
    [null],
    [{ role: "tool", content: "x" }],
    [{ role: "user", content: {} }],
    [{ role: "user", content: " " }],
    [{ role: "user", content: "x", tool: "shell" }],
  ])("rejects malformed chats", (input) => {
    expect(() => validateStudioRequest("chat", { modelId, messages: input })).toThrow();
  });

  it("bounds chat history and total prompt characters", () => {
    expect(() =>
      validateStudioRequest("chat", {
        modelId,
        messages: Array(MAX_CHAT_MESSAGES + 1).fill(messages[0]),
      }),
    ).toThrow();
    const large = { role: "user", content: "x".repeat(MAX_PROMPT_CHARACTERS) };
    expect(validateStudioRequest("chat", { modelId, messages: [large] }).payload.messages).toEqual([
      large,
    ]);
    expect(() =>
      validateStudioRequest("chat", { modelId, messages: [large, messages[0]] }),
    ).toThrow();
  });

  it.each([[], ["all", "all"], ["gpu"], "all", [null], Array(5).fill("all")])(
    "rejects invalid benchmark policy sets",
    (policies) => {
      expect(() => validateStudioRequest("benchmark", { modelId, policies })).toThrow();
    },
  );

  it("accepts all four explicit policies and bounds benchmark prompts", () => {
    for (const policy of COMPUTE_POLICIES) {
      expect(validateStudioRequest("placement", { modelId, policy }).payload.policy).toBe(policy);
    }
    expect(
      validateStudioRequest("benchmark", { modelId, policies: [...COMPUTE_POLICIES] }).payload
        .policies,
    ).toEqual(COMPUTE_POLICIES);
    for (const prompt of ["", " ", 1, "x".repeat(MAX_PROMPT_CHARACTERS + 1)]) {
      expect(() => validateStudioRequest("benchmark", { modelId, prompt })).toThrow();
    }
  });
});

describe("credential inputs", () => {
  it("normalizes valid tokens without broadening the allowed alphabet", () => {
    const token = `hf_${"a".repeat(34)}`;
    expect(normalizeHuggingFaceToken(` ${token}\n`)).toBe(token);
  });
  it.each([
    null,
    {},
    123,
    "hf_short",
    `hf_${"x".repeat(254)}`,
    `hf_${"x".repeat(20)} secret`,
    `hf_${"x".repeat(20)}\nextra`,
  ])("rejects invalid credentials without reflecting them", (token) => {
    expect(() => normalizeHuggingFaceToken(token)).toThrow(
      "Enter a valid Hugging Face token beginning with hf_.",
    );
  });
});

describe("parity and event evidence", () => {
  it("never treats a status string alone as proof of parity", () => {
    expect(hasPassedParity({ status: "passed" })).toBe(false);
    for (const cosineSimilarity of [undefined, NaN, Infinity, 0.998999]) {
      expect(hasPassedParity({ status: "passed", cosineSimilarity, topTokenMatches: true })).toBe(
        false,
      );
    }
    expect(
      hasPassedParity({ status: "passed", cosineSimilarity: 0.999, topTokenMatches: false }),
    ).toBe(false);
    expect(hasPassedParity({ status: "failed", cosineSimilarity: 1, topTokenMatches: true })).toBe(
      false,
    );
    expect(
      hasPassedParity({ status: "passed", cosineSimilarity: 0.999, topTokenMatches: true }),
    ).toBe(true);
  });

  it("rejects malformed events before they reach renderer callbacks", () => {
    const base = { requestId: "request-1", at: new Date(0).toISOString() };
    expect(isWorkerEvent({ ...base, kind: "token", token: "hello" })).toBe(true);
    for (const event of [
      null,
      [],
      { ...base, kind: "secret" },
      { ...base, kind: "token" },
      { ...base, kind: "model", model: { id: 1 } },
      { ...base, kind: "log", message: {} },
      { ...base, kind: "progress", progress: NaN },
      { ...base, kind: "progress", progress: 2 },
      { ...base, kind: "log", at: "bad" },
    ])
      expect(isWorkerEvent(event)).toBe(false);
  });

  it("preserves recorded compute precision without inferring it from parity", () => {
    const model = {
      id: modelId,
      slug: "model",
      state: "ready",
      updatedAt: new Date(0).toISOString(),
      parity: {
        status: "passed",
        cosineSimilarity: 0.9998,
        topTokenMatches: true,
        computePrecision: "fp16-matrices-fp32-sensitive-ops",
      },
    };
    const event = { kind: "model", requestId: "r", at: new Date(0).toISOString(), model };
    expect(isWorkerEvent(event)).toBe(true);
    expect(event.model.parity.computePrecision).toBe("fp16-matrices-fp32-sensitive-ops");
    expect(
      isWorkerEvent({
        ...event,
        model: {
          ...model,
          parity: { ...model.parity, computePrecision: undefined },
        },
      }),
    ).toBe(true);
    expect(
      isWorkerEvent({
        ...event,
        model: {
          ...model,
          parity: { ...model.parity, computePrecision: {} },
        },
      }),
    ).toBe(false);
  });

  it("forwards failed zero-logit parity without inventing unavailable metrics", () => {
    const failedModel = {
      id: modelId,
      slug: "model",
      state: "failed",
      updatedAt: new Date(0).toISOString(),
      parity: { status: "failed", cosineSimilarity: 0, topTokenMatches: false },
      error: "Core ML output did not pass the source-model parity gate.",
    };
    expect(
      isWorkerEvent({
        kind: "model",
        requestId: "r",
        at: new Date(0).toISOString(),
        model: failedModel,
      }),
    ).toBe(true);
    expect(hasPassedParity({ status: "failed", cosineSimilarity: 0, topTokenMatches: false })).toBe(
      false,
    );
  });

  it("does not mislabel requested or planned placement as observed", () => {
    const model = {
      id: modelId,
      slug: "model",
      state: "ready",
      updatedAt: new Date(0).toISOString(),
    };
    const event = { kind: "model", requestId: "r", at: new Date(0).toISOString() };
    const placement = {
      evidence: "planned",
      policy: "all",
      observed: false,
      note: "No trace captured.",
    };
    expect(isWorkerEvent({ ...event, model: { ...model, placement: [placement] } })).toBe(true);
    expect(
      isWorkerEvent({
        ...event,
        model: {
          ...model,
          placement: [{ ...placement, evidence: "observed", observed: false }],
        },
      }),
    ).toBe(true);
    expect(
      isWorkerEvent({
        ...event,
        model: { ...model, placement: [{ ...placement, observed: true }] },
      }),
    ).toBe(false);
  });
});
