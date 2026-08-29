import { describe, expect, it } from "vitest";
import type { ComputePolicy, ModelRecord } from "../src/shared/contracts";

const policies: ComputePolicy[] = ["cpuOnly", "cpuAndGPU", "cpuAndNeuralEngine", "all"];

describe("shared contracts", () => {
  it("keeps the four Core ML policies explicit", () => {
    expect(policies).toHaveLength(4);
    expect(new Set(policies).size).toBe(4);
  });

  it("requires passed parity for a ready fixture", () => {
    const record: ModelRecord = {
      id: "org/model",
      slug: "org-model",
      state: "ready",
      parity: { status: "passed", topTokenMatches: true },
      updatedAt: new Date(0).toISOString(),
    };
    expect(record.state === "ready" && record.parity?.status === "passed").toBe(true);
  });
});
