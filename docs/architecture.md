# Architecture

## Process boundaries

ANE Studio uses three local layers:

```text
Electron renderer
      |
      | typed IPC through contextBridge
      v
Electron main process
      |
      | JSONL over stdin and stdout
      v
Python model worker
      |
      +-- Hugging Face Hub and Transformers
      +-- PyTorch tracing
      +-- Core ML conversion and prediction
```

The renderer has context isolation and sandboxing enabled. It cannot access Node.js, the filesystem, child processes, environment variables, or credentials.

The main process owns window lifecycle, credential encryption, worker supervision, cancellation, and IPC validation. It accepts only the worker commands listed in `src/main/index.ts`.

The Python worker owns model metadata, downloads, conversion, parity, chat, planned placement, and benchmark files. It emits compact JSON objects, one per line. Standard error is treated as a technical log and never parsed as protocol data.

## Artifact flow

```text
Hugging Face model ID
        |
        v
metadata inspection -> supported architecture gate
        |
        v
pinned local snapshot at an exact revision
        |
        v
fixed-shape PyTorch trace -> FP16 Core ML .mlpackage
        |                              |
        +---------- parity ------------+
                                       |
                           chat and policy tests
```

A model record enters `ready` only when logit cosine similarity is at least `0.999` and the source and Core ML top token match. Failed artifacts remain local for diagnosis but cannot run.

## Runtime shape

The first runtime uses one fixed 64-token graph. It right-pads prompts and passes the current final-token position into the model. Each generation step runs the complete fixed graph and returns one vocabulary logit vector. This makes the first implementation portable across the two adapters, but it is not a stateful decoder.

A later stateful runtime should split prefill and decode, use Core ML state for KV or convolution caches where supported, and preserve the current parity and placement contracts.

## Placement evidence

The app records three levels:

1. Requested: the selected `MLComputeUnits` policy allowed a device.
2. Planned: `MLComputePlan` anticipated operation placement.
3. Observed: Instruments activity overlapped a signed inference interval.

The current app automates the first two. It records observed placement as not captured. Timing runs and profiling runs must remain separate because Instruments changes latency.

## Local storage

The main process sets the application data directory to `~/Library/Application Support/ANE Studio` unless `ANE_STUDIO_HOME` is set for tests. Each model gets a content-stable slug and contains:

```text
models/<slug>/
  source/                  pinned Hugging Face snapshot
  artifacts/model.mlpackage/
  results/benchmark-*.json
  manifest.json
```

The manifest records the source revision, architecture, parameter count, license identifier, sequence length, parity, artifact bytes, and artifact digest.

## Cancellation

Conversion runs synchronously inside the worker. Cancel stops the worker process and starts a clean one. Hugging Face and Core ML files already flushed to disk remain inspectable. A later preparation attempt overwrites the canonical artifact only after conversion reaches the save stage.
