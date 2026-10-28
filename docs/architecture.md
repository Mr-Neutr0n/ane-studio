# Architecture

## Process boundaries

The desktop app uses three local layers:

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

The renderer has context isolation and sandboxing enabled. It cannot access Node.js, the filesystem, child processes, environment variables, or saved credentials. The packaged renderer can load only its local assets. Permissions, embedded webviews, and navigation to other documents are blocked.

The main process owns window lifecycle, credential encryption, worker supervision, cancellation, and IPC validation. Every IPC handler checks the sender, main frame, and loaded document URL. The shared contract validates command names, payload fields, model IDs, policies, and request sizes. External links need an allowed HTTPS host and native user confirmation.

The Python worker owns model metadata, downloads, conversion, parity, chat, planned placement, and benchmark files. It emits compact JSON objects, one per line. Standard error is treated as a technical log and never parsed as protocol data.

The desktop app uses a single-instance lock for its data directory. Standalone CLI workers do not share that lock. Give CLI smoke tests a separate `ANE_STUDIO_HOME`; do not run two workers against the same model directory.

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
fixed-shape PyTorch trace -> Core ML .mlpackage
        |                              |
        +---------- parity ------------+
                                       |
                           chat and policy tests
```

A new conversion checks four prompts against the original, unpatched Transformers forward. They include one token at position zero and a prompt that reaches the final context position. Every check needs finite, nonzero logits, cosine similarity of at least `0.999`, and matching top tokens. Trace rewrites are scoped and restored even after failure.

LFM2 uses FP16 weights and conversion. Llama uses an FP32 source reference with selective FP16 conversion for matrix operations. Its other operations retain FP32. Qwen2 uses an FP32 reference and conversion because selective FP16 failed the one-token check. These choices preserve the parity gate rather than relaxing it for smaller artifacts. The parity record stores the precision mode, reference dtype, converter version, and individual checks. Failed artifacts remain local for diagnosis but cannot run.

LFM3 also requires an FP32 source and full FP32 conversion for the tested 1.7B
checkpoint. Its bundled local adapter preserves the tri-hybrid architecture;
downloaded model code remains disabled. Optional thinking is disabled through
its chat template for the short graph. See [model support](model-support.md) for
the pinned implementation, rejected precision variants, and numerical evidence.

## Runtime shape

The default runtime uses one fixed 64-token graph. It right-pads contiguous token IDs and passes the current final-token position into the model. The trace mask is valid only for that input contract, not left padding or gaps in an attention mask. Each generation step runs the complete fixed graph, selects the last hidden state, and projects it into one vocabulary logit vector.

Generation feeds token IDs directly. It does not decode and re-encode the prompt at every step. The chat template, history, prompt, and reply share the fixed context. A prompt that fills the context is rejected with a request to shorten it or start a new chat. This is not a stateful decoder or a sliding-window chat runtime.

A later stateful runtime should split prefill and decode, use Core ML state for KV or convolution caches where supported, and preserve the current parity and placement contracts.

## Placement evidence

The app records three levels:

1. Requested: the selected `MLComputeUnits` policy allowed a device.
2. Planned: `MLComputePlan` anticipated operation placement.
3. Observed: Instruments activity overlapped a signed inference interval.

The desktop UI automates the first two. It records observed placement as not
captured for its timing runs. A separate developer tool captures aligned
Instruments evidence. Timing and profiling remain separate because Instruments
changes latency. The worker explicitly moves completed native compilations into
Python-managed temporary directories and retains them for the loaded model's
lifetime. Electron gives each worker a private temporary directory and removes
it after termination. Initial native compilation still uses Foundation's own
temporary directory until the completed model can be moved.

The pinned Core ML Python bridge can release borrowed NumPy inputs on a native
queue without the GIL. The worker retains immutable input arrays until their
native owners release them; it never changes a still-borrowed input in place.
This mitigates the known binding issue while keeping the same model math.

## Local storage

The main process sets the application data directory to `~/Library/Application Support/ANE Studio` unless `ANE_STUDIO_HOME` is set for tests. Each model gets a content-stable slug and contains:

```text
models/<slug>/
  source/                  pinned Hugging Face snapshot
  artifacts/model.mlpackage/   portable converted package
  results/benchmark-*.json
  manifest.json
```

The manifest records the source revision, architecture, parameter count, license identifier, sequence length, parity, artifact bytes, and artifact digest. Hugging Face metadata and source downloads use the same pinned revision. Managed paths reject symlinks. Manifest writes use an atomic replacement.

Chat, benchmarks, and placement checks validate the manifest and artifact digest before use. Digest results are cached against a filesystem signature, not just the model ID. The worker keeps at most one loaded model and policy runtime. Preparing a model again invalidates that runtime. Hugging Face metadata has a separate cache under the app data directory.

## Cancellation

Conversion runs synchronously inside the worker. Cancel stops the worker process group and rejects pending requests. A clean worker starts on the next request. Saving or deleting a token also stops the existing worker before the next request uses the new credentials. Startup and requests have time limits. Late responses from a stopped worker cannot update a replacement session.

Hugging Face and Core ML files already flushed to disk remain inspectable. An interrupted record does not pass parity and can be prepared again. A later preparation attempt overwrites the canonical artifact only after conversion reaches the save stage. Closing the renderer stops its worker. App shutdown waits for worker termination.

## Packaged runtime

Source development starts the locked worker environment through `uv`. Release packaging builds a separate, relocatable CPython runtime with the locked worker dependencies and license notices. The packaged app launches that Python directly in isolated mode. It does not depend on Bun, uv, Homebrew, or a system Python installation. Model weights are not bundled.
