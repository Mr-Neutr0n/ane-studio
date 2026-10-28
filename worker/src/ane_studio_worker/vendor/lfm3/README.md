# Pinned LFM3 implementation

The three Python implementation files come from Liquid AI's
[LFM3-1.7B checkpoint](https://huggingface.co/LiquidAI/LFM3-1.7B/tree/40ebe0aa6662c9d40389c2f5d565b6a2c6aebacd)
at revision `40ebe0aa6662c9d40389c2f5d565b6a2c6aebacd`. `provenance.json` records
the original file hashes. The source headers license these files under Apache
2.0. The separately included LFM Open License v1.0 applies to model weights.

The model and convolution implementations are kept intact. The configuration
has two compatibility overrides for locked Transformers 5.16: retain the
`conv` layer label and preserve per-attention-type RoPE dictionaries. Its
upstream architecture validation still runs. The generic Transformers
validators otherwise rewrite `conv` and add global RoPE keys, breaking the
official forward path. `register()` explicitly registers these local classes
with Transformers. Callers must still use
`trust_remote_code=False`, `local_files_only=True`, and eager attention. The
downloaded checkpoint's Python files are neither required nor executed.

The code defines model layers and optional CUDA convolution kernels. It has no
shell, subprocess, credential, or filesystem operations. Transformers' optional
Hub kernel decorators are inert without the optional kernels package. Do not
enable Hub kernels. On this app's macOS CPU tracing path, width-7 convolution
uses the ordinary PyTorch convolution fallback; the optional Triton functions
are not used.

Local registration adds source loading only. A checkpoint must independently
pass the app's conversion and parity gates before chat or benchmarking.

The nested Ruff configuration preserves upstream formatting and exempts only
its long lines, import order, and existing convolution `zip()` convention.
All other project lint rules still apply.
