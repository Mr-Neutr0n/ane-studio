# Third-party notices

ANE Studio uses open-source software. Its MIT source license does not replace the licenses of bundled dependencies or downloaded models.

## What a release includes

The app includes Electron, the frontend dependencies, a python-build-standalone CPython runtime, and the locked Python worker dependencies. It does not include model weights, Hugging Face tokens, or a uv executable.

Each release carries these files under `ANE Studio.app/Contents/Resources/licenses/`:

- This notice and the app's MIT license.
- `runtime/desktop/inventory.json` and license texts for bundled JavaScript dependencies.
- Electron's license and `LICENSES.chromium.html` for Chromium, Node.js, and their dependencies.
- `runtime/python-packages/inventory.json` and license or notice files from every bundled Python distribution.
- `runtime/python-standalone/` with the CPython license and standalone runtime dependency notices.

The runtime itself retains the distributions' license files and metadata. `Contents/Resources/runtime/runtime-manifest.json` records Python and package versions, the worker lock digest, and the standalone license source revision. The build fails if a Python or desktop package has no collected license text.

## Desktop dependencies

| Project | License |
| --- | --- |
| Electron | MIT, with separately listed Chromium and Node.js notices |
| React and React DOM | MIT |
| Lucide | ISC |
| Electron Toolkit | MIT |
| Vite and electron-vite, build tools | MIT |

## Python and model worker

| Project | License |
| --- | --- |
| CPython | Python Software Foundation license and included notices |
| python-build-standalone | Included upstream notices for Python and its native libraries |
| Core ML Tools | BSD 3-Clause |
| PyTorch | BSD 3-Clause and bundled third-party notices |
| Transformers, Hugging Face Hub, Accelerate | Apache 2.0 |
| Tokenizers, safetensors, hf-xet | Apache 2.0 |
| NumPy | BSD 3-Clause and bundled library notices |
| certifi | MPL 2.0 |
| tqdm | MPL 2.0 and MIT |
| regex | Apache 2.0 and CNRI-Python |

The inventory, not this short table, lists every bundled distribution. Exact application dependencies are locked in `bun.lock` and `worker/uv.lock`.

The worker also bundles Liquid AI's pinned LFM3 implementation under
`worker/src/ane_studio_worker/vendor/lfm3/`. Its Python source is Apache 2.0
licensed. Two configuration compatibility changes, original source hashes, and
the checkpoint revision are recorded in that directory's `provenance.json`.
The source license text and model-weight license are included there. Downloaded
checkpoint Python code is not executed.

The release runtime uses CPython `3.12.13`, standalone build `20260510`. Standalone dependency notices come from [python-build-standalone revision f0ae464](https://github.com/astral-sh/python-build-standalone/tree/f0ae464f527498d6c4b13f19977447d603cd5edd). The Tokenizers `0.23.1` wheel omits its license file, so the build collects the Apache license from [its exact upstream revision](https://github.com/huggingface/tokenizers/tree/7f1623b90b5adfb9bc327d4c3468d2f70bbce262).

The app does not modify certifi or tqdm source. Their corresponding source is available at [certifi](https://github.com/certifi/python-certifi) and [tqdm](https://github.com/tqdm/tqdm), at the versions in the runtime inventory. The build changes local Mach-O library references so the Python runtime can move with the app. Local ad-hoc signatures on those files are not Apple Developer ID signing or notarization.

Model licenses are listed separately in [MODEL_LICENSES.md](MODEL_LICENSES.md). A checkpoint's own terms apply even when the app accepts its architecture.
