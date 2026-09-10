# Privacy

The desktop app runs model preparation, chat, and benchmarks on the user's Mac.

## Network access

The worker connects to Hugging Face only to inspect and download a requested model. After the pinned snapshot is complete, conversion and inference use local files. The app has no analytics, telemetry, crash upload, update service, or cloud inference endpoint.

## Credentials

Public models need no token. When a user saves a Hugging Face token, Electron `safeStorage` encrypts it through macOS Keychain. The encrypted value is stored with mode `600`. The main process passes the decrypted token only in the worker's environment.

The renderer accepts a token when the user enters it but cannot retrieve the saved token. Closing the token dialog clears its input. The main process isolates the worker from other tools' Hugging Face credentials and disables implicit token lookup. Logs and error messages redact tokens.

## Local data

Model weights, converted packages, manifests, and benchmark results stay under `~/Library/Application Support/ANE Studio`, or the explicit `ANE_STUDIO_HOME` test override. Chat prompts and history stay in memory and are not persisted in this version.

Removing a model deletes its local source, conversion, manifest, and results after confirmation. Downloaded metadata can remain in the app's Hugging Face cache. Removing the whole application data directory also removes that cache and saved credentials.

## Reports

Benchmark JSON records contain model identifiers, prompts, timings, and placement summaries. They do not contain user names, machine serial numbers, Hugging Face tokens, or chat history. Users should review prompts before publishing a report.
