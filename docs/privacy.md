# Privacy

ANE Studio runs model preparation, chat, and benchmarks on the user's Mac.

## Network access

The worker connects to Hugging Face only to inspect and download a requested model. After the pinned snapshot is complete, conversion and inference use local files. The app has no analytics, telemetry, crash upload, update service, or cloud inference endpoint.

## Credentials

Public models need no token. When a user saves a Hugging Face token, Electron `safeStorage` encrypts it through macOS Keychain. The encrypted value is stored with mode `600`. The main process passes the decrypted token only in the worker's environment.

The renderer cannot read the token. Logs and result files must never contain it.

## Local data

Model weights, converted packages, prompts held by the current UI session, and benchmark results remain under the ANE Studio application data directory. Chat history is not persisted in this version. Removing a model deletes its local source, conversion, manifest, and results.

## Reports

Benchmark JSON records contain model identifiers, prompts, timings, and placement summaries. They do not contain user names, machine serial numbers, Hugging Face tokens, or chat history. Users should review prompts before publishing a report.
