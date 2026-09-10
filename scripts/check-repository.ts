import { readFile, stat } from "node:fs/promises";

const child = Bun.spawn(["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"], {
  stdout: "pipe",
  stderr: "pipe",
});
const [status, output] = await Promise.all([child.exited, new Response(child.stdout).text()]);
if (status !== 0) throw new Error("Cannot list repository files for the credential scan.");
const paths = [...new Set(output.split("\0").filter(Boolean))];
let scanned = 0;
for (const path of paths) {
  const info = await stat(path).catch(() => undefined);
  if (!info?.isFile()) continue;
  if (
    /^(?:artifacts|dist|out|results|node_modules)\/|(?:^|\/)\.venv\/|\.ml(?:package|modelc)(?:\/|$)|\.(?:safetensors|gguf)$/.test(
      path,
    )
  ) {
    throw new Error(`Generated runtime or model file must not be tracked: ${path}`);
  }
  if (/(?:^|\/)\.env(?:\.|$)/.test(path) && !path.endsWith(".env.example")) {
    throw new Error(`Environment files must not be tracked: ${path}`);
  }
  if (info.size > 5_000_000)
    throw new Error(`Review large repository file before publishing: ${path}`);
  const content = await readFile(path);
  if (content.subarray(0, 4096).includes(0)) continue;
  if (/hf_[A-Za-z0-9]{20,}/.test(content.toString("utf8"))) {
    // Never print the matching line or credential itself.
    throw new Error(`Possible Hugging Face credential in repository file: ${path}`);
  }
  scanned += 1;
}
console.log(`Repository scan passed for ${scanned} text files. No credential values were printed.`);
