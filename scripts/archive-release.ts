import { createHash } from "node:crypto";
import { createReadStream } from "node:fs";
import { readFile, stat, writeFile } from "node:fs/promises";
import { resolve } from "node:path";
import { checkRuntime } from "./check-runtime";

const pkg = JSON.parse(await readFile(new URL("../package.json", import.meta.url), "utf8"));
const app = resolve(`dist/mac-arm64/${pkg.build.productName}.app`);
await checkRuntime(resolve(app, "Contents/Resources/runtime"));
const verification = JSON.parse(await readFile(resolve("dist/package-verification.json"), "utf8"));
if (
  verification.app !== app ||
  verification.version !== pkg.version ||
  verification.packagedStartup !== "passed"
)
  throw new Error("Run bun run package:test on this app before archiving.");
for (const [file, expected] of [
  ["app.asar", verification.appAsarSha256],
  ["runtime/runtime-manifest.json", verification.runtimeManifestSha256],
]) {
  const actual = createHash("sha256")
    .update(await readFile(resolve(app, "Contents/Resources", file)))
    .digest("hex");
  if (actual !== expected)
    throw new Error("App changed after packaged verification. Repeat bun run package:test.");
}
const filename = `ANE-Studio-${pkg.version}-arm64.zip`;
const path = resolve("dist", filename);
const child = Bun.spawn(
  ["/usr/bin/ditto", "-c", "-k", "--sequesterRsrc", "--keepParent", app, path],
  { stdout: "inherit", stderr: "inherit" },
);
if ((await child.exited) !== 0) throw new Error("Release archive failed.");
const hash = createHash("sha256");
for await (const chunk of createReadStream(path)) hash.update(chunk);
const digest = hash.digest("hex");
await writeFile(resolve("dist", "SHA256SUMS"), `${digest}  ${filename}\n`);
console.log(
  JSON.stringify({
    path,
    bytes: (await stat(path)).size,
    sha256: digest,
    signed: false,
    notarized: false,
  }),
);
