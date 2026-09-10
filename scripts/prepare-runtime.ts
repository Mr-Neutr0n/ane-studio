import { cp, mkdir, readdir, readFile, realpath, writeFile } from "node:fs/promises";
import { createRequire } from "node:module";
import { dirname, join, resolve } from "node:path";
import { relocateRuntime } from "./relocate-runtime";

const root = resolve(import.meta.dir, "..");
if (process.platform !== "darwin" || process.arch !== "arm64") {
  throw new Error("The release runtime must be built on Apple Silicon macOS.");
}
const environment = join(root, "artifacts", "runtime-venv");
const env = { ...process.env, UV_PROJECT_ENVIRONMENT: environment };
async function run(command: string[]) {
  const child = Bun.spawn(command, { cwd: root, env, stdout: "inherit", stderr: "inherit" });
  if ((await child.exited) !== 0) throw new Error(`${command[0]} failed`);
}
await run([
  "uv",
  "sync",
  "--project",
  "worker",
  "--extra",
  "coreml",
  "--locked",
  "--no-editable",
  "--no-dev",
  "--reinstall-package",
  "ane-studio-worker",
  "--managed-python",
  "--python",
  "3.12.13",
]);
await run([join(environment, "bin", "python3.12"), "scripts/bundle-runtime.py"]);
await relocateRuntime(
  join(root, "artifacts", "runtime"),
  dirname(dirname(await realpath(join(environment, "bin", "python3.12")))),
);

const noticeRoot = join(root, "artifacts", "runtime-notices");
await mkdir(join(noticeRoot, "desktop"), { recursive: true });
const seen = new Set<string>();
const inventory: { name: string; version: string; license: string; files: string[] }[] = [];
async function notices(name: string, parent = join(root, "package.json")) {
  const require = createRequire(parent);
  const packagePath = require.resolve(`${name}/package.json`);
  const pkg = JSON.parse(await readFile(packagePath, "utf8"));
  const key = `${pkg.name}@${pkg.version}`;
  if (seen.has(key)) return;
  seen.add(key);
  const output = join(noticeRoot, "desktop", key.replaceAll("/", "__"));
  await mkdir(output, { recursive: true });
  const files: string[] = [];
  for (const file of await readdir(dirname(packagePath))) {
    if (!/^(licen[sc]e|notice|copying|copyright)([.-]|$)/i.test(file)) continue;
    await cp(join(dirname(packagePath), file), join(output, file), { recursive: true });
    files.push(file);
  }
  if (files.length === 0) throw new Error(`Missing bundled desktop license for ${key}`);
  inventory.push({
    name: pkg.name,
    version: pkg.version,
    license: pkg.license ?? "See included notices",
    files,
  });
  for (const dependency of Object.keys(pkg.dependencies ?? {}))
    await notices(dependency, packagePath);
}
const pkg = JSON.parse(await readFile(join(root, "package.json"), "utf8"));
for (const name of Object.keys(pkg.dependencies)) await notices(name);
await cp(
  join(root, "node_modules", "electron", "dist", "LICENSE"),
  join(noticeRoot, "desktop", "ELECTRON-LICENSE.txt"),
);
await cp(
  join(root, "node_modules", "electron", "dist", "LICENSES.chromium.html"),
  join(noticeRoot, "desktop", "LICENSES.chromium.html"),
);
await writeFile(
  join(noticeRoot, "desktop", "inventory.json"),
  `${JSON.stringify(inventory, null, 2)}\n`,
);
console.log("Bundled Python and desktop license notices are ready in artifacts/.");
