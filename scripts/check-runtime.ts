import { createHash } from "node:crypto";
import { lstat, mkdtemp, readdir, readFile, realpath, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { isAbsolute, join, relative, resolve } from "node:path";

import { cleanEnvironment } from "./package-environment";

export { cleanEnvironment } from "./package-environment";

export async function command(args: string[], env: Record<string, string>, input?: string) {
  const child = Bun.spawn(args, {
    env,
    stdout: "pipe",
    stderr: "pipe",
    stdin: input === undefined ? "ignore" : new Blob([input]),
  });
  let timedOut = false;
  const timer = setTimeout(() => {
    timedOut = true;
    child.kill();
  }, 120_000);
  try {
    const [code, output, error] = await Promise.all([
      child.exited,
      new Response(child.stdout).text(),
      new Response(child.stderr).text(),
    ]);
    if (code !== 0 || timedOut)
      throw new Error(`${args[0]} failed${timedOut ? " (timeout)" : ""}: ${error.slice(-3000)}`);
    return output;
  } finally {
    clearTimeout(timer);
  }
}

export async function checkRuntime(runtimePath: string) {
  const runtime = await realpath(runtimePath);
  const native: string[] = [];
  async function walk(directory: string) {
    for (const name of await readdir(directory)) {
      const path = join(directory, name);
      const info = await lstat(path);
      if (info.isSymbolicLink()) {
        const target = relative(runtime, await realpath(path));
        if (target.startsWith("..") || isAbsolute(target))
          throw new Error(`External runtime symlink: ${path}`);
      } else if (info.isDirectory()) await walk(path);
      else if (info.isFile() && info.size >= 8) {
        const bytes = new Uint8Array(await Bun.file(path).slice(0, 8).arrayBuffer());
        const view = new DataView(bytes.buffer);
        if (view.getUint32(0, true) === 0xfeedfacf) {
          if (view.getUint32(4, true) !== 0x0100000c)
            throw new Error(`Non-arm64 runtime binary: ${path}`);
          native.push(path);
        } else if (
          [0xcafebabe, 0xcafebabf, 0xbebafeca, 0xbfbafeca].includes(view.getUint32(0, false))
        ) {
          const check = Bun.spawn(["/usr/bin/lipo", path, "-verify_arch", "arm64"], {
            stdout: "ignore",
            stderr: "ignore",
          });
          if ((await check.exited) !== 0)
            throw new Error(`Universal runtime binary lacks arm64: ${path}`);
          native.push(path);
        } else if (/\.(?:so|dylib)$/.test(path))
          throw new Error(`Unknown runtime binary format: ${path}`);
      }
    }
  }
  await walk(runtime);
  if (native.length === 0) throw new Error("No native runtime binaries found.");
  const home = await mkdtemp(join(tmpdir(), "ane-runtime-check-"));
  const env = cleanEnvironment(home);
  try {
    const libraries = await command(["/usr/bin/otool", "-arch", "arm64", "-L", ...native], env);
    for (const line of libraries.split("\n")) {
      const dependency = /^\s+(\S+) \(compatibility version/.exec(line)?.[1];
      if (
        dependency &&
        !/^(?:@(?:rpath|loader_path|executable_path)\/|\/usr\/lib\/|\/System\/Library\/)/.test(
          dependency,
        )
      ) {
        throw new Error(`Non-portable runtime library: ${dependency}`);
      }
    }
    const loadCommands = await command(["/usr/bin/otool", "-arch", "arm64", "-l", ...native], env);
    for (const match of loadCommands.matchAll(
      /cmd LC_RPATH\s+cmdsize \d+\s+path (.+?) \(offset/g,
    )) {
      if (
        !/^(?:@(?:rpath|loader_path|executable_path)(?:\/|$)|\/usr\/lib\/|\/System\/Library\/)/.test(
          match[1] ?? "",
        )
      ) {
        throw new Error(`Non-portable runtime search path: ${match[1]}`);
      }
    }
    const python = join(runtime, "bin", "python3.12");
    const imports = await command(
      [
        python,
        "-I",
        "-B",
        "-c",
        'import json,sys,torch,transformers,coremltools; print(json.dumps({"python":sys.version.split()[0],"torch":torch.__version__,"transformers":transformers.__version__,"coremltools":coremltools.__version__}))',
      ],
      env,
    );
    const protocol = await command(
      [python, "-I", "-B", "-u", "-m", "ane_studio_worker.cli"],
      env,
      `${JSON.stringify({ id: "health", command: "health", payload: {} })}\n${JSON.stringify({ id: "catalog", command: "catalog", payload: {} })}\n`,
    );
    const messages = protocol
      .trim()
      .split("\n")
      .map((line) => JSON.parse(line));
    if (!messages.find((m) => m.id === "health" && m.ok && m.result.supported))
      throw new Error("Bundled worker health failed.");
    const catalog = messages.find((m) => m.id === "catalog" && m.ok)?.result;
    if (catalog?.models.length !== 0 || catalog.curatedModels.length < 2)
      throw new Error("Clean-data catalog failed.");
    const manifest = JSON.parse(await readFile(join(runtime, "runtime-manifest.json"), "utf8"));
    const currentSource = resolve(import.meta.dir, "../worker/src/ane_studio_worker");
    const bundledWorker = join(runtime, "lib/python3.12/site-packages/ane_studio_worker");
    if (!manifest.workerSources || Object.keys(manifest.workerSources).length === 0)
      throw new Error("Runtime source fingerprint is missing.");
    const expectedFiles = Object.keys(manifest.workerSources).sort();
    for (const directory of [currentSource, bundledWorker]) {
      const files = [];
      for await (const file of new Bun.Glob("**/*.py").scan({ cwd: directory })) files.push(file);
      if (JSON.stringify(files.sort()) !== JSON.stringify(expectedFiles))
        throw new Error("Runtime worker file inventory is stale.");
    }
    const currentLock = createHash("sha256")
      .update(await readFile(resolve(import.meta.dir, "../worker/uv.lock")))
      .digest("hex");
    if (currentLock !== manifest.workerLockSha256)
      throw new Error("Runtime dependency lock is stale.");
    for (const [file, hash] of Object.entries(manifest.workerSources)) {
      for (const directory of [currentSource, bundledWorker]) {
        const actual = createHash("sha256")
          .update(await readFile(join(directory, file)))
          .digest("hex");
        if (actual !== hash) throw new Error(`Runtime worker source is stale: ${file}`);
      }
    }
    console.log(
      JSON.stringify({
        runtime: runtimePath,
        nativeBinaries: native.length,
        imports: JSON.parse(imports),
        health: "passed",
        cleanCatalog: "passed",
        pythonBuild: manifest.standaloneBuild,
      }),
    );
  } finally {
    await rm(home, { recursive: true, force: true });
  }
}

if (import.meta.main) await checkRuntime(resolve(process.argv[2] ?? "artifacts/runtime"));
