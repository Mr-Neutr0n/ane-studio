import { lstat, realpath } from "node:fs/promises";
import { basename, dirname, join, relative, resolve } from "node:path";

export async function relocateRuntime(runtimePath: string, sourcePath: string) {
  const runtime = await realpath(runtimePath);
  const source = await realpath(sourcePath);
  const native: string[] = [];
  for await (const path of new Bun.Glob("**/*").scan({
    cwd: runtime,
    absolute: true,
    onlyFiles: true,
  })) {
    const stat = await lstat(path);
    if (!stat.isFile() || stat.size < 4) continue;
    const view = new DataView(await Bun.file(path).slice(0, 4).arrayBuffer());
    if (
      view.getUint32(0, true) === 0xfeedfacf ||
      [0xcafebabe, 0xcafebabf, 0xbebafeca, 0xbfbafeca].includes(view.getUint32(0, false))
    )
      native.push(path);
  }
  async function output(args: string[]) {
    const child = Bun.spawn(args, { stdout: "pipe", stderr: "pipe" });
    const [code, stdout, stderr] = await Promise.all([
      child.exited,
      new Response(child.stdout).text(),
      new Response(child.stderr).text(),
    ]);
    if (code !== 0) throw new Error(`${args[0]} failed: ${stderr}`);
    return stdout;
  }
  const changes = new Map<string, string[]>();
  const identities = new Map<string, string>();
  let identityBinary = "";
  for (const line of (await output(["/usr/bin/otool", "-arch", "arm64", "-D", ...native])).split(
    "\n",
  )) {
    if (line.endsWith(":")) {
      identityBinary = line.slice(0, -1).replace(/ \(architecture [^)]+\)$/, "");
      continue;
    }
    const identity = line.trim();
    if (!identity || !identityBinary) continue;
    identities.set(identityBinary, identity);
    if (!identity.startsWith("@"))
      changes.set(identityBinary, ["-id", `@rpath/${basename(identityBinary)}`]);
  }
  function replacement(binary: string, old: string) {
    if (!old.startsWith(`${source}/`)) return undefined;
    const target = join(runtime, relative(source, old));
    return `@loader_path/${relative(dirname(binary), target) || "."}`;
  }
  const libraries = await output(["/usr/bin/otool", "-arch", "arm64", "-L", ...native]);
  let binary = "";
  for (const line of libraries.split("\n")) {
    if (line.endsWith(":")) {
      binary = line.slice(0, -1).replace(/ \(architecture [^)]+\)$/, "");
      continue;
    }
    const old = /^\s+(\S+) \(compatibility version/.exec(line)?.[1];
    if (!old || identities.get(binary) === old) continue;
    const portable = replacement(binary, old);
    if (!portable) continue;
    const args = changes.get(binary) ?? [];
    if (join(runtime, relative(source, old)) === binary)
      args.push("-id", `@rpath/${relative(runtime, binary)}`);
    else args.push("-change", old, portable);
    changes.set(binary, args);
  }
  const loads = await output(["/usr/bin/otool", "-arch", "arm64", "-l", ...native]);
  let rpath = false;
  for (const line of loads.split("\n")) {
    if (line.endsWith(":")) {
      binary = line.slice(0, -1).replace(/ \(architecture [^)]+\)$/, "");
      continue;
    }
    if (line.trim() === "cmd LC_RPATH") rpath = true;
    if (!rpath) continue;
    const old = /^\s+path (.+) \(offset/.exec(line)?.[1];
    if (!old) continue;
    rpath = false;
    const portable = replacement(binary, old);
    const args = changes.get(binary) ?? [];
    if (portable) args.push("-rpath", old, portable);
    else if (
      old.startsWith("/") &&
      !old.startsWith("/usr/lib/") &&
      !old.startsWith("/System/Library/")
    ) {
      // Wheels can retain unreachable build-machine search paths. Do not ship them.
      args.push("-delete_rpath", old);
    } else continue;
    changes.set(binary, args);
  }
  for (const [binary, args] of changes) {
    await output(["/usr/bin/install_name_tool", ...args, binary]);
    // Required for modified arm64 Mach-O files. This is local ad-hoc signing,
    // not an Apple Developer ID signature and not notarization.
    await output(["/usr/bin/codesign", "--force", "--sign", "-", "--timestamp=none", binary]);
  }
  console.log(
    `Relocated ${changes.size} native runtime files. No Developer ID signing was performed.`,
  );
}

if (import.meta.main) {
  const source = process.argv[3];
  if (!source)
    throw new Error("Usage: bun scripts/relocate-runtime.ts RUNTIME SOURCE_PYTHON_PREFIX");
  await relocateRuntime(resolve(process.argv[2] ?? "artifacts/runtime"), source);
}
