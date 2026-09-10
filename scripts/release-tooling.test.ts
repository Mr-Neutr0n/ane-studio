import { afterEach, describe, expect, test } from "bun:test";
import { chmod, mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";

const directories: string[] = [];
afterEach(async () => {
  for (const path of directories.splice(0)) await rm(path, { recursive: true, force: true });
});
async function run(mode: string, ref?: string) {
  const directory = await mkdtemp(join(tmpdir(), "ane-release-cli-"));
  directories.push(directory);
  const executable = join(directory, "gh");
  const log = join(directory, "calls.jsonl");
  const version = JSON.parse(await readFile("package.json", "utf8")).version;
  await writeFile(
    executable,
    `#!${process.execPath}
import {appendFileSync} from "node:fs";
const args=process.argv.slice(2);
appendFileSync(process.env.CALL_LOG,JSON.stringify(args)+"\\n");
if(args[0]==="api"){if(process.env.MODE==="existing")process.exit(0);console.error(process.env.MODE==="network"?"HTTP 503":"HTTP 404");process.exit(1);}
if(process.env.MODE==="concurrent"){console.error("already_exists");process.exit(1);}
console.log("draft created");
`,
  );
  await chmod(executable, 0o755);
  const child = Bun.spawn([process.execPath, resolve("scripts/publish-release-draft.ts")], {
    env: {
      ...process.env,
      PATH: `${directory}:${process.env.PATH}`,
      MODE: mode,
      CALL_LOG: log,
      GITHUB_REPOSITORY: "fixture/ane",
      GITHUB_REF: ref ?? `refs/tags/v${version}`,
    },
    stdout: "pipe",
    stderr: "pipe",
  });
  const [code, output, error] = await Promise.all([
    child.exited,
    new Response(child.stdout).text(),
    new Response(child.stderr).text(),
  ]);
  const calls = (await readFile(log, "utf8").catch(() => ""))
    .trim()
    .split("\n")
    .filter(Boolean)
    .map((line) => JSON.parse(line));
  return { code, output, error, calls };
}

describe("release draft safety", () => {
  test("leaves an existing release and assets unchanged", async () => {
    const result = await run("existing");
    expect(result.code).toBe(0);
    expect(result.output).toContain("left unchanged");
    expect(result.calls).toHaveLength(1);
  });
  test("creates only a draft when the tag has no release", async () => {
    const result = await run("missing");
    expect(result.code).toBe(0);
    expect(result.calls).toHaveLength(2);
    expect(result.calls[1]).toContain("--draft");
    expect(result.calls[1]).toContain("--verify-tag");
    expect(result.calls[1]).not.toContain("--clobber");
  });
  test("does not mutate anything after an API error", async () => {
    const result = await run("network");
    expect(result.code).not.toBe(0);
    expect(result.calls).toHaveLength(1);
  });
  test("leaves concurrently created releases unchanged", async () => {
    const result = await run("concurrent");
    expect(result.code).toBe(0);
    expect(result.output).toContain("created concurrently");
    expect(result.calls).toHaveLength(2);
  });
  test("requires the package version tag before invoking GitHub", async () => {
    const result = await run("missing", "refs/heads/main");
    expect(result.code).not.toBe(0);
    expect(result.calls).toHaveLength(0);
  });
});
