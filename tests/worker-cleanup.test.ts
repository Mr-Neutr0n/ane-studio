// @vitest-environment node
import { mkdir, mkdtemp, readdir, rm, symlink, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, expect, it } from "vitest";
import { removeStaleWorkerDirectories } from "../src/main/worker";

const roots: string[] = [];
afterEach(async () => {
  for (const root of roots.splice(0)) await rm(root, { recursive: true, force: true });
});

it("removes real stale worker directories and never follows a symlink", async () => {
  const root = await mkdtemp(join(tmpdir(), "ane-cleanup-"));
  roots.push(root);
  const data = join(root, "data");
  const outside = join(root, "outside");
  await mkdir(join(data, ".worker-tmp-stale", "compiled"), { recursive: true });
  await writeFile(join(data, ".worker-tmp-stale", "compiled", "weights.bin"), "x");
  await mkdir(join(data, "models"), { recursive: true });
  await mkdir(outside);
  await writeFile(join(outside, "keep.txt"), "keep");
  await symlink(outside, join(data, ".worker-tmp-link"));

  await removeStaleWorkerDirectories(data);

  expect((await readdir(data)).sort()).toEqual([".worker-tmp-link", "models"]);
  expect(await readdir(outside)).toEqual(["keep.txt"]);
});
