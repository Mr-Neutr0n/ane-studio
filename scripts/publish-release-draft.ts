import { readFile } from "node:fs/promises";

const pkg = JSON.parse(await readFile("package.json", "utf8"));
const tag = `v${pkg.version}`;
const repository = process.env.GITHUB_REPOSITORY;
if (
  process.env.GITHUB_REF !== `refs/tags/${tag}` ||
  !repository ||
  !/^[\w.-]+\/[\w.-]+$/.test(repository)
) {
  throw new Error("Draft creation requires the matching version tag in GitHub Actions.");
}
async function run(args: string[]) {
  const child = Bun.spawn(args, { stdout: "pipe", stderr: "pipe" });
  const [code, output, error] = await Promise.all([
    child.exited,
    new Response(child.stdout).text(),
    new Response(child.stderr).text(),
  ]);
  return { code, output, error };
}
const existing = await run(["gh", "api", `repos/${repository}/releases/tags/${tag}`, "--silent"]);
if (existing.code === 0) {
  console.log(`Release ${tag} already exists. Its assets were left unchanged.`);
} else {
  if (!existing.error.includes("HTTP 404"))
    throw new Error("Cannot check the existing release. No release was changed.");
  const result = await run([
    "gh",
    "release",
    "create",
    tag,
    `dist/ANE-Studio-${pkg.version}-arm64.zip`,
    "dist/SHA256SUMS",
    "--repo",
    repository,
    "--verify-tag",
    "--draft",
    "--title",
    `${pkg.build.productName} ${tag}`,
    "--notes-file",
    ".github/RELEASE_NOTES.md",
  ]);
  if (result.code !== 0) {
    // A maintainer can publish between the lookup and create. Never overwrite assets.
    if (result.error.includes("already_exists") || result.error.includes("already exists")) {
      console.log(`Release ${tag} was created concurrently. Its assets were left unchanged.`);
    } else
      throw new Error(
        "Draft creation failed. Inspect the release before retrying; no existing assets were overwritten.",
      );
  } else console.log(`Created draft ${tag}. A maintainer must review it before publication.`);
}
