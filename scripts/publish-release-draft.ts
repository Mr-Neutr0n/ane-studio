import { readFile } from "node:fs/promises";

const pkg = JSON.parse(await readFile("package.json", "utf8"));
const tag = `v${pkg.version}`;
const repository = process.env.GITHUB_REPOSITORY;
const checkOnly = process.argv.includes("--check");
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
// The releases/tags endpoint omits drafts, so list every release, drafts included.
const listed = await run(["gh", "api", "--paginate", "--slurp", `repos/${repository}/releases`]);
if (listed.code !== 0)
  throw new Error("Cannot check the existing release. No release was changed.");
let pages: unknown;
try {
  pages = JSON.parse(listed.output);
} catch {
  throw new Error("Cannot read the release list. No release was changed.");
}
if (!Array.isArray(pages) || !pages.every(Array.isArray))
  throw new Error("Unexpected release list. No release was changed.");
const exists = (pages as unknown[][])
  .flat()
  .some((release) => (release as { tag_name?: unknown } | null)?.tag_name === tag);
if (exists) {
  // Fail so a run that did not upload its new archive never reports success.
  throw new Error(
    `Release ${tag} already exists, possibly as a draft. Its assets were left unchanged. Delete it before rebuilding.`,
  );
} else if (checkOnly) {
  console.log(`No release exists for ${tag}.`);
} else {
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
      throw new Error(`Release ${tag} was created concurrently. Its assets were left unchanged.`);
    } else
      throw new Error(
        "Draft creation failed. Inspect the release before retrying; no existing assets were overwritten.",
      );
  } else console.log(`Created draft ${tag}. A maintainer must review it before publication.`);
}
