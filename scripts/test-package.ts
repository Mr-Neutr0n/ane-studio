import { readFile, rm } from "node:fs/promises";
import { join, resolve } from "node:path";
import { checkRuntime } from "./check-runtime";

const pkg = JSON.parse(await readFile(new URL("../package.json", import.meta.url), "utf8"));
const app = resolve(process.argv[2] ?? `dist/mac-arm64/${pkg.build.productName}.app`);
await rm("dist/package-verification.json", { force: true });
await checkRuntime(join(app, "Contents", "Resources", "runtime"));
const child = Bun.spawn(
  ["bunx", "playwright", "test", "--config", "scripts/packaged-playwright.config.ts"],
  {
    env: { ...process.env, ANE_PACKAGE_PATH: app },
    stdout: "inherit",
    stderr: "inherit",
  },
);
if ((await child.exited) !== 0) throw new Error("Packaged Electron startup test failed.");
