import { execFile } from "node:child_process";
import { createHash } from "node:crypto";
import { mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { promisify } from "node:util";
import { _electron as electron, expect, test } from "@playwright/test";
import { cleanEnvironment } from "../package-environment";

test("self-contained app starts with clean data and a system-only PATH", async () => {
  const pkg = JSON.parse(await readFile(resolve("package.json"), "utf8"));
  const app = resolve(
    process.env.ANE_PACKAGE_PATH ?? `dist/mac-arm64/${pkg.build.productName}.app`,
  );
  const plist = join(app, "Contents/Info.plist");
  const plistValue = async (key: string) =>
    (
      await promisify(execFile)("/usr/libexec/PlistBuddy", ["-c", `Print :${key}`, plist])
    ).stdout.trim();
  expect(await plistValue("CFBundleIdentifier")).toBe("dev.harikrishna.anestudio");
  expect(await plistValue("CFBundleDisplayName")).toBe(pkg.build.productName);
  const icon = await plistValue("CFBundleIconFile");
  expect(icon).toBe("icon.icns");
  const bundledIcon = await readFile(join(app, "Contents/Resources", icon));
  const sourceIcon = await readFile(resolve(pkg.build.mac.icon));
  expect(createHash("sha256").update(bundledIcon).digest("hex")).toBe(
    createHash("sha256").update(sourceIcon).digest("hex"),
  );
  const home = await mkdtemp(join(tmpdir(), "ane-package-test-"));
  try {
    const application = await electron.launch({
      executablePath: join(app, "Contents", "MacOS", pkg.build.productName),
      env: cleanEnvironment(home),
      timeout: 60_000,
    });
    try {
      const systemIcon = await application.evaluate(
        async ({ app }, path) => (await app.getFileIcon(path, { size: "large" })).toDataURL(),
        app,
      );
      await writeFile(
        "artifacts/packaged-file-icon.png",
        Buffer.from(systemIcon.split(",")[1] ?? "", "base64"),
      );
      const window = await application.firstWindow();
      await expect(window.getByText("LFM2.5 230M", { exact: true }).first()).toBeVisible({
        timeout: 60_000,
      });
      const result = await window.evaluate(async () => {
        const data = await window.studio.bootstrap();
        return {
          supported: data.hardware.supported,
          models: data.models.length,
          hasToken: data.hasToken,
          curated: data.curatedModels.length,
          nodeUnavailable: typeof (globalThis as Record<string, unknown>).require === "undefined",
        };
      });
      expect(result.supported).toBe(true);
      expect(result.models).toBe(0);
      expect(result.hasToken).toBe(false);
      expect(result.curated).toBeGreaterThanOrEqual(2);
      expect(result.nodeUnavailable).toBe(true);
      await window.screenshot({ path: "artifacts/packaged-startup.png", fullPage: true });
      const evidence = {
        app,
        version: pkg.version,
        cleanPathWithoutUv: true,
        cleanData: true,
        packagedStartup: "passed",
        ...result,
        appAsarSha256: createHash("sha256")
          .update(await readFile(join(app, "Contents/Resources/app.asar")))
          .digest("hex"),
        runtimeManifestSha256: createHash("sha256")
          .update(await readFile(join(app, "Contents/Resources/runtime/runtime-manifest.json")))
          .digest("hex"),
      };
      await writeFile("dist/package-verification.json", `${JSON.stringify(evidence, null, 2)}\n`);
      console.log(JSON.stringify(evidence));
    } finally {
      await application.close();
    }
  } finally {
    await rm(home, { recursive: true, force: true });
  }
});
