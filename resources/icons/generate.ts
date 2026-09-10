import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import { copyFile, mkdir, mkdtemp, readFile, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { chromium } from "@playwright/test";

// Run from the repository root: bun resources/icons/generate.ts
// Uses the existing Playwright dependency and macOS iconutil. No network access.
const directory = dirname(fileURLToPath(import.meta.url));
const iconset = join(directory, "icon.iconset");
const sizes = [
  ["icon_16x16.png", 16],
  ["icon_16x16@2x.png", 32],
  ["icon_32x32.png", 32],
  ["icon_32x32@2x.png", 64],
  ["icon_128x128.png", 128],
  ["icon_128x128@2x.png", 256],
  ["icon_256x256.png", 256],
  ["icon_256x256@2x.png", 512],
  ["icon_512x512.png", 512],
  ["icon_512x512@2x.png", 1024],
] as const;

function iconutil(args: string[]) {
  const result = spawnSync("/usr/bin/iconutil", args, { encoding: "utf8" });
  if (result.error) throw result.error;
  assert.equal(result.status, 0, result.stderr || "iconutil failed");
}

async function verifyPng(path: string, size: number) {
  const bytes = await readFile(path);
  assert.equal(bytes.subarray(0, 8).toString("hex"), "89504e470d0a1a0a");
  assert.equal(bytes.readUInt32BE(16), size, `${path}: wrong width`);
  assert.equal(bytes.readUInt32BE(20), size, `${path}: wrong height`);
  assert.equal(bytes[25], 6, `${path}: expected RGBA transparency`);
}

assert.equal(process.platform, "darwin", "ICNS generation needs macOS iconutil.");
await mkdir(iconset, { recursive: true });
const svg = await readFile(join(directory, "icon.svg"), "utf8");
const browser = await chromium.launch({ headless: true });
try {
  const page = await browser.newPage({ deviceScaleFactor: 1 });
  await page.route("**/*", (route) => route.abort());
  for (const [name, size] of sizes) {
    await page.setViewportSize({ width: size, height: size });
    await page.setContent(
      `<html><head><style>html,body{margin:0;background:transparent;}svg{display:block;width:100vw;height:100vh;}</style></head><body>${svg}</body></html>`,
    );
    await page.screenshot({ path: join(iconset, name), omitBackground: true });
    await verifyPng(join(iconset, name), size);
  }
} finally {
  await browser.close();
}
await copyFile(join(iconset, "icon_512x512@2x.png"), join(directory, "icon.png"));
await verifyPng(join(directory, "icon.png"), 1024);
iconutil(["--convert", "icns", "--output", join(directory, "icon.icns"), iconset]);

const temporary = await mkdtemp(join(tmpdir(), "ane-icon-verify-"));
try {
  const roundtrip = join(temporary, "roundtrip.iconset");
  iconutil(["--convert", "iconset", "--output", roundtrip, join(directory, "icon.icns")]);
  for (const [name, size] of sizes) {
    await verifyPng(join(roundtrip, name), size);
    console.log(`${name}: ${size} x ${size}, RGBA, ICNS round trip passed`);
  }
} finally {
  await rm(temporary, { recursive: true, force: true });
}
console.log("Created editable SVG, 1024px PNG, and verified 10-resolution icon.icns.");
