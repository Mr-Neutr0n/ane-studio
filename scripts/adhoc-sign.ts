import { readFile } from "node:fs/promises";
import { resolve } from "node:path";

// electron-builder edits Electron's Info.plist files after Electron's own ad-hoc
// signatures, which breaks the bundle seal. Gatekeeper then reports a downloaded
// copy as damaged instead of offering Open Anyway. Re-seal the bundle ad hoc here
// rather than through electron-builder, whose "-" identity is also a Keychain
// substring match that can pick a real certificate on a developer Mac.
const pkg = JSON.parse(await readFile(new URL("../package.json", import.meta.url), "utf8"));
const app = resolve(process.argv[2] ?? `dist/mac-arm64/${pkg.build.productName}.app`);
// --deep signs nested frameworks and helper apps before the outer bundle.
const child = Bun.spawn(
  ["/usr/bin/codesign", "--force", "--deep", "--sign", "-", "--timestamp=none", app],
  { stdout: "inherit", stderr: "inherit" },
);
if ((await child.exited) !== 0) throw new Error("Ad-hoc signing of the packaged app failed.");
console.log(`Ad-hoc signed ${app}. This is not Developer ID signing or notarization.`);
