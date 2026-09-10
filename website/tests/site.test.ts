import { describe, expect, test } from "bun:test";
import { readdir, readFile, stat } from "node:fs/promises";
import { resolve } from "node:path";
import { renderHTML, repositoryURL, validateRelease, websiteRoot } from "../scripts/build";
import { productionSecurityHeaders } from "../scripts/headers";
import { resolveSiteURL } from "../scripts/site";

const output = resolve(websiteRoot, "dist");
const html = await readFile(resolve(output, "index.html"), "utf8");
const template = await readFile(resolve(websiteRoot, "src/index.html"), "utf8");
const site = JSON.parse(await readFile(resolve(websiteRoot, "site.json"), "utf8"));
const siteURL = resolveSiteURL(
  site.url,
  process.env.VERCEL_ENV === "production" ? process.env.VERCEL_PROJECT_PRODUCTION_URL : undefined,
);
const release = {
  version: "0.1.0",
  assetUrl: `${repositoryURL}/releases/download/v0.1.0/ANE-Studio-0.1.0-arm64.zip`,
  releaseUrl: `${repositoryURL}/releases/tag/v0.1.0`,
  checksumsUrl: `${repositoryURL}/releases/download/v0.1.0/SHA256SUMS`,
  publishedAt: "2026-01-01T00:00:00Z",
};

describe("static build", () => {
  test("ships a complete, indexed ANE Studio document", () => {
    expect(html).toContain('<html lang="en">');
    expect(html).toContain("ANE Studio | A local model lab for Apple Silicon");
    if (siteURL) expect(html).toContain(`<link rel="canonical" href="${siteURL}"`);
    else expect(html).toContain('name="robots" content="noindex, nofollow"');
    expect(html).not.toContain("harikip.com");
    expect(html).toContain('property="og:image"');
    expect(html).toContain('name="twitter:card"');
    expect(html).not.toMatch(/\{\{[A-Z_]+\}\}/);
    expect(html.match(/<h1[ >]/g)).toHaveLength(1);
  });

  test("all local assets and anchor destinations exist", async () => {
    const ids = [...html.matchAll(/\bid="([^"]+)"/g)].map((match) => match[1]);
    expect(new Set(ids).size).toBe(ids.length);
    const resources = [...html.matchAll(/\b(?:src|href)="([^"]+)"/g)].map((match) => match[1]);
    for (const resource of resources) {
      if (resource.startsWith("#")) {
        expect(ids).toContain(resource.slice(1));
      } else if (resource.startsWith("/")) {
        expect(
          (await stat(resolve(output, `.${resource === "/" ? "/index.html" : resource}`))).isFile(),
        ).toBe(true);
      } else {
        expect(resource).toStartWith("https://");
      }
    }
  });

  test("has no trackers, network runtime, forms, or remote assets", async () => {
    // Canonical is a document URL, not a fetched asset.
    const withoutCanonical = html.replace(/<link rel="canonical"[^>]+>/g, "");
    expect(withoutCanonical).not.toMatch(/<(?:script|img|link)[^>]+(?:src|href)="https?:/i);
    expect(html).not.toMatch(/<form\b|<iframe\b|\bonclick=/i);
    const script = await readFile(resolve(output, "main.js"), "utf8");
    expect(script).not.toMatch(
      /\bfetch\s*\(|XMLHttpRequest|localStorage|document\.cookie|sendBeacon|WebSocket/,
    );
    expect(html).not.toContain("—");
  });

  test("keeps claims and runtime limits explicit", () => {
    for (const fact of [
      "cosine similarity of at least 0.999",
      "same top token",
      "fixed 64-token",
      "No KV cache yet",
      "Not captured",
      "unsigned",
      "not notarized",
      "macOS 15",
      "not a hardware ranking",
      "0.999936",
      "0.999706",
    ]) {
      expect(html).toContain(fact);
    }
    expect(html).toContain("~/Library/Application Support/ANE Studio/");
    expect(html).not.toContain("ANE-only");
    expect(html).not.toContain("100% private");
  });

  test("keeps download fallback honest and source commands usable", async () => {
    const sourceHTML = renderHTML(template, null);
    expect(sourceHTML).toContain('href="#install">Run from source');
    expect(sourceHTML).toContain("There is no public app download linked yet");
    expect(sourceHTML).not.toContain("/releases/download/");
    expect(sourceHTML).not.toContain("Download for macOS");
    for (const command of [
      "git clone https://github.com/Mr-Neutr0n/ane-studio.git",
      "cd ane-studio",
      "bun install",
      "bun run worker:setup",
      "bun run dev",
    ]) {
      expect(sourceHTML).toContain(command);
    }
    const config = JSON.parse(await readFile(resolve(websiteRoot, "release.json"), "utf8"));
    expect(validateRelease(config)).toEqual(config);
  });

  test("stays small and publishes crawler assets", async () => {
    const files = await readdir(output);
    let bytes = 0;
    for (const file of files) bytes += (await stat(resolve(output, file))).size;
    expect(bytes).toBeLessThan(700_000);
    const robots = await readFile(resolve(output, "robots.txt"), "utf8");
    if (siteURL) {
      expect(robots).toContain(`Sitemap: ${siteURL}sitemap.xml`);
      expect(await readFile(resolve(output, "sitemap.xml"), "utf8")).toContain(
        `<loc>${siteURL}</loc>`,
      );
    } else {
      expect(robots).toContain("Disallow: /");
      expect(files).not.toContain("sitemap.xml");
    }
  });

  test("production and preview enforce the same security headers", async () => {
    const vercel = JSON.parse(await readFile(resolve(websiteRoot, "../vercel.json"), "utf8"));
    expect(vercel.outputDirectory).toBe("website/dist");
    const headers = Object.fromEntries(
      vercel.headers[0].headers.map((header: { key: string; value: string }) => [
        header.key,
        header.value,
      ]),
    );
    expect(headers).toEqual(productionSecurityHeaders);
    expect(headers["Content-Security-Policy"]).toContain("connect-src 'none'");
    expect(headers["Content-Security-Policy"]).toContain("frame-ancestors 'none'");
  });
});

describe("published release gate", () => {
  test("renders only versioned project assets with checksums and signing limits", () => {
    expect(validateRelease(release)).toEqual(release);
    const releasedHTML = renderHTML(template, release);
    expect(releasedHTML).toContain(`href="${release.assetUrl}"`);
    expect(releasedHTML).toContain("Download for macOS");
    expect(releasedHTML).toContain(`href="${release.checksumsUrl}"`);
    expect(releasedHTML).toContain("Unsigned and not notarized");
    expect(releasedHTML).not.toContain("There is no public app download linked yet");
    expect(releasedHTML).not.toMatch(/\{\{[A-Z_]+\}\}/);
  });

  test("rejects untrusted URLs, mismatched tags, missing fields, and future promises", () => {
    const cases = [
      undefined,
      [],
      {},
      "https://example.com",
      { ...release, assetUrl: "javascript:alert(1)" },
      { ...release, assetUrl: "https://example.com/app.zip" },
      { ...release, version: '0.1.0" onclick="alert(1)' },
      { ...release, version: "0.2.0" },
      { ...release, releaseUrl: `${repositoryURL}/releases/latest` },
      { ...release, checksumsUrl: "https://example.com/SHA256SUMS" },
      { ...release, publishedAt: "not a date" },
      { ...release, publishedAt: "2999-01-01T00:00:00Z" },
    ];
    for (const invalid of cases) expect(() => validateRelease(invalid)).toThrow();
  });

  test("rejects unknown template placeholders", () => {
    expect(() => renderHTML("{{TYPO_FIELD}}", null)).toThrow("Unknown template field");
  });
});

describe("canonical configuration", () => {
  test("has no unconfirmed canonical and allows a controlled preview or confirmed domain", () => {
    expect(resolveSiteURL(null)).toBeNull();
    expect(resolveSiteURL(null, "ane-studio-site-example.vercel.app")).toBe(
      "https://ane-studio-site-example.vercel.app/",
    );
    expect(resolveSiteURL("https://confirmed.example/")).toBe("https://confirmed.example/");
    expect(() => resolveSiteURL(null, "untrusted.example")).toThrow();
    for (const invalid of [
      "http://example.com",
      "https://user:pass@example.com",
      "https://example.com/path",
      "https://example.com/?query=1",
      123,
    ]) {
      expect(() => resolveSiteURL(invalid)).toThrow();
    }
    const configured = renderHTML(template, null, "https://confirmed.example/");
    expect(configured).toContain('<link rel="canonical" href="https://confirmed.example/"');
    expect(configured).toContain('content="https://confirmed.example/og-image.png"');
    expect(configured).not.toContain("noindex");
  });
});
