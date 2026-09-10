import { cp, mkdir, readFile, rm, writeFile } from "node:fs/promises";
import { resolve } from "node:path";
import { resolveSiteURL } from "./site";

export const websiteRoot = resolve(import.meta.dir, "..");
export const repositoryURL = "https://github.com/Mr-Neutr0n/ane-studio";

export type Release = {
  version: string;
  assetUrl: string;
  releaseUrl: string;
  checksumsUrl: string;
  publishedAt: string;
};

function escapeHTML(value: string): string {
  return value.replace(/[&<>"']/g, (character) => {
    const entities: Record<string, string> = {
      "&": "&amp;",
      "<": "&lt;",
      ">": "&gt;",
      '"': "&quot;",
      "'": "&#39;",
    };
    return entities[character];
  });
}

export function validateRelease(input: unknown): Release | null {
  if (input === null) return null;
  if (typeof input !== "object" || Array.isArray(input)) {
    throw new Error("release.json must be null or a published release object.");
  }
  const release = input as Record<string, unknown>;
  for (const key of ["version", "assetUrl", "releaseUrl", "checksumsUrl", "publishedAt"]) {
    if (typeof release[key] !== "string") throw new Error(`Missing release field: ${key}`);
  }
  const { version, assetUrl, releaseUrl, checksumsUrl, publishedAt } = release as Release;
  if (!/^\d+\.\d+\.\d+$/.test(version)) throw new Error("Use a stable semantic release version.");
  const tag = `v${version}`;
  const expectedPrefix = `${repositoryURL}/releases/download/${tag}/`;
  if (assetUrl !== `${expectedPrefix}ANE-Studio-${version}-arm64.zip`) {
    throw new Error(
      "Release asset must be the versioned Apple Silicon ZIP on the project repository.",
    );
  }
  if (releaseUrl !== `${repositoryURL}/releases/tag/${tag}`) {
    throw new Error("Release notes must use the matching project release tag.");
  }
  if (checksumsUrl !== `${expectedPrefix}SHA256SUMS`) {
    throw new Error("Release must link to the matching SHA256SUMS asset.");
  }
  const published = Date.parse(publishedAt);
  if (!Number.isFinite(published) || published > Date.now()) {
    throw new Error("Release must have an actual, non-future publication date.");
  }
  return { version, assetUrl, releaseUrl, checksumsUrl, publishedAt };
}

export function renderHTML(
  template: string,
  releaseInput: unknown,
  siteURL: string | null = null,
): string {
  const release = validateRelease(releaseInput);
  const values: Record<string, string> = {
    CANONICAL_METADATA: siteURL
      ? `<link rel="canonical" href="${escapeHTML(siteURL)}" /><meta property="og:url" content="${escapeHTML(siteURL)}" />`
      : '<meta name="robots" content="noindex, nofollow" />',
    SOCIAL_IMAGE_URL: siteURL ? escapeHTML(new URL("og-image.png", siteURL).href) : "/og-image.png",
    PRIMARY_URL: release ? escapeHTML(release.assetUrl) : "#install",
    PRIMARY_LABEL: release ? "Download for macOS" : "Run from source",
    PRIMARY_SHORT: release ? "Download" : "Get ANE Studio",
    RELEASE_SHORT: release ? `v${escapeHTML(release.version)} · Unsigned` : "Source available",
    INSTALL_SUMMARY: release
      ? "Download the self-contained Apple Silicon app, or run it from source. The ZIP includes the local Python worker and its dependencies. Models download separately."
      : "The source is ready to run. There is no public app download linked yet. Start with the commands here, then let ANE Studio prepare your first model.",
    RELEASE_PANEL: release
      ? `<div class="release-panel"><a class="button button-primary" href="${escapeHTML(release.assetUrl)}">Download v${escapeHTML(release.version)}<span aria-hidden="true">↓</span></a><p>Unsigned and not notarized. <a href="${escapeHTML(release.releaseUrl)}">Read installation notes</a> and <a href="${escapeHTML(release.checksumsUrl)}">verify SHA256 checksums</a> before opening. Do not disable Gatekeeper globally.</p></div>`
      : "",
  };
  const html = template.replace(/\{\{([A-Z_]+)\}\}/g, (_, key: string) => {
    if (!(key in values)) throw new Error(`Unknown template field: ${key}`);
    return values[key];
  });
  if (/\{\{[^}]+\}\}/.test(html)) throw new Error("Unresolved template field.");
  return html;
}

export async function build(): Promise<void> {
  const output = resolve(websiteRoot, "dist");
  const template = await readFile(resolve(websiteRoot, "src/index.html"), "utf8");
  const release = JSON.parse(await readFile(resolve(websiteRoot, "release.json"), "utf8"));
  const site = JSON.parse(await readFile(resolve(websiteRoot, "site.json"), "utf8"));
  const siteURL = resolveSiteURL(
    site.url,
    process.env.VERCEL_ENV === "production" ? process.env.VERCEL_PROJECT_PRODUCTION_URL : undefined,
  );
  const html = renderHTML(template, release, siteURL);
  await rm(output, { recursive: true, force: true });
  await mkdir(output, { recursive: true });
  await cp(resolve(websiteRoot, "public"), output, { recursive: true });
  for (const file of ["styles.css", "main.js"]) {
    await cp(resolve(websiteRoot, "src", file), resolve(output, file));
  }
  await writeFile(resolve(output, "index.html"), html);
  await writeFile(
    resolve(output, "robots.txt"),
    siteURL
      ? `User-agent: *\nAllow: /\n\nSitemap: ${siteURL}sitemap.xml\n`
      : "User-agent: *\nDisallow: /\n",
  );
  if (siteURL) {
    await writeFile(
      resolve(output, "sitemap.xml"),
      `<?xml version="1.0" encoding="UTF-8"?><urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9"><url><loc>${escapeHTML(siteURL)}</loc></url></urlset>\n`,
    );
  }
  console.log(`Canonical URL: ${siteURL || "not set; domain confirmation pending"}`);
  await writeFile(
    resolve(output, "release.json"),
    `${JSON.stringify(validateRelease(release), null, 2)}
`,
  );
  console.log(`Built static ANE Studio website at ${output}`);
  console.log(
    release
      ? "Primary action: versioned published macOS release"
      : "Primary action: source install, no download promised",
  );
}

if (import.meta.main) await build();
