# ANE Studio website

Static product website. It describes the macOS desktop app; it does not run inference in the browser. No backend, accounts, analytics, external fonts, or credential handling.

## Local preview

From the repository root:

```bash
bun install --cwd website --frozen-lockfile
bun run --cwd website dev
```

Open `http://127.0.0.1:4173`. The preview server binds to loopback only.

## Verify

Install browser test engines once:

```bash
cd website
bunx playwright install chromium webkit
```

On a Linux CI host, use `bunx playwright install --with-deps chromium webkit`.

Run from the repository root:

```bash
bun run --cwd website verify
```

The command builds the static site, runs Bun structural tests, then runs Playwright smoke tests in desktop Chromium, 390px mobile Chromium, 320px Chromium, and mobile WebKit. Browser checks cover navigation, images, screenshot controls, clipboard success and denial, no JavaScript, keyboard controls, reduced motion, no horizontal overflow, local-only requests, security headers, and automated WCAG 2.2 AA checks in collapsed and expanded FAQ states. Automated accessibility checks do not replace a manual screen-reader review.

For a build and structural check without browser engines:

```bash
bun run --cwd website check
```

Screenshots and traces go to ignored `website/test-results/`. The browser HTML report is in ignored `website/playwright-report/`. Build output is ignored at `website/dist/`.

## Content and assets

- `src/index.html`: semantic product content with release placeholders.
- `src/styles.css`: responsive styles, reduced-motion and forced-color support.
- `src/main.js`: optional screenshot switch and copy control. The install commands and FAQ work without JavaScript.
- `public/`: original owl favicon, social card, and compressed screenshots copied from this repository's `docs/assets/`.
- `scripts/build.ts`: dependency-free Bun static build with a strict release URL gate.
- `scripts/serve.ts`: local static preview, not a production backend.
- `release.json`: `null` until an actual public macOS release is available.

The screenshots show ANE Studio development runs, not a benchmark guarantee. Published model claims must link to checkpoint-level evidence. Keep the cosine threshold at least `0.999` **and** require a matching top token. Requested policy, planned placement, and observed activity are separate claims. Only an aligned Instruments trace proves observed accelerator activity.

Public branding is ANE Studio. The repository URL and app data directory stay unchanged. `site.json` holds the canonical URL in one `url` value. It is `null` while the owner confirms the domain. Local builds omit the canonical and ask crawlers not to index the page. Only a production build can fall back to Vercel's stable `VERCEL_PROJECT_PRODUCTION_URL`. Preview builds leave the canonical unset. Set `url` to the confirmed HTTPS origin only after ownership and routing work. The build generates canonical/Open Graph metadata, robots, and sitemap together. Do not attach or overwrite an existing unrelated Vercel project.

## Enable a real download

Do not add a speculative download URL. First publish and validate the tested ZIP, release notes, and checksums. Then replace `release.json` with:

```json
{
  "version": "0.1.0",
  "assetUrl": "https://github.com/Mr-Neutr0n/ane-studio/releases/download/v0.1.0/ANE-Studio-0.1.0-arm64.zip",
  "releaseUrl": "https://github.com/Mr-Neutr0n/ane-studio/releases/tag/v0.1.0",
  "checksumsUrl": "https://github.com/Mr-Neutr0n/ane-studio/releases/download/v0.1.0/SHA256SUMS",
  "publishedAt": "REPLACE_WITH_ACTUAL_PUBLICATION_DATE"
}
```

Use the actual version and publication date. Confirm every URL responds successfully, verify the SHA256 against the packaged artifact, and confirm the downloaded app's startup test before enabling the download. The build validates URL origin, tag, asset name, checksum path, and a non-future publication date. It does not make network requests or independently certify release availability. Run `verify` after changing this file. The site then uses a versioned download as the primary action and retains source instructions. It states that the app is unsigned and not notarized.

## Deployment settings

Root `vercel.json` builds only this directory:

- Framework: none
- Install: `cd website && bun install --frozen-lockfile`
- Build: `cd website && bun run build`
- Output: `website/dist`

Use a distinct Vercel project after the owner confirms the product name and domain. Deploy only website source, assets, the isolated lockfile, and root Vercel config. Use a strict upload allowlist or a clean staging directory. Never upload the Electron app, Python virtual environment, model weights, generated packages, test reports, or credentials.

Production and preview send the same Content Security Policy restrictions, except the local HTTP preview omits `upgrade-insecure-requests` because WebKit upgrades loopback assets to nonexistent HTTPS. Production retains the directive. Both send frame restrictions, referrer policy, MIME protection, and permissions policy. The site can connect to no remote API. Public source, model, and prerequisite links only navigate when a visitor chooses them.
