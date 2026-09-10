# App icon

An original geometric owl on a navy rounded tile. Its white eyes stay readable at small sizes. The violet beak and short chip contacts connect the owl to the local model lab. The icon has no text and uses no third-party artwork, fonts, or remote assets.

The design uses the website's original owl mark as a reference. It is covered by the repository's MIT license.

## Files

- `icon.svg`: editable 1024-unit vector source, with named groups and gradients.
- `icon.png`: 1024 x 1024 RGBA preview and full-size raster.
- `icon.iconset/`: the ten standard macOS icon slots. This generated intermediate is ignored by Git and is not committed. Run the generator to recreate it.
- `icon.icns`: native macOS icon, built with `/usr/bin/iconutil`.
- `generate.ts`: repeatable local rendering and validation.

The iconset contains 16, 32, 128, 256 and 512-point icons at 1x and 2x. Each size renders directly from the SVG, not from an enlarged smaller image. Transparent outer padding leaves room for the tile's shadow.

## Regenerate

From the repository root on macOS, after the normal project dependencies and Playwright browser are installed:

```sh
bun resources/icons/generate.ts
```

The script uses the existing `@playwright/test` dependency and the system `iconutil`. It adds no dependencies and makes no network requests. It checks every PNG's width, height and RGBA format, creates the ICNS, then extracts it and checks all ten icon slots again. Temporary verification files are removed.

## Electron packaging

The release configuration must set these electron-builder properties:

```json
{
  "directories": {
    "buildResources": "resources/icons"
  },
  "mac": {
    "icon": "resources/icons/icon.icns"
  }
}
```

These are packaging assets, not a website favicon. The packaged app must contain the ICNS under `Contents/Resources`, with `CFBundleIconFile` in `Contents/Info.plist` pointing to it. Confirm the icon after launching the packaged app. Electron development mode can keep the host Electron icon unless its Dock icon is set separately.
