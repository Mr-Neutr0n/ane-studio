"""Build the release runtime from a locked, non-editable uv environment.

Run via `bun run runtime:prepare`, not with the system Python.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import shutil
import sys
import sysconfig
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT / "artifacts" / "runtime"
NOTICES = ROOT / "artifacts" / "runtime-notices"
PYTHON_BUILD = "20260510"
LICENSE_REVISION = "f0ae464f527498d6c4b13f19977447d603cd5edd"
LICENSE_NAMES = [
    "LICENSE",
    "LICENSE.bdb.txt",
    "LICENSE.bzip2.txt",
    "LICENSE.cpython.txt",
    "LICENSE.expat.txt",
    "LICENSE.libX11.txt",
    "LICENSE.libXau.txt",
    "LICENSE.libedit.txt",
    "LICENSE.libffi.txt",
    "LICENSE.liblzma.txt",
    "LICENSE.libuuid.txt",
    "LICENSE.libxcb.txt",
    "LICENSE.mpdecimal.txt",
    "LICENSE.ncurses.txt",
    "LICENSE.openssl-1.1.txt",
    "LICENSE.openssl-3.txt",
    "LICENSE.sqlite.txt",
    "LICENSE.tcl.txt",
    "LICENSE.tix.txt",
    "LICENSE.zlib.txt",
    "python-licenses.rst",
]


def notice(url: str, key: str) -> bytes:
    expected = json.loads((ROOT / "scripts" / "runtime-license-lock.json").read_text())[
        key
    ]
    cache = ROOT / "artifacts" / "runtime-license-cache" / expected
    cache.parent.mkdir(parents=True, exist_ok=True)
    if cache.is_file():
        content = cache.read_bytes()
        if hashlib.sha256(content).hexdigest() == expected:
            return content
    for attempt in range(3):
        try:
            with urllib.request.urlopen(url, timeout=30) as response:
                content = response.read()
            break
        except (urllib.error.URLError, TimeoutError):
            if attempt == 2:
                raise
    if hashlib.sha256(content).hexdigest() != expected:
        raise SystemExit(f"Upstream license digest mismatch: {key}")
    cache.write_bytes(content)
    return content


def main() -> None:
    if sys.platform != "darwin" or platform.machine() != "arm64":
        raise SystemExit("Release packaging requires Apple Silicon macOS.")
    if sys.version_info[:3] != (3, 12, 13):
        raise SystemExit(
            "Expected managed CPython 3.12.13. Run bun run runtime:prepare."
        )
    base = Path(sys.base_prefix).resolve()
    if (
        not (base / "BUILD").is_file()
        or (base / "BUILD").read_text().strip() != PYTHON_BUILD
    ):
        raise SystemExit(
            "Expected python-build-standalone 20260510. Use uv 0.11.17 to build releases."
        )
    site = Path(sysconfig.get_paths()["purelib"])
    distributions = sorted(
        importlib.metadata.distributions(path=[str(site)]),
        key=lambda d: d.metadata["Name"].lower(),
    )
    if any(
        path.name not in {"_virtualenv.pth", "distutils-precedence.pth"}
        for path in site.glob("*.pth")
    ):
        raise SystemExit(
            "Runtime environment contains .pth files. Use --no-editable with a clean environment."
        )
    shutil.rmtree(RUNTIME, ignore_errors=True)
    shutil.rmtree(NOTICES, ignore_errors=True)
    ignore = shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo", ".DS_Store")
    shutil.copytree(base, RUNTIME, symlinks=True, ignore=ignore)
    bundled_site = RUNTIME / "lib" / "python3.12" / "site-packages"
    shutil.rmtree(bundled_site)
    shutil.copytree(site, bundled_site, symlinks=True, ignore=ignore)
    for name in ("_virtualenv.pth", "_virtualenv.py"):
        (bundled_site / name).unlink(missing_ok=True)
    for file in (RUNTIME / "bin").iterdir():
        if file.name not in {"python", "python3", "python3.12"}:
            file.unlink()
    for file in bundled_site.glob("*.dist-info/direct_url.json"):
        file.unlink()  # Local build paths are not useful to app users.
    inventory = []
    for dist in distributions:
        name = dist.metadata["Name"]
        license_files = []
        for relative in dist.files or []:
            path = Path(str(relative))
            if any(
                "license" in part.lower()
                or part.lower().startswith(("notice", "copying", "copyright"))
                for part in path.parts
            ):
                source = site / path
                if source.is_file() and source.resolve().is_relative_to(site.resolve()):
                    destination = NOTICES / "python-packages" / name / path
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(source, destination)
                    license_files.append(str(path))
        if not license_files and name == "ane-studio-worker":
            destination = NOTICES / "python-packages" / name / "LICENSE"
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / "LICENSE", destination)
            license_files.append("LICENSE")
        if not license_files and name == "tokenizers" and dist.version == "0.23.1":
            # This wheel omits the upstream license. Keep its exact source notice.
            url = "https://raw.githubusercontent.com/huggingface/tokenizers/7f1623b90b5adfb9bc327d4c3468d2f70bbce262/LICENSE"
            content = notice(url, "tokenizers-0.23.1/LICENSE")
            destination = NOTICES / "python-packages" / name / "LICENSE"
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(content)
            (destination.parent / "SOURCE.txt").write_text(url + "\n")
            license_files.append("LICENSE")
        if not license_files:
            raise SystemExit(
                f"Missing license files for bundled Python package {name}."
            )
        inventory.append(
            {
                "name": name,
                "version": dist.version,
                "license": dist.metadata.get(
                    "License-Expression",
                    dist.metadata.get("License", "See included notices"),
                ),
                "files": license_files,
            }
        )
    standalone = NOTICES / "python-standalone"
    standalone.mkdir(parents=True)
    upstream_files = []
    for name in LICENSE_NAMES:
        url = f"https://raw.githubusercontent.com/astral-sh/python-build-standalone/{LICENSE_REVISION}/{name}"
        content = notice(url, name)
        (standalone / name).write_bytes(content)
        upstream_files.append(
            {"file": name, "sha256": hashlib.sha256(content).hexdigest()}
        )
    (NOTICES / "python-packages" / "inventory.json").write_text(
        json.dumps(inventory, indent=2) + "\n"
    )
    source = ROOT / "worker" / "src" / "ane_studio_worker"
    bundled_worker = bundled_site / "ane_studio_worker"
    source_files = {
        str(path.relative_to(source)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in source.rglob("*.py")
    }
    bundled_files = {
        str(path.relative_to(bundled_worker)): hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
        for path in bundled_worker.rglob("*.py")
    }
    if source_files != bundled_files:
        raise SystemExit(
            "Bundled worker differs from source. Run uv cache clean ane-studio-worker, then rebuild."
        )
    manifest = {
        "workerSources": source_files,
        "python": platform.python_version(),
        "standaloneBuild": PYTHON_BUILD,
        "standaloneLicenseRevision": LICENSE_REVISION,
        "standaloneLicenseFiles": upstream_files,
        "workerLockSha256": hashlib.sha256(
            (ROOT / "worker" / "uv.lock").read_bytes()
        ).hexdigest(),
        "packages": [{"name": i["name"], "version": i["version"]} for i in inventory],
    }
    (RUNTIME / "runtime-manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    for path in RUNTIME.rglob("*"):
        if path.is_symlink() and not path.resolve().is_relative_to(RUNTIME.resolve()):
            raise SystemExit(
                f"Runtime symlink escapes bundle: {path.relative_to(RUNTIME)}"
            )
    print(
        f"Bundled {len(inventory)} Python packages and CPython {platform.python_version()}."
    )


if __name__ == "__main__":
    main()
