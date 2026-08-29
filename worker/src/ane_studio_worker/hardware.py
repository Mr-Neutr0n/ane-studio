from __future__ import annotations

import platform
import subprocess


def _command(*args: str) -> str:
    try:
        return subprocess.run(
            args, check=True, capture_output=True, text=True, timeout=5
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def hardware_info() -> dict[str, object]:
    architecture = platform.machine()
    version = platform.mac_ver()[0]
    major = int(version.split(".")[0]) if version else 0
    supported = platform.system() == "Darwin" and architecture == "arm64" and major >= 15
    reason = None
    if platform.system() != "Darwin":
        reason = "ANE Studio runs only on macOS."
    elif architecture != "arm64":
        reason = "ANE Studio requires an Apple Silicon Mac."
    elif major < 15:
        reason = "ANE Studio requires macOS 15 or newer."
    memory_text = _command("sysctl", "-n", "hw.memsize")
    battery = _command("pmset", "-g", "batt")
    return {
        "supported": supported,
        "platform": platform.system(),
        "architecture": architecture,
        "osVersion": version,
        "chip": _command("sysctl", "-n", "machdep.cpu.brand_string") or "Unknown Apple Silicon",
        "memoryBytes": int(memory_text) if memory_text.isdigit() else 0,
        "powerSource": "ac" if "AC Power" in battery else "battery",
        **({"reason": reason} if reason else {}),
    }
