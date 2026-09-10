from __future__ import annotations

from ane_studio_worker import hardware


def test_rejects_intel_mac(monkeypatch) -> None:
    monkeypatch.setattr(hardware.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(hardware.platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(hardware.platform, "mac_ver", lambda: ("15.4", ("", "", ""), ""))
    monkeypatch.setattr(hardware, "_command", lambda *args: "")
    result = hardware.hardware_info()
    assert result["supported"] is False
    assert "Apple Silicon" in result["reason"]


def test_accepts_supported_apple_silicon(monkeypatch) -> None:
    monkeypatch.setattr(hardware.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(hardware.platform, "machine", lambda: "arm64")
    monkeypatch.setattr(hardware.platform, "mac_ver", lambda: ("15.0", ("", "", ""), ""))
    monkeypatch.setattr(
        hardware, "_command", lambda *args: "Apple M5" if "brand" in args[-1] else ""
    )
    result = hardware.hardware_info()
    assert result["supported"] is True
    assert "reason" not in result


def test_unknown_power_source_is_not_reported_as_battery(monkeypatch) -> None:
    monkeypatch.setattr(hardware, "_command", lambda *args: "")
    assert hardware.hardware_info()["powerSource"] == "unknown"


def test_malformed_os_version_is_unsupported_not_crash(monkeypatch) -> None:
    monkeypatch.setattr(hardware.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(hardware.platform, "machine", lambda: "arm64")
    monkeypatch.setattr(hardware.platform, "mac_ver", lambda: ("unknown", ("", "", ""), ""))
    monkeypatch.setattr(hardware, "_command", lambda *args: "")
    result = hardware.hardware_info()
    assert result["supported"] is False
    assert "macOS 15" in result["reason"]
