from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "summarize-profile.py"
spec = importlib.util.spec_from_file_location("profile_summary", SCRIPT)
assert spec and spec.loader
profile_summary = importlib.util.module_from_spec(spec)
spec.loader.exec_module(profile_summary)


def xml_table(columns, body, schema_name="os-signpost"):
    schema = "".join(f"<col><mnemonic>{name}</mnemonic></col>" for name in columns)
    return (
        f'<trace-query-result><node><schema name="{schema_name}">{schema}</schema>'
        f"{body}</node></trace-query-result>"
    )


def test_trace_summary_requires_aligned_predictions_and_resolves_references(tmp_path):
    signposts = tmp_path / "signposts.xml"
    signposts.write_text(
        xml_table(
            ["time", "process", "identifier", "event-type", "name", "subsystem"],
            '<row><time>100</time><process id="p" fmt="python (123)"/>'
            '<id id="i">1</id><type fmt="Begin"/><name id="n">local_generation</name>'
            '<subsystem id="s">dev.harikrishna.anestudio</subsystem></row>'
            '<row><time>200</time><process ref="p"/><id ref="i"/><type fmt="End"/>'
            '<name ref="n"/><subsystem ref="s"/></row>',
        )
    )
    ane = tmp_path / "ane.xml"
    ane.write_text(
        xml_table(
            ["start", "duration", "event-label"],
            "<row><time>110</time><duration>10</duration>"
            '<label id="prediction" fmt="AneInference Prediction"/></row>'
            "<row><time>150</time><duration>10</duration>"
            '<label fmt="Neural Engine Compile"/></row>'
            '<row><time>250</time><duration>10</duration><label ref="prediction"/></row>',
            "ane-hw-intervals-internal",
        )
    )
    workload = {
        "modelId": "org/model",
        "processId": 123,
        "revision": "a" * 40,
        "artifactSha256": "b" * 64,
        "requestedPolicy": "cpuAndNeuralEngine",
        "generations": [{"samples": [1]}],
    }
    summary = profile_summary.summarize(signposts, ane, workload)
    assert summary["anePredictionIntervalsInTrace"] == 2
    assert summary["anePredictionIntervalsOverlappingGeneration"] == 1
    assert summary["aneObservedDuringEveryGeneration"] is True

    workload["generations"].append({"samples": [1]})
    with pytest.raises(ValueError, match="complete signpost"):
        profile_summary.summarize(signposts, ane, workload)
    workload["generations"].pop()
    workload["processId"] = 456
    with pytest.raises(ValueError, match="does not match"):
        profile_summary.summarize(signposts, ane, workload)
    workload["processId"] = 123
    ane.write_text("<trace-query-result/>")
    with pytest.raises(ValueError, match="expected ane-hw"):
        profile_summary.summarize(signposts, ane, workload)
    ane.write_text(xml_table(["start", "duration", "event-label"], "", "ane-hw-intervals-internal"))
    assert not profile_summary.summarize(signposts, ane, workload)[
        "aneObservedDuringEveryGeneration"
    ]


def test_no_signposts_cannot_be_reported_as_a_negative_ane_result(tmp_path):
    path = tmp_path / "empty.xml"
    path.write_text("<trace-query-result/>")
    with pytest.raises(ValueError, match="expected os-signpost"):
        profile_summary.summarize(path, path, {"generations": []})


def test_interval_spanning_two_generations_is_counted_once(tmp_path):
    signposts = tmp_path / "signposts.xml"
    begin_end = "".join(
        f'<row><time>{time}</time><process fmt="python (123)"/><id>{identifier}</id>'
        f'<type fmt="{kind}"/><name>local_generation</name>'
        "<subsystem>dev.harikrishna.anestudio</subsystem></row>"
        for time, identifier, kind in [
            (100, 1, "Begin"),
            (200, 1, "End"),
            (210, 2, "Begin"),
            (300, 2, "End"),
        ]
    )
    signposts.write_text(
        xml_table(["time", "process", "identifier", "event-type", "name", "subsystem"], begin_end)
    )
    ane = tmp_path / "ane.xml"
    ane.write_text(
        xml_table(
            ["start", "duration", "event-label"],
            '<row><time>190</time><duration>30</duration><label fmt="Prediction"/></row>',
            "ane-hw-intervals-internal",
        )
    )
    workload = {
        "modelId": "org/model",
        "processId": 123,
        "revision": "a" * 40,
        "artifactSha256": "b" * 64,
        "requestedPolicy": "cpuAndNeuralEngine",
        "generations": [{"samples": [1]}, {"samples": [1]}],
    }
    summary = profile_summary.summarize(signposts, ane, workload)
    assert summary["anePredictionsPerGeneration"] == [1, 1]
    assert summary["anePredictionIntervalsInTrace"] == 1
    assert summary["anePredictionIntervalsOverlappingGeneration"] == 1


def test_workload_without_trace_is_rejected_clearly(monkeypatch, tmp_path):
    workload = tmp_path / "workload.json"
    workload.write_text(json.dumps({"tracePath": None}))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "summarize-profile.py",
            "--trace",
            str(tmp_path / "run.trace"),
            "--workload",
            str(workload),
            "--output",
            str(tmp_path / "summary.json"),
        ],
    )

    def unexpected(*args, **kwargs):
        pytest.fail("export must not run without a recorded trace")

    monkeypatch.setattr(profile_summary.subprocess, "run", unexpected)
    with pytest.raises(ValueError, match="no trace"):
        profile_summary.main()
