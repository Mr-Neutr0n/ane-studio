from __future__ import annotations

import importlib.util
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
