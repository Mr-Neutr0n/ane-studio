"""Correlate exported Instruments ANE predictions with this worker's signposts."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path


def rows(path: Path, schema: str, required: set[str]) -> list[dict[str, str]]:
    root = ET.parse(path).getroot()
    references = {
        element.attrib["id"]: element
        for element in root.iter()
        if "id" in element.attrib
    }
    result = []
    columns = [
        column.findtext("mnemonic") for column in root.findall("node/schema/col")
    ]
    if not any(
        item.get("name") == schema for item in root.findall("node/schema")
    ) or not required.issubset(columns):
        raise ValueError(
            f"Export does not contain the expected {schema} table and columns"
        )
    for table in root.findall("node"):
        for row in table.findall("row"):
            values = {}
            for name, element in zip(columns, row, strict=True):
                element = references.get(element.get("ref"), element)
                values[name] = element.text or ""
                values[f"{name}:formatted"] = element.get("fmt", "")
            result.append(values)
    return result


def summarize(signposts: Path, ane: Path, workload: dict) -> dict:
    pending = {}
    intervals = []
    for row in rows(
        signposts,
        "os-signpost",
        {"time", "process", "identifier", "event-type", "name", "subsystem"},
    ):
        if (
            row.get("subsystem") != "dev.harikrishna.anestudio"
            or row.get("name") != "local_generation"
        ):
            continue
        process = re.search(r"\((\d+)\)$", row["process:formatted"])
        if not process or int(process[1]) != workload["processId"]:
            raise ValueError("Signpost process does not match the worker workload")
        key = (row["process:formatted"], row["identifier"])
        event_type = row["event-type:formatted"]
        if event_type == "Begin":
            if key in pending:
                raise ValueError("Duplicate generation begin signpost")
            pending[key] = int(row["time"])
        elif event_type == "End":
            if key not in pending:
                raise ValueError("Generation end has no matching begin signpost")
            start = pending.pop(key)
            end = int(row["time"])
            if end <= start:
                raise ValueError("Generation signpost has a nonpositive duration")
            intervals.append((start, end))
    if pending or not intervals or len(intervals) != len(workload["generations"]):
        raise ValueError(
            "Trace must contain one complete signpost interval per generation"
        )
    predictions = [
        row
        for row in rows(
            ane, "ane-hw-intervals-internal", {"start", "duration", "event-label"}
        )
        if "prediction" in row.get("event-label:formatted", "").lower()
    ]
    overlaps = []
    per_generation = []
    for start, end in intervals:
        matched = [
            row
            for row in predictions
            if int(row["start"]) < end
            and int(row["start"]) + int(row["duration"]) > start
        ]
        overlaps.extend(matched)
        per_generation.append(len(matched))
    return {
        "modelId": workload["modelId"],
        "revision": workload["revision"],
        "artifactSha256": workload["artifactSha256"],
        "requestedPolicy": workload["requestedPolicy"],
        "completeGenerationIntervals": len(intervals),
        "workerPredictions": sum(
            len(item["samples"]) for item in workload["generations"]
        ),
        "anePredictionIntervalsInTrace": len(predictions),
        "anePredictionIntervalsOverlappingGeneration": len(overlaps),
        "anePredictionsPerGeneration": per_generation,
        "aneObservedDuringEveryGeneration": all(per_generation),
        "note": "ANE hardware intervals are system-wide. Compare the same model's CPU-only control.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("trace", "workload", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    workload = json.loads(args.workload.read_text())
    if Path(workload["tracePath"]).resolve() != args.trace.resolve():
        raise ValueError("Trace path does not match the worker workload")
    # Both lanes come from this one trace, never caller-supplied XML from different runs.
    paths = {}
    for name, schema in (
        ("signposts", "os-signpost"),
        ("ane", "ane-hw-intervals-internal"),
    ):
        path = args.output.with_suffix(f".{name}.xml")
        subprocess.run(
            [
                "xcrun",
                "xctrace",
                "export",
                "--input",
                str(args.trace),
                "--xpath",
                f'/trace-toc/run/data/table[@schema="{schema}"]',
                "--output",
                str(path),
            ],
            check=True,
        )
        paths[name] = path
    result = summarize(paths["signposts"], paths["ane"], workload)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
