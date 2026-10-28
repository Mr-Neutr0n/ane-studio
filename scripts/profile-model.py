"""Run the production worker's parity-gated chat path under Instruments.

Compile inference-signpost.c to an ignored dylib before using --signpost-library.
The JSON output is workload evidence. Accelerator evidence comes from the trace.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import platform
import signal
import subprocess
import tempfile
import time
import uuid
from contextlib import contextmanager
from pathlib import Path

from ane_studio_worker.coreml_backend import POLICIES, CoreMLRuntime
from ane_studio_worker.service import Service


@contextmanager
def record_trace(path: Path | None, duration: int):
    """Attach after loading and wait for Instruments' ready notification."""
    if path is None:
        yield None
        return
    path = path.resolve()
    if path.exists():
        raise ValueError(f"Trace already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    notification = f"dev.harikrishna.anestudio.trace.{uuid.uuid4().hex}"
    system = ctypes.CDLL(None)
    system.notify_register_check.argtypes = [
        ctypes.c_char_p,
        ctypes.POINTER(ctypes.c_int),
    ]
    system.notify_check.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_int)]
    system.notify_cancel.argtypes = [ctypes.c_int]
    token, state = ctypes.c_int(), ctypes.c_int()
    if system.notify_register_check(notification.encode(), ctypes.byref(token)) != 0:
        raise RuntimeError("Could not subscribe to the trace-start notification")
    system.notify_check(token, ctypes.byref(state))
    recorder = None
    try:
        recorder = subprocess.Popen(
            [
                "xcrun",
                "xctrace",
                "record",
                "--template",
                "Core ML",
                "--instrument",
                "Points of Interest",
                "--output",
                str(path),
                "--attach",
                str(os.getpid()),
                "--time-limit",
                f"{duration}s",
                "--no-prompt",
                "--notify-tracing-started",
                notification,
            ]
        )
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            if recorder.poll() is not None:
                raise RuntimeError("Instruments exited before tracing started")
            if system.notify_check(token, ctypes.byref(state)) != 0:
                raise RuntimeError("Could not read the trace-start notification")
            if state.value:
                break
            time.sleep(0.05)
        else:
            raise TimeoutError("Instruments did not start tracing within 30 seconds")
        yield recorder, time.monotonic() + duration
    finally:
        if recorder is not None and recorder.poll() is None:
            recorder.send_signal(signal.SIGINT)
            try:
                recorder.wait(timeout=120)
            except subprocess.TimeoutExpired:
                recorder.terminate()
                recorder.wait(timeout=10)
                raise TimeoutError(
                    "Instruments did not finish saving the trace"
                ) from None
        system.notify_cancel(token)
    if recorder.returncode != 0:
        raise RuntimeError(f"Instruments failed with exit status {recorder.returncode}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--home", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--policy", choices=POLICIES, default="cpuAndNeuralEngine")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--signpost-library", type=Path)
    parser.add_argument(
        "--trace", type=Path, help="Attach Instruments after loading the model"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trace-seconds", type=int, default=120)
    args = parser.parse_args()
    if not 1 <= args.repeats <= 100:
        parser.error("--repeats must be between 1 and 100")
    if not 1 <= args.max_new_tokens <= 128:
        parser.error("--max-new-tokens must be between 1 and 128")
    if args.trace and not args.signpost_library:
        parser.error("--trace requires --signpost-library")
    if not 1 <= args.trace_seconds <= 600:
        parser.error("--trace-seconds must be between 1 and 600")
    os.environ["ANE_STUDIO_HOME"] = str(args.home.resolve())
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    args.home.mkdir(parents=True, exist_ok=True)
    previous_tempdir = tempfile.tempdir
    previous_environment = os.environ.get("TMPDIR")
    with tempfile.TemporaryDirectory(
        prefix=".profile-tmp-", dir=args.home.resolve()
    ) as directory:
        os.environ["TMPDIR"] = directory + os.sep
        tempfile.tempdir = directory
        try:
            _run_model(args)
        finally:
            tempfile.tempdir = previous_tempdir
            if previous_environment is None:
                os.environ.pop("TMPDIR", None)
            else:
                os.environ["TMPDIR"] = previous_environment


def _run_model(args: argparse.Namespace) -> None:
    library = None
    if args.signpost_library:
        library = ctypes.CDLL(str(args.signpost_library.resolve()))
        for name in ("inference_begin", "inference_end"):
            getattr(library, name).argtypes = []
            getattr(library, name).restype = None

    service = Service(lambda event: None)
    record, _, source, package = service._paths(args.model)
    print(
        json.dumps({"stage": "loading", "model": args.model, "policy": args.policy}),
        flush=True,
    )
    runtime = CoreMLRuntime(package, source, record["maxSequenceLength"], args.policy)
    generations = []
    with record_trace(args.trace, args.trace_seconds) as recording:
        recorder, deadline = recording if recording else (None, float("inf"))
        for index in range(args.repeats):
            if recorder is not None and (
                recorder.poll() is not None or time.monotonic() >= deadline
            ):
                raise RuntimeError("Recording ended before all generations completed")
            start = time.monotonic_ns()
            if library:
                library.inference_begin()
            try:
                result = runtime.generate(
                    [{"role": "user", "content": "Say hello in one short sentence."}],
                    args.max_new_tokens,
                    0,
                )
            finally:
                if library:
                    library.inference_end()
            if recorder is not None and (
                recorder.poll() is not None or time.monotonic() >= deadline
            ):
                raise RuntimeError("Recording ended during generation")
            generations.append({"index": index, "startMonotonicNs": start, **result})
            print(
                json.dumps(
                    {"stage": "generated", "index": index, "text": result["text"]}
                ),
                flush=True,
            )
    evidence = {
        "modelId": args.model,
        "processId": os.getpid(),
        "tracePath": str(args.trace.resolve()) if args.trace else None,
        "revision": record["revision"],
        "artifactSha256": record["artifactSha256"],
        "parity": record["parity"],
        "requestedPolicy": args.policy,
        "maxNewTokens": args.max_new_tokens,
        "platform": platform.platform(),
        "signpost": "dev.harikrishna.anestudio/PointsOfInterest/local_generation"
        if library
        else None,
        "signpostsEnabled": library is not None,
        "profiled": args.trace is not None,
        "observedPlacement": "requires separate trace analysis",
        "generations": generations,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(evidence, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
