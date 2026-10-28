from __future__ import annotations

import gc
import threading
import weakref

import pytest

from ane_studio_worker import input_owners


@pytest.fixture(autouse=True)
def clear_released_owners():
    input_owners.prune()
    yield
    gc.collect()
    input_owners.prune()


def test_feed_survives_until_native_owner_releases_then_prunes_on_python_thread():
    np = pytest.importorskip("numpy")
    freed_on = []

    class Tracked(np.ndarray):
        def __del__(self):
            freed_on.append(threading.get_ident())

    array = np.array([1, 2, 3], dtype=np.int32).view(Tracked)
    reference = weakref.ref(array)
    feed = {"input_ids": array}
    native_owner = [array]
    input_owners.hold(feed)
    del array, feed
    assert input_owners.prune() == 0
    assert reference() is native_owner[0]
    assert not freed_on
    native_owner.clear()
    assert not freed_on
    assert input_owners.prune() == 1
    assert reference() is None
    assert freed_on == [threading.get_ident()]


def test_pruning_reclaims_many_retired_feeds_without_mutating_inputs():
    np = pytest.importorskip("numpy")
    baseline = len(input_owners._owners)
    arrays = [np.array([index, index + 1], dtype=np.int32) for index in range(100)]
    references = [weakref.ref(array) for array in arrays]
    for array in arrays:
        input_owners.hold({"input_ids": array})
    del array
    assert len(input_owners._owners) == baseline + 100
    assert input_owners.prune() == 0
    assert all(np.array_equal(array, [index, index + 1]) for index, array in enumerate(arrays))
    arrays.clear()
    assert input_owners.prune() == 100
    assert all(reference() is None for reference in references)
    assert len(input_owners._owners) == baseline


def test_global_ownership_outlives_runtime_objects():
    np = pytest.importorskip("numpy")

    class Runtime:
        pass

    runtime = Runtime()
    runtime.feed = {"input_ids": np.array([7], dtype=np.int32)}
    reference = weakref.ref(runtime.feed["input_ids"])
    native_owner = [runtime.feed["input_ids"]]
    input_owners.hold(runtime.feed)
    del runtime
    gc.collect()
    assert input_owners.prune() == 0
    assert reference() is native_owner[0]
    native_owner.clear()
    assert input_owners.prune() == 1
    assert reference() is None


def test_exit_drain_keeps_shared_arrays_and_respects_two_second_budget(monkeypatch):
    np = pytest.importorskip("numpy")
    array = np.array([5], dtype=np.int32)
    reference = weakref.ref(array)
    input_owners.hold({"input_ids": array})
    clock = [0.0]
    sleeps = []

    def sleep(seconds):
        assert 0 < seconds <= 0.01
        sleeps.append(seconds)
        clock[0] += seconds

    monkeypatch.setattr(input_owners.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(input_owners.time, "sleep", sleep)
    input_owners._drain_at_exit()
    assert sum(sleeps) == pytest.approx(2.0)
    assert reference() is array
    del array
    assert input_owners.prune() == 1
    assert reference() is None
