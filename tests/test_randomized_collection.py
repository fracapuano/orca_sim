import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from orca_sim.demos import collect_randomized as collection


def metadata(seed=10):
    return dict(
        fixed_layout=False,
        seed=seed,
        cube_spawn_regions=collection.read_json(collection.REGIONS),
        object_layout=dict(
            red_cube=[0.25 + seed * 0.001, 0.201, 0.353],
            blue_cube=[0.65 + seed * 0.001, 0.201, 0.353],
            stack_target=[0.5, 0.201, 0.326],
        ),
    )


def test_distinct_seed_layouts_are_in_shared_ten_cm_regions():
    regions = collection.read_json(collection.REGIONS)
    layouts = [collection.check_layout(metadata(seed), regions) for seed in range(20)]
    assert len(set(layouts)) == 20


@pytest.mark.parametrize(
    "change",
    [
        lambda m: m.update(fixed_layout=True),
        lambda m: m.update(seed=None),
        lambda m: m["object_layout"]["red_cube"].__setitem__(0, 0.5),
        lambda m: m["object_layout"]["blue_cube"].__setitem__(2, 0.4),
        lambda m: m["object_layout"]["stack_target"].__setitem__(0, 0.6),
    ],
)
def test_rejects_wrong_distribution_or_changes_beyond_cube_xy(change):
    meta = metadata()
    change(meta)
    with pytest.raises((ValueError, AssertionError)):
        collection.check_layout(meta, collection.read_json(collection.REGIONS))


def test_staging_cleanup_keeps_reports_and_rejects_symlinks(tmp_path):
    recording = tmp_path / "attempt_000000/recording"
    frames = recording / "frames"
    frames.mkdir(parents=True)
    (frames / "frame.png").write_bytes(b"generated staging")
    (recording / "report.json").write_text("{}")
    collection.remove_staging_frames(tmp_path, 0)
    assert (recording / "report.json").exists() and not frames.exists()
    outside = tmp_path / "keep"
    outside.mkdir()
    frames.symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="Unsafe"):
        collection.remove_staging_frames(tmp_path, 0)
    assert outside.exists()


def test_coordinator_retries_failed_seed_and_keeps_exact_success_count(
    tmp_path, monkeypatch
):
    running, peak, launched, assembled = {}, 0, [], []

    class Process:
        def __init__(self, command, stdout, stderr):
            nonlocal peak
            self.command, self.polls, self.code = command, 0, None
            self.pid = len(launched) + 1
            launched.append(command)
            if "record" in command:
                self.kind = "record"
                folder = Path(command[command.index("--output-dir") + 1])
                seed = int(command[command.index("--seed") + 1])
                (folder / "frames").mkdir(parents=True)
                collection.write_json(folder / "report.json", dict(success=seed != 10))
                collection.write_json(folder / "recording.json", metadata(seed))
                self.exit_code = int(seed == 10)
            else:
                self.kind = "package"
                folder = Path(command[command.index("--dataset-dir") + 1])
                (folder / "provenance").mkdir(parents=True)
                collection.write_json(
                    folder / "provenance/validation.json",
                    dict(
                        numeric_alignment_verified=True,
                        lerobot_item_decode_verified=True,
                    ),
                )
                self.exit_code = 0
            running[self.pid] = self.kind
            peak = max(peak, list(running.values()).count("record"))

        def poll(self):
            self.polls += 1
            if self.code is None and self.polls >= 2:
                self.code = self.exit_code
                del running[self.pid]
            return self.code

    monkeypatch.setattr(collection.subprocess, "Popen", Process)
    monkeypatch.setattr(
        collection.subprocess,
        "run",
        lambda command, **kwargs: assembled.append(command),
    )
    monkeypatch.setattr(collection.time, "sleep", lambda _: None)
    monkeypatch.setattr(
        collection.shutil, "disk_usage", lambda _: SimpleNamespace(free=100 * 1024**3)
    )
    root = tmp_path / "collection"
    collection.collect(
        root,
        episodes=3,
        workers=3,
        seed=10,
        max_attempts=8,
        package_python="packaging-python",
        push=False,
        hub_parent=None,
    )
    state = json.loads((root / "status.json").read_text())
    assert peak == 3 and not running
    assert state["status"] == "complete_local"
    assert state["successful_attempts"] == [1, 2, 3]
    assert [item["seed"] for item in state["attempts"]] == [10, 11, 12, 13]
    assert state["attempts"][0]["status"] == "failed"
    assert len(assembled) == 1 and "--push" not in assembled[0]
    assert not list(root.glob("attempt_*/recording/frames"))
