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


def test_official_lerobot_aggregation_preserves_episode_and_camera_alignment(tmp_path):
    """Synthetic three-frame fixtures only; never simulated data or Hub writes."""
    pytest.importorskip("lerobot")
    import numpy as np
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from orca_sim.demos.collect_lerobot_smoke import features

    root = tmp_path / "synthetic_aggregation_fixture"
    root.mkdir()
    regions = collection.read_json(collection.REGIONS)
    collection.write_json(root / "spawn_regions.json", regions)
    collection.write_json(
        root / "status.json",
        dict(
            target_episodes=3,
            successful_attempts=[0, 1, 2],
            spawn_regions=regions,
            fixture="synthetic unit test, not demonstrations",
        ),
    )
    for episode in range(3):
        folder = root / f"attempt_{episode:06d}/dataset"
        meta = metadata(episode)
        meta.update(frames=3, action_semantics="Synthetic unit-test commands only.")
        arrays = dict(
            time=np.arange(3) / collection.FPS,
            state=np.full((3, 48), episode / 10),
            velocity=np.zeros((3, 48)),
            action=np.full((3, 48), episode / 20),
            cube_poses=np.tile(
                np.r_[
                    meta["object_layout"]["red_cube"],
                    [1, 0, 0, 0],
                    meta["object_layout"]["blue_cube"],
                    [1, 0, 0, 0],
                ],
                (3, 1),
            ),
        )
        ds = LeRobotDataset.create(
            repo_id=collection.REPO_ID,
            root=folder,
            fps=collection.FPS,
            robot_type="synthetic_fixture",
            features=features([f"joint_{i}" for i in range(48)]),
            use_videos=True,
            video_backend="pyav",
            vcodec="h264",
            encoder_threads=1,
        )
        try:
            for frame in range(3):
                value = {
                    "observation.state": arrays["state"][frame].astype(np.float32),
                    "observation.velocity": arrays["velocity"][frame].astype(
                        np.float32
                    ),
                    "action": arrays["action"][frame].astype(np.float32),
                    "observation.cube_poses": arrays["cube_poses"][frame].astype(
                        np.float32
                    ),
                    "next.done": np.array([frame == 2]),
                    "next.success": np.array([frame == 2]),
                    "task": collection.TASK,
                }
                for camera_index, camera in enumerate(collection.CAMERAS):
                    value[f"observation.images.{camera}"] = np.full(
                        (480, 640, 3),
                        episode * 50 + frame * 5 + camera_index,
                        dtype=np.uint8,
                    )
                ds.add_frame(value)
            ds.save_episode(parallel_encoding=False)
        finally:
            ds.finalize()
            ds.stop_image_writer()
        provenance = folder / "provenance"
        provenance.mkdir()
        collection.write_json(provenance / "recording.json", meta)
        collection.write_json(
            provenance / "report.json", dict(success=True, fixture=True)
        )
        np.savez_compressed(provenance / "samples.npz", **arrays)
    collection.assemble(root)
    checks = collection.read_json(root / "dataset/provenance/validation.json")
    assert checks["episodes"] == 3 and checks["frames"] == 9
    assert checks["unique_layouts"] == checks["unique_seeds"] == 3
    assert not (root / "upload_result.json").exists()
