import pytest

from orca_sim.demos.collect_lerobot_smoke import CAMERAS, features


def test_smoke_schema_has_matching_named_absolute_joint_targets():
    joints = [f"joint_{i}" for i in range(48)]
    schema = features(joints)
    for key in ("observation.state", "observation.velocity", "action"):
        assert schema[key] == dict(dtype="float32", shape=(48,), names=joints)
    for camera in CAMERAS:
        assert schema[f"observation.images.{camera}"] == dict(
            dtype="video", shape=(480, 640, 3), names=["height", "width", "channels"]
        )
    assert schema["observation.cube_poses"]["shape"] == (14,)
    assert schema["observation.cube_poses"]["names"][3:7] == [
        "red.qw",
        "red.qx",
        "red.qy",
        "red.qz",
    ]
    for key in ("next.done", "next.success"):
        assert schema[key] == dict(dtype="bool", shape=(1,), names=None)


@pytest.mark.parametrize(
    "joints", [[], ["joint"] * 48, [f"joint_{i}" for i in range(47)]]
)
def test_smoke_rejects_wrong_embodiment_or_ambiguous_names(joints):
    with pytest.raises(ValueError, match="48 uniquely named"):
        features(joints)


@pytest.mark.parametrize("episodes,workers", [(0, 3), (3, 0), (-1, 1)])
def test_batch_rejects_invalid_concurrency_before_creating_outputs(
    tmp_path, episodes, workers
):
    from orca_sim.demos.collect_lerobot_smoke import record_batch

    output = tmp_path / "batch"
    with pytest.raises(ValueError, match="must be positive"):
        record_batch(output, episodes, workers)
    assert not output.exists()


def test_batch_uses_bounded_processes_and_disjoint_outputs(tmp_path, monkeypatch):
    import json
    from pathlib import Path
    from orca_sim.demos import collect_lerobot_smoke as smoke

    active, peak, paths = 0, 0, []

    class Process:
        def __init__(self, command, stdout, stderr):
            nonlocal active, peak
            assert command[1:5] == [
                "-m",
                "orca_sim.demos.collect_lerobot_smoke",
                "record",
                "--output-dir",
            ]
            self.path = Path(command[-1])
            assert self.path not in paths
            paths.append(self.path)
            self.path.mkdir()
            (self.path / "report.json").write_text(json.dumps(dict(success=True)))
            (self.path / "record_timing.json").write_text(
                json.dumps(dict(record_total_seconds=1))
            )
            self.pid, self.polls, self.code = len(paths), 0, None
            active += 1
            peak = max(peak, active)

        def poll(self):
            nonlocal active
            self.polls += 1
            if self.code is None and self.polls == 3:
                active -= 1
                self.code = 0
            return self.code

    monkeypatch.setattr(smoke.subprocess, "Popen", Process)
    monkeypatch.setattr(smoke, "sleep", lambda _: None)
    result = smoke.record_batch(tmp_path / "batch", episodes=3, workers=2)
    assert peak == 2 and active == 0
    assert result["successful_episodes"] == 3
    assert [path.name for path in paths] == [
        "episode_000000",
        "episode_000001",
        "episode_000002",
    ]
