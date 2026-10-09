"""Record, package, validate and privately upload one ORCA-hand stacking episode.

Recording and packaging are separate commands so MuJoCo can keep its validated
Python environment while LeRobot uses its own dependencies. No network writes
occur until the explicit upload command. Existing datasets are never overwritten.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil
import subprocess
import sys
from time import perf_counter, sleep

import numpy as np

REPO_ID = "fracapuano/orcapanda_cube_stacking_scripted"
CAMERAS = ("front", "right_wrist", "left_wrist")
TASK = (
    "Grasp the red cube with the right arm and the blue cube with the left arm; "
    "place blue on the center X, stack red on blue, and release both cubes."
)
FPS = 20


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + "\n")


def features(joints):
    if len(joints) != 48 or len(set(joints)) != 48:
        raise ValueError("Expected the 48 uniquely named ORCA/Panda actuator joints")

    def vector(names):
        return dict(dtype="float32", shape=(len(names),), names=list(names))

    return {
        "observation.state": vector(joints),
        "observation.velocity": vector(joints),
        "action": vector(joints),
        "observation.cube_poses": vector(
            [
                f"{cube}.{axis}"
                for cube in ("red", "blue")
                for axis in ("x", "y", "z", "qw", "qx", "qy", "qz")
            ]
        ),
        "next.done": dict(dtype="bool", shape=(1,), names=None),
        "next.success": dict(dtype="bool", shape=(1,), names=None),
        **{
            f"observation.images.{name}": dict(
                dtype="video",
                shape=(480, 640, 3),
                names=["height", "width", "channels"],
            )
            for name in CAMERAS
        },
    }


def record(root):
    from PIL import Image
    from orca_sim.demos import bimanual_stack as demo

    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=False)
    began = perf_counter()
    timings = dict(render_seconds=0.0, physics_seconds=0.0, image_wait_seconds=0.0)
    samples, controls = [], []
    metadata = {}
    original = demo.OrcaBimanualPandaCubeStackingMujoco

    class RecordingEnv(original):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            assert self.frame_skip == 1 and abs(self.model.opt.timestep - 0.001) < 1e-12
            assert tuple(self.camera_names) == CAMERAS and self.model.nu == 48
            self.pool = ThreadPoolExecutor(max_workers=3)
            self.pending = []
            self.steps, self.next_progress = 0, 0
            for camera in CAMERAS:
                (root / "frames" / camera).mkdir(parents=True)
            metadata.update(
                joint_names=list(self.actuator_joints),
                hand_assignment=self.hand_assignment,
                action_low=self.action_space.low.tolist(),
                action_high=self.action_space.high.tolist(),
                home_qpos=self.data.qpos.tolist(),
                model_sha256=hashlib.sha256(self.xml.encode()).hexdigest(),
                robot_type="bimanual_orcapanda_native_orca_hands",
                simulation_timestep_s=0.001,
                fps=FPS,
                resolution=[640, 480],
                cameras=list(CAMERAS),
                fixed_layout=True,
                source_commit=subprocess.check_output(
                    ["git", "rev-parse", "HEAD"], text=True
                ).strip(),
                mujoco_version=importlib.metadata.version("mujoco"),
                numpy_version=np.__version__,
                alignment="Images and state are captured before the recorded action is applied.",
                action_semantics="Absolute joint-position targets in radians, in joint_names order. "
                "The expert updates at 1000 Hz; action is a 20 Hz subsample, not a 50 ms "
                "zero-order-held command. Full 1000 Hz controls are retained separately. "
                "Equivalent behavior under a 20 Hz learned policy is not validated.",
            )

        def step(self, action):
            applied = np.clip(action, self.action_space.low, self.action_space.high)
            if self.steps % 50 == 0:
                t = perf_counter()
                images = self.render_cameras(width=640, height=480)
                timings["render_seconds"] += perf_counter() - t
                index = len(samples)
                assert abs(self.data.time - index / FPS) < 1e-8
                samples.append(
                    dict(
                        time=float(self.data.time),
                        state=self.data.qpos[self.qadr].copy(),
                        velocity=self.data.qvel[self.dadr].copy(),
                        action=applied.copy(),
                        qpos=self.data.qpos.copy(),
                        qvel=self.data.qvel.copy(),
                        cube_poses=np.concatenate(
                            [
                                np.r_[
                                    self.data.body(name).xpos,
                                    self.data.body(name).xquat,
                                ]
                                for name in ("red_cube", "blue_cube")
                            ]
                        ),
                    )
                )
                for camera, pixels in images.items():
                    path = root / "frames" / camera / f"{index:06d}.png"
                    self.pending.append(
                        self.pool.submit(
                            Image.fromarray(pixels).save, path, compress_level=1
                        )
                    )
                # Bounded backpressure: never drop images or grow memory without limit.
                t = perf_counter()
                while len(self.pending) > 9:
                    self.pending.pop(0).result()
                timings["image_wait_seconds"] += perf_counter() - t
            controls.append(applied.copy())
            t = perf_counter()
            result = super().step(action)
            timings["physics_seconds"] += perf_counter() - t
            self.steps += 1
            if self.data.time >= self.next_progress:
                print(
                    f"RECORD sim={self.data.time:.1f}/69s wall={perf_counter() - began:.1f}s frames/camera={len(samples)}",
                    flush=True,
                )
                self.next_progress += 5
            return result

        def close(self):
            try:
                for future in self.pending:
                    future.result()
            finally:
                self.pool.shutdown(wait=True)
                super().close()

    demo.OrcaBimanualPandaCubeStackingMujoco = RecordingEnv
    try:
        report = demo.run(root, video=False)
    finally:
        demo.OrcaBimanualPandaCubeStackingMujoco = original
    if not report["success"]:
        raise RuntimeError(
            "Recorded grasp/stack failed; retained local artifacts, no dataset/upload"
        )
    np.savez_compressed(
        root / "samples.npz",
        **{key: np.asarray([sample[key] for sample in samples]) for key in samples[0]},
    )
    np.savez_compressed(
        root / "controls_1000hz.npz", action=np.asarray(controls), timestep_s=0.001
    )
    metadata.update(frames=len(samples), simulation_seconds=report["duration_s"])
    write_json(root / "recording.json", metadata)
    timings["record_total_seconds"] = perf_counter() - began
    write_json(root / "record_timing.json", timings)
    print(json.dumps(timings, indent=2), flush=True)


def record_batch(root, episodes=3, workers=3):
    """Independent processes, disjoint outputs; no shared dataset or Hub writes.

    This is a fixed-layout throughput test, not randomized demonstration collection.
    Cameras remain sequential within each worker to avoid nested render workers.
    """
    if episodes < 1 or workers < 1:
        raise ValueError("episodes and workers must be positive")
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=False)
    began = perf_counter()
    active, results = {}, []
    next_episode, next_progress = 0, 0.0
    try:
        while active or next_episode < episodes:
            while next_episode < episodes and len(active) < workers:
                i = next_episode
                output = root / f"episode_{i:06d}"
                log_path = root / f"episode_{i:06d}.log"
                log = log_path.open("w")
                try:
                    process = subprocess.Popen(
                        [
                            sys.executable,
                            "-m",
                            "orca_sim.demos.collect_lerobot_smoke",
                            "record",
                            "--output-dir",
                            str(output),
                        ],
                        stdout=log,
                        stderr=subprocess.STDOUT,
                    )
                except BaseException:
                    log.close()
                    raise
                active[i] = (process, log, log_path, output, perf_counter())
                print(f"START episode={i} pid={process.pid}", flush=True)
                next_episode += 1
            for i, (process, log, log_path, output, started) in list(active.items()):
                code = process.poll()
                if code is None:
                    continue
                log.close()
                report_path = output / "report.json"
                report = (
                    json.loads(report_path.read_text()) if report_path.exists() else {}
                )
                timing_path = output / "record_timing.json"
                timing = (
                    json.loads(timing_path.read_text()) if timing_path.exists() else {}
                )
                result = dict(
                    episode=i,
                    returncode=code,
                    success=code == 0 and bool(report.get("success")),
                    process_wall_seconds=perf_counter() - started,
                    timings=timing,
                    output_dir=str(output),
                )
                results.append(result)
                del active[i]
                print("FINISH " + json.dumps(result), flush=True)
            if perf_counter() - began >= next_progress:
                progress = []
                for i, (_, _, log_path, _, _) in active.items():
                    lines = [
                        line
                        for line in log_path.read_text().splitlines()
                        if line.startswith("RECORD ")
                    ]
                    progress.append(
                        f"episode {i}: {lines[-1] if lines else 'initializing'}"
                    )
                if progress:
                    print(
                        f"BATCH wall={perf_counter() - began:.1f}s | "
                        + " | ".join(progress),
                        flush=True,
                    )
                next_progress = perf_counter() - began + 20
            if active:
                sleep(0.2)
    finally:
        # Only stop children owned by this batch, preserving partial recordings.
        for process, log, _, _, _ in active.values():
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            log.close()
    wall = perf_counter() - began
    successes = sum(result["success"] for result in results)
    summary = dict(
        workers=min(workers, episodes),
        episodes=episodes,
        successful_episodes=successes,
        batch_wall_seconds=wall,
        successful_episodes_per_hour=successes * 3600 / wall,
        results=sorted(results, key=lambda r: r["episode"]),
        scope="Fixed-layout recording only; no packaging, upload, or randomized success-rate test.",
    )
    write_json(root / "batch_result.json", summary)
    print(json.dumps(summary, indent=2), flush=True)
    if successes != episodes:
        raise RuntimeError("Some recordings failed; inspect the separate episode logs")
    return summary


def validate(root, repo_id, expected=None):
    import av
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    root = Path(root)
    dataset = LeRobotDataset(repo_id, root=root, video_backend="pyav")
    assert dataset.num_episodes == 1 and dataset.fps == FPS
    count = dataset.num_frames
    if expected is not None:
        assert count == len(expected["time"])
    table = dataset.hf_dataset.with_format("numpy")
    np.testing.assert_allclose(
        np.asarray(table["timestamp"]), np.arange(count) / FPS, atol=1e-5
    )
    assert np.all(np.asarray(table["episode_index"]) == 0)
    for key, source in (
        ("observation.state", "state"),
        ("observation.velocity", "velocity"),
        ("action", "action"),
        ("observation.cube_poses", "cube_poses"),
    ):
        values = np.asarray(table[key])
        assert np.isfinite(values).all()
        if expected is not None:
            np.testing.assert_allclose(
                values, expected[source].astype(np.float32), atol=0, rtol=0
            )
    done = np.asarray(table["next.done"]).reshape(-1)
    success = np.asarray(table["next.success"]).reshape(-1)
    assert not done[:-1].any() and done[-1] and not success[:-1].any() and success[-1]
    videos = {}
    for camera in CAMERAS:
        paths = list((root / "videos" / f"observation.images.{camera}").rglob("*.mp4"))
        assert len(paths) == 1
        decoded = 0
        with av.open(str(paths[0])) as container:
            for frame in container.decode(video=0):
                assert (frame.width, frame.height) == (640, 480)
                assert abs(float(frame.time) - decoded / FPS) < 1e-6
                decoded += 1
        assert decoded == count, (camera, decoded, count)
        videos[camera] = decoded
    for index in (0, count // 4, count // 2, 3 * count // 4, count - 1):
        item = dataset[index]
        for camera in CAMERAS:
            image = item[f"observation.images.{camera}"]
            assert tuple(image.shape) == (3, 480, 640)
            assert image.isfinite().all() and 0 <= image.min() <= image.max() <= 1
    result = dict(
        episodes=1,
        frames=count,
        fps=FPS,
        decoded_video_frames=videos,
        numeric_alignment_verified=expected is not None,
        lerobot_item_decode_verified=True,
    )
    return result


def package(recording, root, repo_id):
    from PIL import Image
    from huggingface_hub import DatasetCard, DatasetCardData
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    began = perf_counter()
    recording, root = Path(recording), Path(root)
    meta = json.loads((recording / "recording.json").read_text())
    report = json.loads((recording / "report.json").read_text())
    assert report["success"]
    arrays = np.load(recording / "samples.npz")
    n = len(arrays["time"])
    assert n == meta["frames"] and n > 1000
    for camera in CAMERAS:
        assert len(list((recording / "frames" / camera).glob("*.png"))) == n
    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        root=root,
        fps=FPS,
        robot_type=meta["robot_type"],
        features=features(meta["joint_names"]),
        use_videos=True,
        image_writer_threads=6,
        video_backend="pyav",
        vcodec="h264",
        encoder_threads=2,
    )
    try:
        for i in range(n):
            frame = {
                "observation.state": arrays["state"][i].astype(np.float32),
                "observation.velocity": arrays["velocity"][i].astype(np.float32),
                "action": arrays["action"][i].astype(np.float32),
                "observation.cube_poses": arrays["cube_poses"][i].astype(np.float32),
                "next.done": np.array([i == n - 1]),
                "next.success": np.array([i == n - 1]),
                "task": TASK,
            }
            for camera in CAMERAS:
                with Image.open(
                    recording / "frames" / camera / f"{i:06d}.png"
                ) as image:
                    frame[f"observation.images.{camera}"] = np.asarray(image).copy()
            dataset.add_frame(frame)
            if i % 200 == 0:
                print(f"PACKAGE {i}/{n} frames", flush=True)
        dataset.save_episode(parallel_encoding=False)
    finally:
        dataset.finalize()
        dataset.stop_image_writer()
    package_seconds = perf_counter() - began
    provenance = root / "provenance"
    provenance.mkdir()
    for name in (
        "recording.json",
        "report.json",
        "record_timing.json",
        "controls_1000hz.npz",
        "samples.npz",
    ):
        shutil.copyfile(recording / name, provenance / name)
    shutil.copyfile(__file__, provenance / "collector.py")
    write_json(
        provenance / "packaging_versions.json",
        {
            "collector_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True
            ).strip(),
            "collector_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            **{
                name: importlib.metadata.version(name)
                for name in ("lerobot", "datasets", "av", "torch", "huggingface_hub")
            },
        },
    )
    card_data = DatasetCardData(
        tags=["LeRobot", "mujoco", "smoke-test"],
        task_categories=["robotics"],
        configs=[dict(config_name="default", data_files="data/*/*.parquet")],
    )
    text = f"""# Bimanual ORCA/Panda stacking — one-episode smoke test

One freshly simulated, successful native-MuJoCo episode. **Not the 200-demo randomized training set.**
The fixed red/blue layout and fixed center X are recorded in `provenance/report.json`.
No Genesis, object welds, external forces or physics changes were introduced for recording.

- {n} synchronized samples at 20 Hz; front, right-wrist and left-wrist RGB at 640×480.
- `observation.state`: 48 measured actuated joint angles in radians.
- `observation.velocity`: the same joints' velocities in radians/second.
- `action`: 48 absolute joint-position targets in radians, in the exact feature-name order.
- Order: right Panda arm (7), its native LEFT ORCA hand (17), left Panda arm (7), its native RIGHT ORCA hand (17).
- `observation.cube_poses`: optional privileged object positions (metres) and wxyz quaternions; not robot proprioception.
- `next.done` and `next.success`: true only on the terminal sample.
- Images/state precede the corresponding command. Timestamps are episode-relative seconds.

## Important control-rate limitation

{meta["action_semantics"]}
The 1000 Hz command stream is in `provenance/controls_1000hz.npz`; full sampled simulator state is in `provenance/samples.npz`.
The last sample precedes the final four physics steps. Successful collection does not validate
a learned policy, 20 Hz zero-order-held replay, randomized grasp robustness, or collision-free motion.

## Reproducibility and loading

Simulator base commit: `{meta["source_commit"]}`. MuJoCo {meta["mujoco_version"]}; LeRobot {importlib.metadata.version("lerobot")}, dataset format v3.0.
The collector source and packaging versions are included in `provenance/collector.py` and `provenance/packaging_versions.json`.
Camera rendering remains sequential; encoding is H.264. Robot/cube configuration and result are in `provenance/`.
This repository is private. Authentication with an authorized Hugging Face account is required.

```python
from lerobot.datasets.lerobot_dataset import LeRobotDataset
dataset = LeRobotDataset("{repo_id}", video_backend="pyav")
sample = dataset[0]
```

Task: {TASK}
"""
    DatasetCard("---\n" + card_data.to_yaml() + "\n---\n\n" + text).save(
        root / "README.md"
    )
    t = perf_counter()
    checks = validate(root, repo_id, arrays)
    write_json(provenance / "validation.json", checks)
    timing = dict(
        package_seconds=package_seconds,
        validation_seconds=perf_counter() - t,
        total_seconds=perf_counter() - began,
    )
    write_json(provenance / "package_timing.json", timing)
    print(json.dumps(dict(validation=checks, timings=timing), indent=2), flush=True)


def upload(root, repo_id):
    from huggingface_hub import HfApi

    root = Path(root)
    validated = perf_counter()
    checks = validate(root, repo_id)
    revalidation_seconds = perf_counter() - validated
    assert json.loads((root / "provenance/report.json").read_text())["success"]
    api = HfApi()
    # Fail rather than overwrite another run, even if it appeared after preflight.
    began = perf_counter()
    api.create_repo(repo_id, repo_type="dataset", private=True, exist_ok=False)
    assert api.repo_info(repo_id, repo_type="dataset").private
    commit = api.upload_folder(
        repo_id=repo_id,
        repo_type="dataset",
        folder_path=root,
        allow_patterns=[
            "README.md",
            "meta/**",
            "data/**",
            "videos/**",
            "provenance/**",
        ],
        commit_message="Add one validated bimanual ORCA-hand stacking smoke episode",
    )
    api.create_tag(repo_id, tag="v3.0", revision=commit.oid, repo_type="dataset")
    info = api.repo_info(repo_id, repo_type="dataset")
    assert info.private and info.sha == commit.oid
    result = dict(
        repo_id=repo_id,
        private=True,
        commit=commit.oid,
        upload_seconds=perf_counter() - began,
        revalidation_seconds=revalidation_seconds,
        local_validation=checks,
    )
    write_json(root.parent / "upload_result.json", result)
    print(json.dumps(result, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    rec = commands.add_parser("record")
    rec.add_argument("--output-dir", type=Path, required=True)
    batch = commands.add_parser("record-batch")
    batch.add_argument("--output-dir", type=Path, required=True)
    batch.add_argument("--episodes", type=int, default=3)
    batch.add_argument("--workers", type=int, default=3)
    pack = commands.add_parser("package")
    pack.add_argument("--recording-dir", type=Path, required=True)
    pack.add_argument("--dataset-dir", type=Path, required=True)
    pack.add_argument("--repo-id", default=REPO_ID)
    push = commands.add_parser("upload")
    push.add_argument("--dataset-dir", type=Path, required=True)
    push.add_argument("--repo-id", default=REPO_ID)
    args = parser.parse_args()
    if args.command == "record":
        record(args.output_dir)
    elif args.command == "record-batch":
        record_batch(args.output_dir, args.episodes, args.workers)
    elif args.command == "package":
        package(args.recording_dir, args.dataset_dir, args.repo_id)
    else:
        upload(args.dataset_dir, args.repo_id)


if __name__ == "__main__":
    main()
