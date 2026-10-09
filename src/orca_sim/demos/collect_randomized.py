"""Bounded, successful-only randomized collection; one coordinator owns publication.

Three simulator processes use disjoint roots. One packaging process creates
validated one-episode shards. Official LeRobot aggregation assembles the dataset.
Only cube XY varies; the environment owns the sampling distribution.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np

from orca_sim.demos.collect_lerobot_smoke import CAMERAS, FPS, REPO_ID, TASK, write_json


MODULE = "orca_sim.demos.collect_randomized"
SMOKE_MODULE = "orca_sim.demos.collect_lerobot_smoke"
REGIONS = (
    Path(__file__).resolve().parents[1]
    / "assets/mujoco/bimanual_cube_spawn_regions.json"
)


def read_json(path):
    return json.loads(Path(path).read_text())


def save_status(root, state):
    temporary = root / "status.tmp"
    write_json(temporary, state)
    temporary.replace(root / "status.json")


def check_layout(meta, regions):
    """Reject mislabeled fixed-layout episodes or any non-XY randomization."""
    if meta["fixed_layout"] or meta["seed"] is None:
        raise ValueError(
            "Randomized collection requires an explicit seed and distribution"
        )
    assert meta["cube_spawn_regions"] == regions
    layout = meta["object_layout"]
    np.testing.assert_array_equal(layout["stack_target"], [0.5, 0.201, 0.326])
    xy = []
    for name in ("red_cube", "blue_cube"):
        assert layout[name][2] == 0.353
        for index, axis in enumerate(("x", "y")):
            value = layout[name][index]
            low, high = regions[name][axis]
            assert low <= value <= high
            xy.append(value)
    return tuple(xy)


def remove_staging_frames(root, attempt):
    """Only this run's generated PNGs; never models, reports, or encoded videos."""
    parent = (root / f"attempt_{attempt:06d}").resolve()
    frames = parent / "recording/frames"
    if parent.parent != root.resolve() or frames.is_symlink():
        raise ValueError("Unsafe staging path")
    if frames.exists():
        if frames.resolve().parent != parent / "recording":
            raise ValueError("Unsafe staging path")
        shutil.rmtree(frames)


def launch(command, log_path):
    log = log_path.open("w")
    try:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
    except BaseException:
        log.close()
        raise
    return process, log, time.monotonic()


def collect(
    root, *, episodes, workers, seed, max_attempts, package_python, push, hub_parent
):
    if not (1 <= workers <= 3 and 1 <= episodes <= max_attempts and seed >= 0):
        raise ValueError(
            "Require 1–3 workers, positive episodes <= max_attempts, nonnegative seed"
        )
    if push and not hub_parent:
        raise ValueError("Publication requires the expected current Hub commit")
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=False)
    regions = read_json(REGIONS)
    write_json(root / "spawn_regions.json", regions)
    state = dict(
        status="recording",
        target_episodes=episodes,
        workers=workers,
        base_seed=seed,
        max_attempts=max_attempts,
        repo_id=REPO_ID,
        private=True,
        attempts=[],
        successful_attempts=[],
        randomization="Only independent cube XY within the shared 10x10 cm regions",
        spawn_regions=regions,
        started_unix=time.time(),
    )
    active, waiting, packer = {}, [], None
    seen, failures, next_status = set(), 0, 0
    save_status(root, state)
    try:
        while len(state["successful_attempts"]) < episodes:
            reserved = (
                len(state["successful_attempts"])
                + len(active)
                + len(waiting)
                + bool(packer)
            )
            while (
                len(active) < workers
                and reserved < episodes
                and len(active) + len(waiting) + bool(packer) < 2 * workers
                and len(state["attempts"]) < max_attempts
            ):
                if shutil.disk_usage(root).free < 12 * 1024**3:
                    raise RuntimeError(
                        "Less than 12 GiB free; stopped safely before disk exhaustion"
                    )
                index = len(state["attempts"])
                attempt = dict(attempt=index, seed=seed + index, status="recording")
                state["attempts"].append(attempt)
                folder = root / f"attempt_{index:06d}"
                folder.mkdir()
                command = [
                    sys.executable,
                    "-m",
                    SMOKE_MODULE,
                    "record",
                    "--output-dir",
                    str(folder / "recording"),
                    "--cube-spawn-regions",
                    str(root / "spawn_regions.json"),
                    "--seed",
                    str(attempt["seed"]),
                ]
                active[index] = launch(command, folder / "record.log")
                attempt["pid"] = active[index][0].pid
                reserved += 1
                print(
                    f"START attempt={index} seed={attempt['seed']} pid={attempt['pid']}",
                    flush=True,
                )
                save_status(root, state)
            for index, (process, log, started) in list(active.items()):
                if time.monotonic() - started > 2400:
                    raise RuntimeError(
                        f"Recording {index} exceeded 40 minutes; inspect its log"
                    )
                code = process.poll()
                if code is None:
                    continue
                log.close()
                del active[index]
                attempt = state["attempts"][index]
                folder = root / f"attempt_{index:06d}"
                report_path = folder / "recording/report.json"
                report = read_json(report_path) if report_path.exists() else {}
                attempt.update(
                    returncode=code,
                    recording_wall_s=time.monotonic() - started,
                    error=report.get("error"),
                    outcome=report,
                )
                if code != 0 or not report.get("success"):
                    attempt["status"] = "failed"
                    failures += 1
                    # Physical task failures retain reset/report/trace/log evidence.
                    # Unexpected process failures stop instead of burning more episodes.
                    if not report:
                        raise RuntimeError(
                            f"Recording {index} crashed without a task report"
                        )
                    remove_staging_frames(root, index)
                    print(
                        f"FAILED attempt={index}: {report.get('error') or 'unstable stack'}",
                        flush=True,
                    )
                    if failures >= 9:
                        raise RuntimeError(
                            "Nine consecutive task failures; controller needs investigation"
                        )
                else:
                    meta = read_json(folder / "recording/recording.json")
                    xy = check_layout(meta, regions)
                    if xy in seen:
                        raise RuntimeError(
                            "Duplicate cube layout detected; refusing duplicate data"
                        )
                    seen.add(xy)
                    attempt.update(status="awaiting_package", initial_cube_xy=list(xy))
                    waiting.append(index)
                    failures = 0
                save_status(root, state)
            if packer is not None:
                index, process, log, started = packer
                if time.monotonic() - started > 1800:
                    raise RuntimeError(f"Packaging {index} exceeded 30 minutes")
                code = process.poll()
                if code is not None:
                    log.close()
                    packer = None
                    if code:
                        raise RuntimeError(
                            f"Packaging {index} failed; recording retained"
                        )
                    shard = root / f"attempt_{index:06d}/dataset"
                    validation = read_json(shard / "provenance/validation.json")
                    assert validation["numeric_alignment_verified"]
                    assert validation["lerobot_item_decode_verified"]
                    state["attempts"][index].update(
                        status="validated", packaging_wall_s=time.monotonic() - started
                    )
                    state["successful_attempts"].append(index)
                    remove_staging_frames(root, index)
                    print(
                        f"ACCEPTED {len(state['successful_attempts'])}/{episodes} attempt={index}",
                        flush=True,
                    )
                    save_status(root, state)
            if packer is None and waiting:
                index = waiting.pop(0)
                folder = root / f"attempt_{index:06d}"
                command = [
                    package_python,
                    "-m",
                    SMOKE_MODULE,
                    "package",
                    "--recording-dir",
                    str(folder / "recording"),
                    "--dataset-dir",
                    str(folder / "dataset"),
                    "--repo-id",
                    REPO_ID,
                ]
                process, log, started = launch(command, folder / "package.log")
                packer = index, process, log, started
                state["attempts"][index]["status"] = "packaging"
                save_status(root, state)
            if (
                not active
                and not waiting
                and packer is None
                and len(state["successful_attempts"]) < episodes
            ):
                if len(state["attempts"]) >= max_attempts:
                    raise RuntimeError(
                        "Attempt budget exhausted before enough successes"
                    )
            if time.monotonic() >= next_status:
                state["elapsed_seconds"] = time.time() - state["started_unix"]
                save_status(root, state)
                print(
                    f"PROGRESS accepted={len(state['successful_attempts'])}/{episodes} "
                    f"recording={list(active)} packaging={packer[0] if packer else None} "
                    f"queued={waiting} attempts={len(state['attempts'])}",
                    flush=True,
                )
                next_status = time.monotonic() + 30
            time.sleep(0.5)
        state["successful_attempts"].sort()
        state["status"] = "assembling"
        save_status(root, state)
        command = [package_python, "-m", MODULE, "assemble", "--root", str(root)]
        if push:
            command += ["--push", "--hub-parent", hub_parent]
        with (root / "assembly.log").open("w") as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
        state["status"] = "uploaded" if push else "complete_local"
        state["finished_unix"] = time.time()
        save_status(root, state)
    except BaseException as exc:
        state.update(status="stopped", error=repr(exc))
        save_status(root, state)
        raise
    finally:
        children = list(active.values())
        if packer:
            children.append(packer[1:])
        for process, log, _ in children:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            log.close()


def assemble(root, *, push=False, hub_parent=None):
    from huggingface_hub import DatasetCard, DatasetCardData, HfApi
    from lerobot.datasets.aggregate import aggregate_datasets
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    root = Path(root).resolve()
    state = read_json(root / "status.json")
    indices = sorted(state["successful_attempts"])
    assert len(indices) == state["target_episodes"] and len(set(indices)) == len(
        indices
    )
    sources = [root / f"attempt_{index:06d}/dataset" for index in indices]
    layouts, seeds, records = set(), set(), []
    for episode, (index, source) in enumerate(zip(indices, sources)):
        meta = read_json(source / "provenance/recording.json")
        xy = check_layout(meta, state["spawn_regions"])
        if xy in layouts or meta["seed"] in seeds:
            raise ValueError("Repeated seed or initial cube layout")
        layouts.add(xy)
        seeds.add(meta["seed"])
        assert read_json(source / "provenance/report.json")["success"]
        records.append(
            dict(
                episode_index=episode,
                attempt=index,
                seed=meta["seed"],
                object_layout=meta["object_layout"],
                frames=meta["frames"],
            )
        )
    destination = root / "dataset"
    # One small video per episode avoids repeated concatenation and video offsets.
    aggregate_datasets(
        [REPO_ID] * len(sources),
        REPO_ID,
        roots=sources,
        aggr_root=destination,
        video_files_size_in_mb=0.001,
    )
    dataset = LeRobotDataset(REPO_ID, root=destination, video_backend="pyav")
    assert dataset.num_episodes == len(sources) and dataset.fps == FPS
    offset = 0
    for episode, (source, record) in enumerate(zip(sources, records)):
        n = record["frames"]
        table = dataset.hf_dataset.select(range(offset, offset + n)).with_format(
            "numpy"
        )
        np.testing.assert_array_equal(table["episode_index"], np.full(n, episode))
        np.testing.assert_allclose(table["timestamp"], np.arange(n) / FPS, atol=1e-5)
        with np.load(source / "provenance/samples.npz") as arrays:
            for key, value in (
                ("observation.state", "state"),
                ("observation.velocity", "velocity"),
                ("action", "action"),
                ("observation.cube_poses", "cube_poses"),
            ):
                np.testing.assert_array_equal(
                    table[key], arrays[value].astype(np.float32)
                )
            np.testing.assert_allclose(
                arrays["cube_poses"][0, :3], record["object_layout"]["red_cube"]
            )
            np.testing.assert_allclose(
                arrays["cube_poses"][0, 7:10], record["object_layout"]["blue_cube"]
            )
        for flag in ("next.done", "next.success"):
            values = np.asarray(table[flag]).reshape(-1)
            assert values[-1] and not values[:-1].any()
        # Aggregation copies the already fully decoded/validated videos unchanged.
        original = LeRobotDataset(REPO_ID, root=source, video_backend="pyav")
        for frame in (0, n // 2, n - 1):
            before, after = original[frame], dataset[offset + frame]
            for camera in CAMERAS:
                key = f"observation.images.{camera}"
                assert (before[key] == after[key]).all()
        offset += n
    assert dataset.num_frames == offset
    provenance = destination / "provenance"
    provenance.mkdir()
    for episode, source in enumerate(sources):
        shutil.copytree(source / "provenance", provenance / f"episode_{episode:06d}")
    write_json(provenance / "episodes.json", records)
    write_json(provenance / "collection.json", state)
    shutil.copyfile(root / "spawn_regions.json", provenance / "spawn_regions.json")
    shutil.copyfile(__file__, provenance / "collection_coordinator.py")
    checks = dict(
        episodes=len(indices),
        frames=offset,
        unique_layouts=len(layouts),
        unique_seeds=len(seeds),
        all_numeric_samples_verified=True,
        every_episode_camera_boundary_decode_verified=True,
    )
    write_json(provenance / "validation.json", checks)
    first = read_json(sources[0] / "provenance/recording.json")
    text = f"""# Bimanual ORCA/Panda cube stacking — randomized scripted demonstrations

{len(indices)} successful native-MuJoCo demonstrations, collected with three independent recording workers.
Only initial cube XY changes. Red: x=[0.25, 0.35], y=[0.151, 0.251] m;
blue: x=[0.65, 0.75], y=[0.151, 0.251] m. These are two 10×10 cm regions.
Robot home poses, hand contacts, camera settings, cube size/orientation and center X are fixed.
The X remains at [0.5, 0.201, 0.326] m. Cubes are 5.6 cm wide.

Task: {TASK}

Each attempt has a distinct seed. `provenance/episodes.json` maps episode indices to
seeds and measured initial layouts. `provenance/collection.json` retains outcomes of
all attempts, including failures. **Only successful attempts are included**, so the
accepted distribution is success-conditioned, not guaranteed uniform over the regions.
The environment's shared reset distribution is in `provenance/spawn_regions.json`.

Three synchronized RGB cameras: front, right_wrist, left_wrist, 640×480 at {FPS} Hz.
`observation.state` contains 48 measured joint angles; `observation.velocity` contains
the same joint velocities. `action` contains 48 absolute joint-position targets in radians.
Feature names give the order: right Panda (7) + native LEFT ORCA hand (17), then left
Panda (7) + native RIGHT ORCA hand (17). `observation.cube_poses` is privileged object
state, not robot proprioception. `next.done` and `next.success` are true only at the
terminal sample. Each episode has its own relative timestamps.

## Control-rate limitation

{first["action_semantics"]}
Images/state precede the sampled command. Each episode's full-rate commands and sampled
simulator state are retained in its provenance folder. The terminal sample precedes
the last few physics steps. A successful stack is not a collision-free-motion certificate.
No welds, cube teleportation after reset, external forces or physics changes are used.

## Loading

```python
from lerobot.datasets.lerobot_dataset import LeRobotDataset
dataset = LeRobotDataset("{REPO_ID}", video_backend="pyav")
```

Private dataset: authenticate with an authorized Hugging Face account. LeRobot v3 format.
"""
    card = DatasetCardData(
        tags=["LeRobot", "mujoco", "scripted", "randomized-cube-xy"],
        task_categories=["robotics"],
        configs=[dict(config_name="default", data_files="data/*/*.parquet")],
    )
    DatasetCard("---\n" + card.to_yaml() + "\n---\n\n" + text).save(
        destination / "README.md"
    )
    if push:
        api = HfApi()
        info = api.repo_info(REPO_ID, repo_type="dataset")
        if not info.private or info.sha != hub_parent:
            raise RuntimeError(
                "Hub privacy or parent commit changed; local dataset retained, no overwrite"
            )
        # Preserve the previous smoke dataset before replacing its curated paths.
        api.create_tag(
            REPO_ID,
            repo_type="dataset",
            tag="smoke-before-randomized-200",
            revision=hub_parent,
            exist_ok=True,
        )
        result = api.upload_folder(
            repo_id=REPO_ID,
            repo_type="dataset",
            folder_path=destination,
            allow_patterns=[
                "README.md",
                "meta/**",
                "data/**",
                "videos/**",
                "provenance/**",
            ],
            delete_patterns=["meta/**", "data/**", "videos/**", "provenance/**"],
            parent_commit=hub_parent,
            commit_message=f"Add {len(indices)} validated randomized ORCA/Panda stacking demonstrations",
        )
        api.delete_tag(REPO_ID, repo_type="dataset", tag="v3.0")
        api.create_tag(REPO_ID, repo_type="dataset", tag="v3.0", revision=result.oid)
        updated = api.repo_info(REPO_ID, repo_type="dataset")
        assert updated.private and updated.sha == result.oid
        # Check the remote metadata at the exact published commit.
        from huggingface_hub import hf_hub_download

        remote = hf_hub_download(
            REPO_ID, "meta/info.json", repo_type="dataset", revision=result.oid
        )
        assert read_json(remote)["total_episodes"] == len(indices)
        write_json(
            root / "upload_result.json",
            dict(
                repo_id=REPO_ID,
                commit=result.oid,
                private=True,
                episodes=len(indices),
                validation=checks,
            ),
        )
    print(json.dumps(checks, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run")
    run.add_argument("--root", type=Path, required=True)
    run.add_argument("--episodes", type=int, default=200)
    run.add_argument("--workers", type=int, default=3)
    run.add_argument("--seed", type=int, default=202610090)
    run.add_argument("--max-attempts", type=int, default=600)
    run.add_argument("--package-python", required=True)
    run.add_argument("--push", action="store_true")
    run.add_argument("--hub-parent")
    merge = sub.add_parser("assemble")
    merge.add_argument("--root", type=Path, required=True)
    merge.add_argument("--push", action="store_true")
    merge.add_argument("--hub-parent")
    args = vars(parser.parse_args())
    command = args.pop("command")
    (collect if command == "run" else assemble)(**args)


if __name__ == "__main__":
    main()
