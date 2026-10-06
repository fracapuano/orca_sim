#!/usr/bin/env python3
"""Compare MJCF contacts under identical increasing open-loop grip commands.

No GraspPreload call, state feedback, motor-gain adjustment or extra force cap.
The old release, previous mask-only fix and current MJCF share one command
schedule. Report actual measured forces: targets are not force commands.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np

if __package__:
    from .standalone_hand_grasp import scene_xml
    from .standalone_squeeze_stress import PROGRESSIVE_DURATION, run
else:
    from standalone_hand_grasp import scene_xml
    from standalone_squeeze_stress import PROGRESSIVE_DURATION, run

from orca_sim.versions import PACKAGE_ROOT

CASES = {
    "original_release": ("skin_disabled", "legacy"),
    "previous_contacts": ("released", "legacy"),
    "corrected_contacts": ("released", "native"),
}


def verify_mechanics(side):
    """Fail before stepping if an A/B changes anything beyond contact treatment."""
    models = [
        mujoco.MjModel.from_xml_string(scene_xml(side, variant=v, contact_model=c))
        for v, c in CASES.values()
    ]
    fields = (
        "body_mass",
        "body_inertia",
        "body_pos",
        "body_quat",
        "geom_size",
        "geom_pos",
        "geom_quat",
        "geom_friction",
        "geom_margin",
        "mesh_vert",
        "mesh_face",
        "jnt_range",
        "jnt_pos",
        "jnt_axis",
        "dof_damping",
        "dof_armature",
        "dof_frictionloss",
        "qpos0",
        "actuator_trnid",
        "actuator_gainprm",
        "actuator_biasprm",
        "actuator_ctrlrange",
        "actuator_forcerange",
        "actuator_gear",
    )
    for model in models:
        assert model.neq == 0
        assert model.opt.timestep == 0.002
        np.testing.assert_array_equal(model.opt.gravity, [0, 0, -9.81])
        for field in fields:
            np.testing.assert_array_equal(
                getattr(model, field), getattr(models[0], field), err_msg=field
            )
    return list(fields)


def summarize_trace(path):
    record = json.loads(path.read_text())
    samples = record["samples"]
    after = [s for s in samples if s["t"] >= 5]
    lost = [s for s in after if s["force_bearing_hand_contacts"] == 0]
    retained = all(
        s["force_bearing_hand_contacts"] >= 2
        and "t" in s["digits"]
        and len(s["digits"]) >= 2
        and not s["support_contacts"]
        for s in after
    )
    checkpoints = []
    for t in (5, 10, 15, 20, 25, 30, 40):
        s = min(samples, key=lambda s: abs(s["t"] - t))
        checkpoints.append(
            {
                k: s[k]
                for k in (
                    "t",
                    "cube_pos",
                    "sum_hand_normal_force_N",
                    "saturated_motor_count",
                    "translation_from_pre_squeeze_m",
                    "rotation_from_pre_squeeze_deg",
                    "skin_vertex_depth_m",
                )
            }
        )
    return dict(
        record["result"],
        checkpoints=checkpoints,
        continuous_opposing_contact_after_5s=retained,
        first_zero_force_contact_sample_s=lost[0]["t"] if lost else None,
    )


def force_plot(output, side):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True, layout="constrained")
    colors = ("#777777", "#cf4c42", "#218a60")
    for (case, (variant, _)), color in zip(CASES.items(), colors):
        trace = json.loads(
            (output / side / case / f"{side}_{variant}_progressive_1.json").read_text()
        )["samples"]
        t = [s["t"] for s in trace]
        axes[0].plot(
            t,
            [s["sum_hand_normal_force_N"] for s in trace],
            label=case.replace("_", " "),
            color=color,
        )
        axes[1].plot(t, [s["skin_vertex_depth_m"] * 1000 for s in trace], color=color)
    for ax in axes:
        for t in (5, 15, 25):
            ax.axvline(t, color="black", alpha=0.25, linestyle="--")
        ax.grid(alpha=0.15)
        ax.set_xlim(0, 40)
    axes[0].legend(loc="upper left")
    axes[0].set_ylabel("Summed contact normal force (N)")
    axes[0].set_title(
        f"{side.title()} hand: identical increasing commands; native +/-1 Nm per motor"
    )
    axes[1].set_ylabel("Skin vertex penetration (mm)")
    axes[1].set_xlabel(
        "Time (s): grasp 0–5 | ramp to strong pose 5–15 | ramp to joint limits 15–25 | hold 25–40"
    )
    fig.savefig(output / f"{side}_force_and_penetration.png", dpi=160)
    plt.close(fig)


def compare_videos(output, side):
    import imageio.v2 as imageio

    for first in ("previous_contacts", "original_release"):
        variant = CASES[first][0]
        with (
            imageio.get_reader(
                output / side / first / f"{side}_{variant}_progressive_1.mp4"
            ) as a,
            imageio.get_reader(
                output
                / side
                / "corrected_contacts"
                / f"{side}_released_progressive_1.mp4"
            ) as b,
            imageio.get_writer(
                output / f"{side}_{first}_vs_corrected.mp4",
                fps=30,
                codec="libx264",
                quality=8,
            ) as writer,
        ):
            for left, right in zip(a, b):
                writer.append_data(np.concatenate([left, right], axis=1))


def comparison(output, video=False, sides=("right", "left")):
    output.mkdir(parents=True, exist_ok=True)
    results = []
    for side in sides:
        fields = verify_mechanics(side)
        side_results = []
        for case, (variant, contact) in CASES.items():
            directory = output / side / case
            run(
                side=side,
                variant=variant,
                contact_model=contact,
                profile="progressive",
                duration=PROGRESSIVE_DURATION,
                output_dir=directory,
                video=video,
            )
            r = summarize_trace(directory / f"{side}_{variant}_progressive_1.json")
            r["case"] = case
            side_results.append(r)
            print(
                f"{side}/{case}: held={r['retained_at_end']}, skin={r['max_skin_vertex_depth_m'] * 1000:.3f}mm, contact load peak={r['peak_total_normal_N']:.1f}N",
                flush=True,
            )
        assert len({r["command_trace_sha256"] for r in side_results}) == 1, (
            "A/B command traces differ"
        )
        results.extend(side_results)
        if video:
            compare_videos(output, side)
        force_plot(output, side)
    files = [
        Path(__file__),
        Path(__file__).with_name("standalone_squeeze_stress.py"),
        Path(__file__).with_name("standalone_hand_grasp.py"),
    ]
    files += [
        PACKAGE_ROOT / "models" / "v2" / "mjcf" / f"orcahand_{side}{suffix}"
        for side in sides
        for suffix in (".mjcf", "_body.xml")
    ]
    report = {
        "command_schedule": "0–5 s acquire; 5–15 s nominal to previous strong pose; 15–25 s strong pose to all ten MCP/PIP actuator upper limits; 25–40 s hold. Wrist, opposition and abduction unchanged.",
        "controller": "Native position servos only; kp=2, force limits +/-1 Nm. No additional cap, GraspPreload, state feedback, object constraint, or external holding force.",
        "identical_commands_within_each_side": True,
        "identical_compiled_mechanics_fields": fields,
        "source_sha256": {
            str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files
        },
        "measurement": "Contact penetration, torque and finite state checked every 2 ms; skin vertices and force/pose traces sampled every 20 ms. Vertex test is a lower bound. Force is sum of normal contact loads, not net force. Increasing target demand is not guaranteed to increase force monotonically.",
        "results": results,
    }
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", action="store_true")
    parser.add_argument("--side", choices=("left", "right", "both"), default="both")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("renders/standalone_hand/progressive_grip"),
    )
    args = parser.parse_args()
    comparison(
        args.output_dir,
        args.video,
        ("right", "left") if args.side == "both" else (args.side,),
    )


if __name__ == "__main__":
    main()
