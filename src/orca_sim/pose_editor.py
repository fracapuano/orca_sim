"""Local, live MuJoCo starting-pose sliders; no simulation runs while editing.

Run: python -m orca_sim.pose_editor
Only loopback is served. The server exposes no arbitrary file or shell access.
"""

from __future__ import annotations

import argparse
import copy
import base64
import io
import json
import secrets
import tempfile
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import mujoco
import numpy as np

from orca_sim.mujoco_panda import (
    BASE_XY_LIMITS,
    CUBE_REST_Z,
    HOME_POSE,
    TABLETOP_LAYOUT,
    TABLE_HEIGHT,
    OBJECT_XY_LIMITS,
    PACKAGE,
    OrcaPandaCubeStackingMujoco,
    base_quaternion,
    start_configuration,
    tabletop_configuration,
)


class PoseEditor:
    def __init__(self, save_path=None, mode="layout"):
        if mode not in ("layout", "hand", "objects"):
            raise ValueError("Unknown editor mode")
        self.mode = mode
        self.save_path = Path(
            save_path or (TABLETOP_LAYOUT if mode == "objects" else HOME_POSE)
        ).resolve()
        self.env = OrcaPandaCubeStackingMujoco(
            start_pose=self.save_path
            if mode != "objects" and self.save_path.exists()
            else None,
            object_layout=self.save_path
            if mode == "objects" and self.save_path.exists()
            else None,
        )
        self.config = self.env.start_pose
        self.locked_arm = list(self.config["arm_qpos"])
        self.locked_base = list(self.config["base_pos"])
        self.locked_yaw = self.config["base_yaw"]
        self.locked_hand = dict(self.config["hand_qpos"])
        self.saved_base = list(self.config["base_pos"])
        self.saved_yaw = self.config["base_yaw"]
        self.saved_hand = dict(self.config["hand_qpos"])
        self.object_layout = copy.deepcopy(self.env.object_layout)
        self.saved_objects = self.objects_in_robot_frame()
        self.spec = mujoco.MjSpec.from_string(self.env.xml)
        self.renderer = None

    def objects_in_robot_frame(self):
        yaw = self.config["base_yaw"]
        c, s = np.cos(yaw), np.sin(yaw)
        rotation = np.array([[c, -s], [s, c]])
        base = np.array(self.config["base_pos"][:2])
        return {
            name: (
                rotation.T @ (np.array(self.object_layout[name][:2]) - base)
            ).tolist()
            for name in ("red_cube", "blue_cube")
        }

    def object_payload(self, payload):
        positions = payload.get("objects_robot")
        if not isinstance(positions, dict) or set(positions) != {
            "red_cube",
            "blue_cube",
        }:
            raise ValueError("Expected robot-relative positions for both cubes")
        yaw = self.locked_yaw
        c, s = np.cos(yaw), np.sin(yaw)
        rotation = np.array([[c, -s], [s, c]])
        result = {}
        for name, coords in positions.items():
            coords = np.asarray(coords, dtype=float)
            if coords.shape != (2,) or not np.isfinite(coords).all():
                raise ValueError(f"Expected finite forward/left coordinates for {name}")
            xy = np.array(self.locked_base[:2]) + rotation @ coords
            result[name] = [*xy.tolist(), CUBE_REST_Z]
        result["stack_target"] = [*result["blue_cube"][:2], TABLE_HEIGHT + 0.001]
        return tabletop_configuration(result)

    def state(self):
        m, d = self.env.model, self.env.data
        collisions = []
        for c in d.contact:
            names = [m.body(m.geom_bodyid[g]).name for g in (c.geom1, c.geom2)]
            robot_hit = c.dist < -0.001 and any(
                n.startswith(("link", "right_")) for n in names
            )
            cube_hit = (
                self.mode == "objects"
                and c.dist < -1e-6
                and set(names) == {"red_cube", "blue_cube"}
            )
            if robot_hit or cube_hit:
                collisions.append({"bodies": names, "depth_mm": -float(c.dist) * 1000})
        return {
            **self.config,
            "hand_joint_names": list(self.env.actuator_joints[7:]),
            "hand_joint_limits": dict(
                zip(
                    self.env.actuator_joints[7:],
                    np.c_[
                        self.env.action_space.low[7:], self.env.action_space.high[7:]
                    ].tolist(),
                )
            ),
            "saved_hand": dict(self.saved_hand),
            "saved_base_pos": list(self.saved_base),
            "saved_base_yaw": self.saved_yaw,
            "base_xy_limits": BASE_XY_LIMITS,
            "editor_mode": self.mode,
            "objects_robot": self.objects_in_robot_frame(),
            "saved_objects_robot": copy.deepcopy(self.saved_objects),
            "object_xy_world_limits": OBJECT_XY_LIMITS,
            "arm_home_locked": True,
            "hand_home_locked": self.mode != "hand",
            "hand_position": d.site("hand_frame").xpos.tolist(),
            "collisions": collisions,
            "save_path": str(self.save_path),
        }

    def update(self, payload):
        config = start_configuration(payload)
        if config["arm_qpos"] != self.locked_arm:
            raise ValueError("Arm joint angles are locked to your chosen home")
        if self.mode != "hand" and config["hand_qpos"] != self.locked_hand:
            raise ValueError("Hand joint angles are locked while editing the layout")
        if self.mode != "layout" and (
            config["base_pos"] != self.locked_base
            or config["base_yaw"] != self.locked_yaw
        ):
            raise ValueError("Base position and yaw are locked in this editor")
        objects = self.object_payload(payload) if self.mode == "objects" else None
        e, m, d = self.env, self.env.model, self.env.data
        hand = np.asarray([config["hand_qpos"][name] for name in e.actuator_joints[7:]])
        if np.any(hand < e.action_space.low[7:]) or np.any(
            hand > e.action_space.high[7:]
        ):
            raise ValueError("Joint angle exceeds its native range")
        camera = payload.get(
            "camera", [135.0, -22.0, 0.55 if self.mode == "hand" else 1.85]
        )
        if np.asarray(camera).shape != (3,) or not np.isfinite(camera).all():
            raise ValueError("Invalid camera")
        azimuth, elevation, distance = map(float, camera)
        if not (-90 <= elevation <= 90 and 0.18 <= distance <= 4):
            raise ValueError("Camera is out of range")
        focus = payload.get("focus", "hand" if self.mode == "hand" else "scene")
        if focus not in ("hand", "scene"):
            raise ValueError("Unknown camera focus")
        if (
            config["base_pos"] != self.config["base_pos"]
            or config["base_yaw"] != self.config["base_yaw"]
        ):
            # link0 is static: editing mjModel.body_pos/quat in place leaves its
            # collision BVH stale. Recompile the MJCF spec, including collisions.
            # MjSpec caches meshes; camera/hand-only changes need no compilation.
            base = self.spec.body("link0")
            base.pos = config["base_pos"]
            base.quat = base_quaternion(config["base_yaw"])
            model = self.spec.compile()
            data = mujoco.MjData(model)
            if self.renderer is not None:
                self.renderer.close()
                self.renderer = None
            e.model, e.data = m, d = model, data
            e.xml = self.spec.to_xml()
        e.reset()
        d.qpos[e.qadr[7:]] = hand
        d.ctrl[7:] = hand
        if objects is not None:
            # Paused layout editing only: freejoint coordinates and mocap state
            # move the preview objects. Rollouts still use ordinary free bodies.
            for name in ("red_cube", "blue_cube"):
                address = int(m.joint(name + "_freejoint").qposadr[0])
                d.qpos[address : address + 3] = objects[name]
            target = m.body("stack_target").mocapid[0]
            d.mocap_pos[target] = objects["stack_target"]
            self.object_layout = objects
        mujoco.mj_forward(m, d)
        if focus == "hand":
            # Center on finger-joint positions, not the tower/forearm mount.
            finger_bodies = m.jnt_bodyid[m.actuator_trnid[8:, 0]]
            e.camera.lookat[:] = np.mean(d.xpos[finger_bodies], axis=0)
        else:
            e.camera.lookat[:] = [0.04, 0, 0.48]
        e.camera.azimuth, e.camera.elevation, e.camera.distance = (
            azimuth,
            elevation,
            distance,
        )
        self.config = config
        return self.state()

    def preview(self, payload):
        state = self.update(payload)
        if self.renderer is None:
            self.renderer = mujoco.Renderer(self.env.model, height=540, width=960)
        self.renderer.update_scene(self.env.data, camera=self.env.camera)
        from PIL import Image

        encoded = io.BytesIO()
        Image.fromarray(self.renderer.render()).save(encoded, format="JPEG", quality=87)
        state["image"] = (
            "data:image/jpeg;base64," + base64.b64encode(encoded.getvalue()).decode()
        )
        return state

    def save(self, payload):
        state = self.update(payload)
        if state["collisions"]:
            raise ValueError(
                "Pose has intersecting contacts; adjust the sliders before saving"
            )
        self.save_path.parent.mkdir(parents=True, exist_ok=True)
        if self.save_path.exists():
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
            backup = self.save_path.with_name(
                self.save_path.stem + "." + stamp + ".bak"
            )
            backup.write_bytes(self.save_path.read_bytes())
        with tempfile.NamedTemporaryFile(
            mode="w",
            dir=self.save_path.parent,
            prefix=".pose-",
            suffix=".tmp",
            delete=False,
        ) as f:
            f.write(
                json.dumps(
                    self.object_layout if self.mode == "objects" else self.config,
                    indent=2,
                )
                + "\n"
            )
            temporary = Path(f.name)
        temporary.replace(self.save_path)
        self.saved_hand = dict(self.config["hand_qpos"])
        self.saved_base = list(self.config["base_pos"])
        self.saved_yaw = self.config["base_yaw"]
        self.saved_objects = self.objects_in_robot_frame()
        return {"saved": str(self.save_path), **self.state()}

    def close(self):
        if self.renderer is not None:
            self.renderer.close()
        self.env.close()


def make_handler(editor, token):
    class Handler(BaseHTTPRequestHandler):
        def reply(self, status, content, kind="application/json"):
            body = (
                content if isinstance(content, bytes) else json.dumps(content).encode()
            )
            self.send_response(status)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(body)

        def allowed(self):
            host = f"127.0.0.1:{self.server.server_port}"
            return (
                self.headers.get("Host") == host
                and self.headers.get("Origin", f"http://{host}") == f"http://{host}"
            )

        def do_GET(self):
            if not self.allowed():
                self.reply(403, {"error": "Loopback origin required"})
            elif self.path == "/":
                html = (PACKAGE / "assets/mujoco/pose_editor.html").read_text()
                html = html.replace(
                    "__INITIAL__", json.dumps(editor.state()).replace("<", "\\u003c")
                ).replace("__TOKEN__", json.dumps(token))
                self.reply(200, html.encode(), "text/html; charset=utf-8")
            else:
                self.reply(404, {"error": "Not found"})

        def do_POST(self):
            if not self.allowed() or not secrets.compare_digest(
                self.headers.get("X-Pose-Token", ""), token
            ):
                self.reply(403, {"error": "Invalid local request"})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length < 16384:
                    raise ValueError("Invalid request size")
                payload = json.loads(self.rfile.read(length))
                if self.path == "/preview":
                    result = editor.preview(payload)
                elif self.path == "/save":
                    result = editor.save(payload)
                else:
                    self.reply(404, {"error": "Not found"})
                    return
                self.reply(200, result)
            except (ValueError, TypeError, KeyError) as exc:
                self.reply(400, {"error": str(exc)})

        def log_message(self, *args):
            pass

    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--save-path",
        type=Path,
        help="Output JSON (default: object layout in objects mode, robot home otherwise)",
    )
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument(
        "--mode", choices=("layout", "hand", "objects"), default="layout"
    )
    args = parser.parse_args()
    editor = PoseEditor(args.save_path, mode=args.mode)
    server = HTTPServer(
        ("127.0.0.1", args.port), make_handler(editor, secrets.token_urlsafe(32))
    )
    print(f"Pose editor: http://127.0.0.1:{server.server_port}/", flush=True)
    print(f"Save configuration: {editor.save_path}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        editor.close()


if __name__ == "__main__":
    main()
