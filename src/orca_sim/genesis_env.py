"""Genesis environment for the bimanual ORCA-Panda cube task."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, ClassVar

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from orca_sim.genesis_assets import build_genesis_bimanual_orcapanda_mjcf

RIGHT_WRIST_LINK = "orcahand_right_ForeArmStructure-Model_e18f2368"
WORKBENCH_MJCF_PATH = (
    Path(__file__).resolve().parent / "assets" / "genesis" / "workbench.xml"
)
CUBE_MJCF_PATH = (
    Path(__file__).resolve().parent / "assets" / "genesis" / "grasp_cube.xml"
)
N_ARM_DOFS = 7
N_HAND_DOFS = 17
N_ROBOT_DOFS = 2 * (N_ARM_DOFS + N_HAND_DOFS)
LEFT_ARM_DOFS = np.arange(0, 7)
LEFT_HAND_DOFS = np.arange(7, 24)
RIGHT_ARM_DOFS = np.arange(24, 31)
RIGHT_HAND_DOFS = np.arange(31, 48)

TABLE_SIZE = np.array([0.75, 1.15, 0.05], dtype=np.float64)
TABLE_POSITION = np.array([0.42, 0.0, 0.275], dtype=np.float64)
TABLE_TOP_Z = float(TABLE_POSITION[2] + TABLE_SIZE[2] / 2)
CUBE_SIZE = 0.05
CUBE_CONTACT_SKIN = 0.0005
CUBE_POSITION = np.array([0.50, 0.15, TABLE_TOP_Z + CUBE_SIZE / 2], dtype=np.float64)
GRASP_WRIST_XY = np.array([0.42, 0.28], dtype=np.float64)
APPROACH_WRIST_POSITION = np.array([0.43, 0.36, 0.70], dtype=np.float64)

CAMERA_RESOLUTION = (1280, 720)
CAMERA_POSITION = (1.80, -0.72, 1.28)
CAMERA_LOOKAT = (0.31, 0.0, 0.55)
CAMERA_FOV = 42

HOME_ARM_QPOS = np.array([0.0, -0.6, 0.0, -2.1, 0.0, 1.55, 0.8], dtype=np.float64)
HOME_QPOS = np.concatenate(
    [HOME_ARM_QPOS, np.zeros(N_HAND_DOFS), HOME_ARM_QPOS, np.zeros(N_HAND_DOFS)]
)

# Symmetric elbow-up posture with both wrists above and facing into the shared
# workspace.  It was selected from IK solutions for wrist targets at
# (0.45, +/-0.30, 0.55) by preferring joint-limit margin and a well-conditioned
# translational Jacobian.  The right-arm relation is the Panda mirror mapping
# for the two opposing base mounts.
READY_LEFT_ARM_QPOS = np.array(
    [
        -1.02438856,
        -0.01926053,
        0.30822373,
        -2.03127841,
        1.33388590,
        1.01791308,
        0.35839412,
    ],
    dtype=np.float64,
)
READY_RIGHT_ARM_QPOS = np.array(
    [
        1.02438856,
        -0.01926053,
        -0.30822373,
        -2.03127841,
        -1.33388590,
        1.01791308,
        1.24160588,
    ],
    dtype=np.float64,
)
READY_QPOS = np.concatenate(
    [
        READY_LEFT_ARM_QPOS,
        np.zeros(N_HAND_DOFS),
        READY_RIGHT_ARM_QPOS,
        np.zeros(N_HAND_DOFS),
    ]
)

RIGHT_HAND_CLOSED = np.zeros(N_HAND_DOFS, dtype=np.float64)
RIGHT_HAND_CLOSED[[2, 3, 5, 6, 8, 9, 11, 12]] = 1.45
# Positive thumb opposition brings the distal pad across from the index finger.
# The previous negative value placed both contact surfaces on the same side of
# the cube, so closing the hand could only push the object away.
RIGHT_HAND_CLOSED[13:17] = [0.55, 0.0, 0.3, 0.8]


def _as_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value, dtype=np.float64)


def _quat_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Multiply two w-x-y-z quaternions."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dtype=np.float64,
    )


@dataclass(frozen=True)
class ScriptedGraspResult:
    """Measured outcome of the deterministic grasp-and-lift validation."""

    success: bool
    finite: bool
    lift_height_m: float
    final_linear_speed_m_s: float
    max_linear_speed_m_s: float
    final_hand_cube_contacts: int
    max_collision_penetration_m: float
    max_visual_penetration_m: float
    max_ik_error: float
    initial_cube_position: tuple[float, float, float]
    grasp_cube_position: tuple[float, float, float]
    final_cube_position: tuple[float, float, float]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class BimanualOrcaPandaCubeGenesisEnv(gym.Env[np.ndarray, np.ndarray]):
    """A Genesis-powered bimanual ORCA-Panda cube pick-and-place scene.

    Actions are absolute targets for all 48 actuated joints. Observations contain
    robot joint positions/velocities followed by the cube free-joint pose/velocity.
    """

    metadata: ClassVar[dict[str, Any]] = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 30,
    }

    def __init__(
        self,
        render_mode: str | None = None,
        *,
        backend: str = "cpu",
        precision: str = "64",
        frame_skip: int = 1,
        max_episode_steps: int = 2_000,
    ) -> None:
        super().__init__()
        if render_mode not in {None, "human", "rgb_array"}:
            raise ValueError(f"Unsupported render_mode: {render_mode}")
        if frame_skip < 1:
            raise ValueError("frame_skip must be at least 1")

        try:
            import genesis as gs
        except ImportError as exc:
            raise RuntimeError(
                "Genesis is not installed. Install with `pip install -e '.[genesis]'`."
            ) from exc

        self.gs = gs
        self.render_mode = render_mode
        self.frame_skip = frame_skip
        self.max_episode_steps = max_episode_steps
        self._elapsed_steps = 0
        self._closed = False

        if not getattr(gs, "_initialized", False):
            try:
                selected_backend = getattr(gs, backend)
            except AttributeError as exc:
                raise ValueError(f"Unsupported Genesis backend: {backend}") from exc
            gs.init(
                backend=selected_backend,
                precision=precision,
                logging_level=logging.WARNING,
                seed=0,
            )

        self.asset_path = build_genesis_bimanual_orcapanda_mjcf()
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=0.0025, substeps=5),
            rigid_options=gs.options.RigidOptions(
                enable_self_collision=True,
                enable_neutral_collision=False,
                max_collision_pairs=2_000,
                max_contacts=2_000,
                iterations=150,
                noslip_iterations=5,
                enable_torsional_friction=True,
                box_box_detection=True,
                constraint_timeconst=0.001,
            ),
            vis_options=gs.options.VisOptions(
                shadow=True,
                plane_reflection=False,
                background_color=(0.14, 0.14, 0.145),
                ambient_light=(0.30, 0.31, 0.34),
                lights=[
                    gs.options.vis.DirectionalLight(
                        dir=(-1.4, -0.8, -2.2),
                        color=(1.0, 0.94, 0.86),
                        intensity=2.8,
                    ),
                ],
            ),
            viewer_options=gs.options.ViewerOptions(
                camera_pos=CAMERA_POSITION,
                camera_lookat=CAMERA_LOOKAT,
                camera_fov=CAMERA_FOV,
                res=CAMERA_RESOLUTION,
            ),
            renderer=gs.renderers.Rasterizer(),
            show_viewer=render_mode == "human",
        )

        self.plane = self.scene.add_entity(
            morph=gs.morphs.Plane(),
            material=gs.materials.Rigid(friction=1.0),
            surface=gs.surfaces.Rough(
                color=(0.075, 0.082, 0.095, 1.0),
                vis_mode="visual",
                smooth=True,
            ),
        )
        self.table = self.scene.add_entity(
            morph=gs.morphs.MJCF(
                file=str(WORKBENCH_MJCF_PATH),
                requires_jac_and_IK=False,
            ),
            surface=gs.surfaces.Default(
                vis_mode="visual",
                smooth=True,
                default_roughness=0.55,
            ),
        )
        self.robot = self.scene.add_entity(
            morph=gs.morphs.MJCF(
                file=str(self.asset_path),
                pos=(0.0, 0.0, TABLE_TOP_Z),
                decimate=True,
                decimate_face_num=256,
            ),
            material=gs.materials.Rigid(friction=2.5, friction_torsional=0.008),
            surface=gs.surfaces.Default(
                vis_mode="visual",
                smooth=True,
                default_roughness=0.42,
            ),
        )
        self.cube = self.scene.add_entity(
            morph=gs.morphs.MJCF(
                file=str(CUBE_MJCF_PATH),
            ),
            surface=gs.surfaces.Default(
                vis_mode="visual",
                smooth=True,
                default_roughness=0.50,
            ),
        )
        # Genesis treats MJCF collision masks as entity-local by default, so
        # masks are ignored between independently imported MJCF entities. This
        # scene intentionally defines one global contact matrix for the robot,
        # workbench, and cube; opt those entities into scene-wide masks before
        # the collider builds its valid-pair table.
        for entity in (self.table, self.robot, self.cube):
            entity._is_local_collision_mask = False
        self.camera = None
        if render_mode == "rgb_array":
            self.camera = self.scene.add_camera(
                res=CAMERA_RESOLUTION,
                pos=CAMERA_POSITION,
                lookat=CAMERA_LOOKAT,
                fov=CAMERA_FOV,
            )

        self.scene.build()
        self._validate_robot_topology()
        if len(self.cube.geoms) != 2:
            raise RuntimeError("Expected exact-support and hand-skin cube geoms")
        self._cube_hand_geom_idx = int(self.cube.geoms[1].idx)
        self.right_wrist = self.robot.get_link(RIGHT_WRIST_LINK)
        self._configure_controller()

        lower, upper = self.robot.get_dofs_limit()
        self.action_low = _as_numpy(lower).astype(np.float32)
        self.action_high = _as_numpy(upper).astype(np.float32)
        self.action_space = spaces.Box(
            low=self.action_low,
            high=self.action_high,
            dtype=np.float32,
        )

        self.robot.set_qpos(HOME_QPOS, zero_velocity=True)
        wrist_home_quat = _as_numpy(self.right_wrist.get_quat())
        rotate_world_x_minus_90 = np.array(
            [np.sqrt(0.5), -np.sqrt(0.5), 0.0, 0.0], dtype=np.float64
        )
        self._grasp_quat = _quat_multiply(
            rotate_world_x_minus_90,
            wrist_home_quat,
        )
        self._approach_qpos, approach_error = self._solve_right_arm_ik(
            position=APPROACH_WRIST_POSITION,
            init_qpos=READY_QPOS,
        )
        self._grasp_open_qpos, grasp_error = self._solve_right_arm_ik(
            position=np.array([*GRASP_WRIST_XY, 0.39]),
            init_qpos=self._approach_qpos,
        )
        max_error = max(
            float(np.linalg.norm(approach_error)),
            float(np.linalg.norm(grasp_error)),
        )
        if max_error > 5e-3:
            raise RuntimeError(
                f"Could not solve the nominal approach IK poses: error={max_error}"
            )
        # IK only commands the right arm.  Keep the other arm in the ready pose
        # so it remains clear of the task throughout the demonstration.
        self._approach_qpos[LEFT_ARM_DOFS] = READY_LEFT_ARM_QPOS
        self._grasp_open_qpos[LEFT_ARM_DOFS] = READY_LEFT_ARM_QPOS
        self._command_qpos = READY_QPOS.copy()
        self._episode_cube_position = CUBE_POSITION.copy()
        self._rollout_recorder: Callable[[np.ndarray], None] | None = None
        self._rollout_record_interval = 1
        self._rollout_physics_step = 0

        obs = self._get_obs()
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=obs.shape,
            dtype=np.float64,
        )

    def _validate_robot_topology(self) -> None:
        if self.robot.n_dofs != N_ROBOT_DOFS or self.robot.n_qs != N_ROBOT_DOFS:
            raise RuntimeError(
                "Expected the bimanual ORCA-Panda asset to expose 48 one-DOF joints, "
                f"got n_dofs={self.robot.n_dofs}, n_qs={self.robot.n_qs}"
            )
        expected_arm_names = [
            *(f"left_panda_joint{i}" for i in range(1, 8)),
            *(f"right_panda_joint{i}" for i in range(1, 8)),
        ]
        available = {joint.name for joint in self.robot.joints}
        missing = [name for name in expected_arm_names if name not in available]
        if missing:
            raise RuntimeError(f"Bimanual asset is missing Panda joints: {missing}")

    def _configure_controller(self) -> None:
        effort = np.concatenate(
            [
                np.full(N_ARM_DOFS, 200.0),
                np.full(N_HAND_DOFS, 0.5),
                np.full(N_ARM_DOFS, 200.0),
                np.full(N_HAND_DOFS, 0.5),
            ]
        )
        kp = np.concatenate(
            [
                np.full(N_ARM_DOFS, 6_000.0),
                np.full(N_HAND_DOFS, 120.0),
                np.full(N_ARM_DOFS, 6_000.0),
                np.full(N_HAND_DOFS, 120.0),
            ]
        )
        kv = np.concatenate(
            [
                np.full(N_ARM_DOFS, 600.0),
                np.full(N_HAND_DOFS, 12.0),
                np.full(N_ARM_DOFS, 600.0),
                np.full(N_HAND_DOFS, 12.0),
            ]
        )
        self.robot.set_dofs_kp(kp)
        self.robot.set_dofs_kv(kv)
        self.robot.set_dofs_force_range(-effort, effort)

    def _solve_right_arm_ik(
        self,
        *,
        position: np.ndarray,
        init_qpos: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        qpos, error = self.robot.inverse_kinematics(
            link=self.right_wrist,
            pos=np.asarray(position, dtype=np.float64),
            quat=self._grasp_quat,
            init_qpos=init_qpos,
            dofs_idx_local=RIGHT_ARM_DOFS,
            max_samples=300,
            max_solver_iters=300,
            return_error=True,
        )
        return _as_numpy(qpos), _as_numpy(error)

    def _get_obs(self) -> np.ndarray:
        return np.concatenate(
            [
                _as_numpy(self.robot.get_qpos()),
                _as_numpy(self.robot.get_dofs_velocity()),
                _as_numpy(self.cube.get_qpos()),
                _as_numpy(self.cube.get_dofs_velocity()),
            ]
        )

    def _cube_position(self) -> np.ndarray:
        return _as_numpy(self.cube.get_pos())

    def _hand_cube_contacts(self) -> int:
        contacts = self.cube.get_contacts(with_entity=self.robot)
        return int(_as_numpy(contacts["position"]).shape[0])

    def _hand_cube_max_penetration(self) -> float:
        return self._hand_cube_penetrations()[0]

    def _hand_cube_penetrations(self) -> tuple[float, float]:
        """Return maximum contact overlap and overlap into rendered geometry."""
        contacts = self.cube.get_contacts(with_entity=self.robot)
        penetration = _as_numpy(contacts["penetration"])
        if penetration.size == 0:
            return 0.0, 0.0
        geom_a = _as_numpy(contacts["geom_a"]).astype(np.int64)
        geom_b = _as_numpy(contacts["geom_b"]).astype(np.int64)
        hand_skin_contact = (geom_a == self._cube_hand_geom_idx) | (
            geom_b == self._cube_hand_geom_idx
        )
        visual_penetration = penetration.copy()
        visual_penetration[hand_skin_contact] = np.maximum(
            0.0,
            penetration[hand_skin_contact] - CUBE_CONTACT_SKIN,
        )
        return float(np.max(penetration)), float(np.max(visual_penetration))

    @staticmethod
    def _right_digit_from_link_name(link_name: str) -> str | None:
        if "P-Finger" in link_name or "_P-" in link_name:
            return "pinky"
        if any(
            token in link_name for token in ("424a8e75", "8660a1eb", "6ec59111")
        ):
            return "ring"
        if any(
            token in link_name for token in ("34afb748", "08efa608", "e04a96f2")
        ):
            return "middle"
        if "_I-" in link_name:
            return "index"
        if "_T-" in link_name or "T-TP" in link_name:
            return "thumb"
        return None

    def _get_info(self) -> dict[str, Any]:
        cube_position = self._cube_position()
        cube_velocity = _as_numpy(self.cube.get_dofs_velocity())
        lift_height = float(cube_position[2] - self._episode_cube_position[2])
        finite = bool(np.all(np.isfinite(self._get_obs())))
        collision_penetration, visual_penetration = self._hand_cube_penetrations()
        return {
            "cube_position": cube_position.copy(),
            "cube_lift_height": lift_height,
            "hand_cube_contacts": self._hand_cube_contacts(),
            "hand_cube_collision_penetration": collision_penetration,
            "hand_cube_visual_penetration": visual_penetration,
            "cube_linear_speed": float(np.linalg.norm(cube_velocity[:3])),
            "finite": finite,
            "is_success": finite and lift_height >= 0.10,
        }

    def _physics_step(self) -> None:
        self.scene.step(update_visualizer=self.render_mode == "human")
        if self._rollout_recorder is not None:
            self._rollout_physics_step += 1
            if self._rollout_physics_step % self._rollout_record_interval == 0:
                frame = self.render()
                if frame is not None:
                    self._rollout_recorder(frame)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        options = options or {}
        cube_position = np.asarray(
            options.get("cube_position", CUBE_POSITION), dtype=np.float64
        )
        robot_qpos = np.asarray(
            options.get("robot_qpos", READY_QPOS), dtype=np.float64
        )
        settle_steps = int(options.get("settle_steps", 0))
        if cube_position.shape != (3,):
            raise ValueError(
                f"Expected cube_position shape (3,), got {cube_position.shape}"
            )
        if robot_qpos.shape != (N_ROBOT_DOFS,):
            raise ValueError(f"Expected robot_qpos shape (48,), got {robot_qpos.shape}")
        if settle_steps < 0:
            raise ValueError("settle_steps cannot be negative")

        self.scene.reset()
        self._episode_cube_position = cube_position.copy()
        cube_qpos = np.concatenate([cube_position, [1.0, 0.0, 0.0, 0.0]])
        self.cube.set_qpos(cube_qpos, zero_velocity=True)
        self.robot.set_qpos(robot_qpos, zero_velocity=True)
        self._command_qpos = robot_qpos.copy()
        self.robot.control_dofs_position(self._command_qpos)
        for _ in range(settle_steps):
            self._physics_step()
        self._elapsed_steps = 0
        return self._get_obs(), self._get_info()

    def step(
        self,
        action: np.ndarray,
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        action = np.asarray(action, dtype=np.float64)
        if action.shape != (N_ROBOT_DOFS,):
            raise ValueError(f"Expected action shape (48,), got {action.shape}")
        self._command_qpos = np.clip(action, self.action_low, self.action_high)
        self.robot.control_dofs_position(self._command_qpos)
        for _ in range(self.frame_skip):
            self._physics_step()
        self._elapsed_steps += 1

        obs = self._get_obs()
        info = self._get_info()
        reward = float(info["cube_lift_height"] + 0.002 * info["hand_cube_contacts"])
        if info["is_success"]:
            reward += 1.0
        terminated = not info["finite"]
        truncated = self._elapsed_steps >= self.max_episode_steps
        return obs, reward, terminated, truncated, info

    def _drive_to(
        self,
        target: np.ndarray,
        steps: int,
        speed_samples: list[float],
        penetration_samples: list[float] | None = None,
        visual_penetration_samples: list[float] | None = None,
    ) -> None:
        start = self._command_qpos.copy()
        target = target.copy()
        for alpha in np.linspace(0.0, 1.0, steps, endpoint=True):
            self._command_qpos = (1.0 - alpha) * start + alpha * target
            self.robot.control_dofs_position(self._command_qpos)
            self._physics_step()
            velocity = _as_numpy(self.cube.get_dofs_velocity())
            speed_samples.append(float(np.linalg.norm(velocity[:3])))
            if penetration_samples is not None:
                depth, visual_depth = self._hand_cube_penetrations()
                penetration_samples.append(depth)
                if visual_penetration_samples is not None:
                    visual_penetration_samples.append(visual_depth)

    def _hold(
        self,
        steps: int,
        speed_samples: list[float],
        penetration_samples: list[float] | None = None,
        visual_penetration_samples: list[float] | None = None,
    ) -> None:
        self._drive_to(
            self._command_qpos,
            steps,
            speed_samples,
            penetration_samples,
            visual_penetration_samples,
        )

    def _close_hand(
        self,
        speed_samples: list[float],
        penetration_samples: list[float],
        visual_penetration_samples: list[float],
    ) -> None:
        """Close to a force-limited opposing pinch, then add bounded preload."""
        start = self._command_qpos.copy()
        closed = start.copy()
        closed[RIGHT_HAND_DOFS] = RIGHT_HAND_CLOSED
        max_contacts = 0
        contacted_links: set[str] = set()
        links = self.scene.rigid_solver.links
        opposing_contact_alpha: float | None = None

        for alpha in np.linspace(0.0, 1.0, 1_001, endpoint=True):
            command = (1.0 - alpha) * start + alpha * closed
            self._command_qpos = command
            self.robot.control_dofs_position(command)
            self._physics_step()
            velocity = _as_numpy(self.cube.get_dofs_velocity())
            speed_samples.append(float(np.linalg.norm(velocity[:3])))
            collision_depth, visual_depth = self._hand_cube_penetrations()
            penetration_samples.append(collision_depth)
            visual_penetration_samples.append(visual_depth)
            max_contacts = max(max_contacts, self._hand_cube_contacts())
            contacts = self.cube.get_contacts(with_entity=self.robot)
            active_digits: set[str] = set()
            for link_a, link_b in zip(
                _as_numpy(contacts["link_a"]).astype(np.int64),
                _as_numpy(contacts["link_b"]).astype(np.int64),
                strict=True,
            ):
                name_a = links[int(link_a)].name
                name_b = links[int(link_b)].name
                robot_link_name = name_a if name_a.startswith("orca") else name_b
                contacted_links.add(robot_link_name)
                digit = self._right_digit_from_link_name(robot_link_name)
                if digit is not None:
                    active_digits.add(digit)

            has_opposing_contact = "thumb" in active_digits and bool(
                active_digits - {"thumb"}
            )
            if has_opposing_contact and opposing_contact_alpha is None:
                opposing_contact_alpha = float(alpha)
            # Continue only a small, bounded fraction past first opposition.
            # This supplies grip preload without ever resuming toward the
            # fully-closed through-object target that caused the original bug.
            if (
                opposing_contact_alpha is not None
                and alpha >= opposing_contact_alpha + 0.12
            ):
                return

        raise RuntimeError(
            "Force-limited closure did not establish opposing contacts "
            f"(peak={max_contacts}, links={sorted(contacted_links)}, "
            f"cube={self._cube_position().tolist()})"
        )

    def run_scripted_grasp(
        self,
        *,
        capture: Callable[[str, np.ndarray], None] | None = None,
        record: Callable[[np.ndarray], None] | None = None,
        record_fps: int = 30,
        cube_position: np.ndarray | None = None,
    ) -> ScriptedGraspResult:
        """Approach from the ready pose, then run the validated grasp and lift."""
        if record is not None and self.render_mode != "rgb_array":
            raise ValueError("Rollout recording requires render_mode='rgb_array'")
        if record_fps <= 0:
            raise ValueError("record_fps must be positive")
        requested_cube_position = np.asarray(
            CUBE_POSITION if cube_position is None else cube_position,
            dtype=np.float64,
        )
        if requested_cube_position.shape != (3,):
            raise ValueError(
                "Expected cube_position shape (3,), got "
                f"{requested_cube_position.shape}"
            )
        cube_delta = requested_cube_position - CUBE_POSITION
        if np.allclose(cube_delta, 0.0):
            approach_qpos = self._approach_qpos
            grasp_open_qpos = self._grasp_open_qpos
        else:
            approach_qpos, approach_error = self._solve_right_arm_ik(
                position=APPROACH_WRIST_POSITION + cube_delta,
                init_qpos=READY_QPOS,
            )
            grasp_open_qpos, grasp_error = self._solve_right_arm_ik(
                position=np.array([*GRASP_WRIST_XY, 0.39]) + cube_delta,
                init_qpos=approach_qpos,
            )
            if max(
                float(np.linalg.norm(approach_error)),
                float(np.linalg.norm(grasp_error)),
            ) > 5e-3:
                raise RuntimeError("Could not solve the requested cube grasp pose")
            approach_qpos[LEFT_ARM_DOFS] = READY_LEFT_ARM_QPOS
            grasp_open_qpos[LEFT_ARM_DOFS] = READY_LEFT_ARM_QPOS
        self._rollout_recorder = record
        self._rollout_record_interval = max(1, round(1.0 / (0.0025 * record_fps)))
        self._rollout_physics_step = 0
        self.reset(
            options={"settle_steps": 200, "cube_position": requested_cube_position}
        )
        speed_samples: list[float] = []
        penetration_samples: list[float] = []
        visual_penetration_samples: list[float] = []
        initial_position = self._cube_position()
        if capture is not None:
            frame = self.render()
            if frame is not None:
                capture("ready", frame)

        # Move above the workspace first.  A direct joint-space interpolation
        # from ready to the low grasp pose sweeps the elbow through the table.
        self._drive_to(
            approach_qpos,
            1_200,
            speed_samples,
            penetration_samples,
            visual_penetration_samples,
        )
        self._hold(
            200,
            speed_samples,
            penetration_samples,
            visual_penetration_samples,
        )
        self._drive_to(
            grasp_open_qpos,
            1_000,
            speed_samples,
            penetration_samples,
            visual_penetration_samples,
        )
        self._hold(
            200,
            speed_samples,
            penetration_samples,
            visual_penetration_samples,
        )
        if capture is not None:
            frame = self.render()
            if frame is not None:
                capture("open", frame)

        self._close_hand(
            speed_samples,
            penetration_samples,
            visual_penetration_samples,
        )
        closed = self._command_qpos.copy()
        grasp_hand_qpos = self._command_qpos[RIGHT_HAND_DOFS].copy()
        self._drive_to(
            closed,
            500,
            speed_samples,
            penetration_samples,
            visual_penetration_samples,
        )
        grasp_position = self._cube_position()
        if capture is not None:
            frame = self.render()
            if frame is not None:
                capture("grasp", frame)

        max_ik_error = 0.0
        for z in (0.42, 0.45, 0.48, 0.51, 0.54, 0.57):
            solution, error = self._solve_right_arm_ik(
                position=np.array([*GRASP_WRIST_XY, z]) + cube_delta,
                init_qpos=self._command_qpos,
            )
            max_ik_error = max(max_ik_error, float(np.linalg.norm(error)))
            target = self._command_qpos.copy()
            target[RIGHT_ARM_DOFS] = solution[RIGHT_ARM_DOFS]
            target[RIGHT_HAND_DOFS] = grasp_hand_qpos
            self._drive_to(
                target,
                700,
                speed_samples,
                penetration_samples,
                visual_penetration_samples,
            )
        self._hold(
            300,
            speed_samples,
            penetration_samples,
            visual_penetration_samples,
        )

        final_position = self._cube_position()
        final_velocity = _as_numpy(self.cube.get_dofs_velocity())
        final_speed = float(np.linalg.norm(final_velocity[:3]))
        contacts = self._hand_cube_contacts()
        obs = self._get_obs()
        finite = bool(np.all(np.isfinite(obs)))
        lift_height = float(final_position[2] - initial_position[2])
        max_speed = max(speed_samples, default=0.0)
        max_collision_penetration = max(penetration_samples, default=0.0)
        max_visual_penetration = max(visual_penetration_samples, default=0.0)
        success = bool(
            finite
            and lift_height >= 0.12
            and contacts >= 2
            and final_speed < 0.05
            and max_speed < 5.0
            and max_visual_penetration <= 1e-4
            and max_ik_error < 5e-3
        )
        if capture is not None:
            frame = self.render()
            if frame is not None:
                capture("lift", frame)
        self._rollout_recorder = None
        return ScriptedGraspResult(
            success=success,
            finite=finite,
            lift_height_m=lift_height,
            final_linear_speed_m_s=final_speed,
            max_linear_speed_m_s=max_speed,
            final_hand_cube_contacts=contacts,
            max_collision_penetration_m=max_collision_penetration,
            max_visual_penetration_m=max_visual_penetration,
            max_ik_error=max_ik_error,
            initial_cube_position=tuple(float(x) for x in initial_position),
            grasp_cube_position=tuple(float(x) for x in grasp_position),
            final_cube_position=tuple(float(x) for x in final_position),
        )

    def render(self) -> np.ndarray | None:
        if self.render_mode == "rgb_array":
            if self.camera is None:
                raise RuntimeError("RGB camera was not created")
            return np.asarray(self.camera.render(rgb=True, force_render=True)[0])
        return None

    def close(self) -> None:
        if not self._closed:
            self.scene.destroy()
            self._closed = True
