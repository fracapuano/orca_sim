"""Camera geometry/physics tests run headlessly; renderer API uses a fake GL context."""

import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import pytest

from orca_sim import mujoco_bimanual as bim


@pytest.fixture
def env():
    e = bim.OrcaBimanualPandaCubeStackingMujoco()
    yield e
    e.close()


def test_exactly_three_fixed_cameras_with_arm_scoped_palm_attachments(env):
    m, d = env.model, env.data
    assert m.ncam == 3
    assert {m.camera(i).name for i in range(m.ncam)} == set(env.camera_names)
    front = m.camera("front").id
    assert m.cam_bodyid[front] == 0
    assert d.cam_xpos[front, 1] < min(
        v["base_pos"][1] for v in env.start_pose.values() if isinstance(v, dict)
    )
    assert -d.cam_xmat[front].reshape(3, 3)[1, 2] > 0
    for arm, robot in env.robots.items():
        camera = m.camera(arm + "_wrist").id
        palm = m.jnt_bodyid[m.joint(robot.hand_side + "_wrist").id]
        assert m.cam_bodyid[camera] == palm
        assert env.body_robots[palm] == arm
        assert m.cam_mode[camera] == mujoco.mjtCamLight.mjCAMLIGHT_FIXED
        # Under the palm heel, pointing toward fingers and out of the palm.
        assert m.cam_pos[camera, 1] > 0.04
        assert m.cam_pos[camera, 2] < -0.05
        local_forward = d.xmat[palm].reshape(3, 3).T @ (
            -d.cam_xmat[camera].reshape(3, 3)[:, 2]
        )
        np.testing.assert_allclose(local_forward, [0, 0.5, np.sqrt(0.75)], atol=1e-12)


@pytest.mark.parametrize("side", bim.SIDES)
@pytest.mark.parametrize("joint", ["arm", "wrist"])
def test_camera_follows_own_arm_and_hand_wrist_not_other_robot(env, side, joint):
    m, d = env.model, env.data
    before_pos, before_mat = d.cam_xpos.copy(), d.cam_xmat.copy()
    robot = env.robots[side]
    index = (
        robot.arm_qadr[0]
        if joint == "arm"
        else m.jnt_qposadr[m.joint(robot.hand_side + "_wrist").id]
    )
    d.qpos[index] += 0.15
    mujoco.mj_forward(m, d)
    for name in env.camera_names:
        i = m.camera(name).id
        if name == side + "_wrist":
            assert np.linalg.norm(d.cam_xmat[i] - before_mat[i]) > 0.1
            assert np.linalg.norm(d.cam_xpos[i] - before_pos[i]) > 0.001
        else:
            np.testing.assert_array_equal(d.cam_xpos[i], before_pos[i])
            np.testing.assert_array_equal(d.cam_xmat[i], before_mat[i])


def test_camera_only_mjcf_addition_does_not_change_physics(env):
    root = ET.fromstring(env.xml)
    for parent in root.iter():
        for camera in parent.findall("camera"):
            parent.remove(camera)
    baseline = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    data = mujoco.MjData(baseline)
    mujoco.mj_resetDataKeyframe(baseline, data, baseline.key("orcapanda_home").id)
    assert baseline.ncam == 0
    for field in (
        "body_mass",
        "body_inertia",
        "geom_contype",
        "geom_conaffinity",
        "geom_size",
        "geom_pos",
        "geom_quat",
        "geom_friction",
        "geom_solref",
        "geom_solimp",
        "actuator_gainprm",
        "actuator_biasprm",
        "actuator_forcerange",
        "jnt_range",
        "key_qpos",
        "key_ctrl",
    ):
        np.testing.assert_array_equal(
            getattr(env.model, field), getattr(baseline, field)
        )
    home = env.data.ctrl.copy()
    for i in range(300):
        command = home.copy()
        command[[0, 24]] += 0.03 * np.sin(i * 0.01)
        env.step(command)
        data.ctrl[:] = command
        mujoco.mj_step(baseline, data, nstep=env.frame_skip)
        mujoco.mj_forward(baseline, data)
        np.testing.assert_array_equal(env.data.qpos, data.qpos)
        np.testing.assert_array_equal(env.data.qvel, data.qvel)
        np.testing.assert_array_equal(env.data.actuator_force, data.actuator_force)


class FakeRenderer:
    def __init__(self, model, *, width, height):
        self.width, self.height, self.closed = width, height, False
        self.pixels = np.zeros((height, width, 3), dtype=np.uint8)
        self.calls = []

    def update_scene(self, data, *, camera):
        self.calls.append((data.time, camera))
        self.pixels.fill(bim.CAMERA_NAMES.index(camera) + 1)

    def render(self):
        return self.pixels

    def close(self):
        self.closed = True


def test_capture_is_synchronous_independent_buffers_and_reuses_context(
    env, monkeypatch
):
    monkeypatch.setattr(bim.mujoco, "Renderer", FakeRenderer)
    qpos, ctrl, time = env.data.qpos.copy(), env.data.ctrl.copy(), env.data.time
    images = env.render_cameras()
    renderer = env._camera_renderer
    assert list(images) == list(env.camera_names)
    assert renderer.calls == [(time, n) for n in env.camera_names]
    for i, rgb in enumerate(images.values()):
        assert rgb.shape == (480, 640, 3) and rgb.dtype == np.uint8
        assert np.all(rgb == i + 1)
        assert not np.shares_memory(rgb, renderer.pixels)
    np.testing.assert_array_equal(qpos, env.data.qpos)
    np.testing.assert_array_equal(ctrl, env.data.ctrl)
    assert time == env.data.time
    assert env._renderer is None  # Ordinary free-view rendering is independent.
    env.render_camera("front")
    assert env._camera_renderer is renderer
    env.render_camera("front", width=320, height=240)
    assert renderer.closed
    replacement = env._camera_renderer
    env.close()
    env.close()
    assert replacement.closed and env._camera_renderer is None


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(name="unknown"),
        dict(name="front", width=0),
        dict(name="front", height=721),
        dict(name="front", width=1281),
        dict(name="front", width=640.0),
        dict(name="front", height=True),
    ],
)
def test_invalid_camera_request_does_not_allocate_renderer(env, kwargs):
    with pytest.raises(ValueError):
        env.render_camera(**kwargs)
    assert env._camera_renderer is None
