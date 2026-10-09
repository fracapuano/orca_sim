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
