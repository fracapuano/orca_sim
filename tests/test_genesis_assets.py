from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from orca_sim.genesis_assets import prepare_bimanual_orcapanda_mjcf


def test_prepare_mjcf_preserves_panda_and_recomputes_hand_inertias(
    tmp_path: Path,
) -> None:
    mesh_dir = tmp_path / "meshes"
    mesh_dir.mkdir()
    (mesh_dir / "part.stl").write_bytes(b"mesh placeholder")
    source = tmp_path / "source.xml"
    source.write_text(
        """\
<mujoco model="test">
  <compiler meshdir="meshes"/>
  <asset><mesh name="part" file="part.stl"/></asset>
  <worldbody>
    <body name="panda_link">
      <inertial mass="2" pos="0 0 0" diaginertia="1 1 1"/>
    </body>
    <body name="orcahand_right_palm">
      <inertial mass="8" pos="0 0 0" diaginertia="100 100 100"/>
      <geom mesh="part" density="0" contype="0" conaffinity="0"/>
      <geom mesh="part" mass="8"/>
    </body>
  </worldbody>
</mujoco>
""",
        encoding="utf-8",
    )

    output = prepare_bimanual_orcapanda_mjcf(
        source,
        tmp_path / "generated" / "genesis.xml",
        hand_density=725.0,
    )
    root = ET.parse(output).getroot()

    mesh = root.find(".//asset/mesh")
    assert mesh is not None
    assert Path(mesh.get("file", "")).is_absolute()
    assert root.find(".//body[@name='panda_link']/inertial") is not None

    hand = root.find(".//body[@name='orcahand_right_palm']")
    assert hand is not None
    assert hand.find("inertial") is None
    visual, collision = hand.findall("geom")
    assert visual.get("density") == "0"
    assert collision.get("density") == "725"
    assert "mass" not in collision.attrib


def test_prepare_mjcf_rejects_non_orca_asset(tmp_path: Path) -> None:
    source = tmp_path / "source.xml"
    source.write_text("<mujoco><worldbody/></mujoco>", encoding="utf-8")
    with pytest.raises(ValueError, match="expected ORCA-hand"):
        prepare_bimanual_orcapanda_mjcf(source, tmp_path / "output.xml")
