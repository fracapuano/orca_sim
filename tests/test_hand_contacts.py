"""Isolate hand contacts: no Panda, IK, grasp attachment or force adapter."""

import hashlib
import xml.etree.ElementTree as ET
from xml.sax.saxutils import escape

import mujoco
import numpy as np
import pytest

from orca_sim.contact_diagnostics import skin_geoms, skin_vertex_depth
from orca_sim.versions import PACKAGE_ROOT


def fixture_model(side, legacy=False, skin_disabled=False):
    root = PACKAGE_ROOT / "models/v2"
    body = ET.parse(root / f"mjcf/orcahand_{side}_body.xml").getroot().find("body")
    if legacy:
        body.attrib.pop("childclass")
    if skin_disabled:
        for geom in body.iter("geom"):
            if "Skin" in geom.get("mesh", ""):
                geom.set("contype", "0")
                geom.set("conaffinity", "0")
    x = -0.0143 if side == "right" else -0.0057
    option = '<option cone="pyramidal" impratio="1"/>' if legacy else ""
    xml = f'''<mujoco>
      <include file="{escape(str(root / "assets/options.xml"))}"/>
      <include file="{escape(str(root / f"mjcf/orcahand_{side}.mjcf"))}"/>
      {option}
      <worldbody>
        <geom name="floor" type="plane" size="1 1 .01"/>
        <body pos="0 0 .45">{ET.tostring(body, encoding="unicode")}</body>
        <body name="cube" pos="{x} .1762 .4049">
          <freejoint/>
          <geom name="cube_geom" type="box" size=".025 .025 .025"
                mass=".08" contype="1" conaffinity="3"/>
        </body>
        <body name="support" mocap="true" pos="{x} .1762 .3719">
          <geom name="support_geom" type="box" size=".04 .04 .008"
                contype="2" conaffinity="0"/>
        </body>
      </worldbody>
    </mujoco>'''
    return mujoco.MjModel.from_xml_string(xml)


@pytest.mark.parametrize("side", ["left", "right"])
def test_contacts_change_but_mechanics_and_limits_do_not(side):
    fixed = fixture_model(side)
    previous = fixture_model(side, legacy=True, skin_disabled=True)
    skins = skin_geoms(fixed)
    assert len(skins) == 10
    np.testing.assert_array_equal(fixed.geom_contype[skins], 1)
    np.testing.assert_array_equal(fixed.geom_conaffinity[skins], 1)
    np.testing.assert_array_equal(previous.geom_contype[skins], 0)
    assert fixed.opt.cone == mujoco.mjtCone.mjCONE_ELLIPTIC
    assert fixed.opt.impratio == 10
    assert fixed.opt.timestep == previous.opt.timestep == 0.002
    for field in (
        "body_mass",
        "body_inertia",
        "body_pos",
        "body_quat",
        "mesh_vert",
        "mesh_face",
        "geom_friction",
        "geom_margin",
        "geom_size",
        "geom_pos",
        "geom_quat",
        "jnt_range",
        "jnt_pos",
        "jnt_axis",
        "dof_damping",
        "dof_armature",
        "dof_frictionloss",
        "actuator_gainprm",
        "actuator_biasprm",
        "actuator_ctrlrange",
        "actuator_forcerange",
    ):
        np.testing.assert_array_equal(getattr(fixed, field), getattr(previous, field))
    np.testing.assert_array_equal(fixed.actuator_forcerange, np.tile([-1, 1], (17, 1)))
    np.testing.assert_array_equal(fixed.actuator_gainprm[:, 0], 2)
    for i in range(fixed.ngeom):
        if fixed.body(int(fixed.geom_bodyid[i])).name.startswith(side + "_"):
            np.testing.assert_allclose(fixed.geom_solref[i], [0.004, 1])
            np.testing.assert_allclose(fixed.geom_solimp[i, :3], [0.999, 0.9999, 0.001])
            assert fixed.geom_priority[i] == 1


def progressive_commands(model, side, time):
    """Identical open-loop targets in both variants; MCP/PIP reach native limits."""
    base = np.zeros(model.nu)
    strong = base.copy()
    limit = base.copy()
    for i, joint in enumerate(model.actuator_trnid[:, 0]):
        name = model.joint(int(joint)).name.removeprefix(side + "_")
        if name in {f"{d}-{j}" for d in "imrp" for j in ("mcp", "pip")}:
            base[i] = 1.45 * 0.6
        elif name in {"t-cmc", "t-mcp", "t-pip"}:
            base[i] = {"t-cmc": 0.55, "t-mcp": 0.3, "t-pip": 0.8}[name] * 0.6
        strong[i] = base[i] / 0.6 if name.endswith(("-mcp", "-pip")) else base[i]
        limit[i] = (
            model.actuator_ctrlrange[i, 1]
            if name.endswith(("-mcp", "-pip"))
            else strong[i]
        )
    return (
        base
        + np.clip((time - 5) / 10, 0, 1) * (strong - base)
        + np.clip((time - 15) / 10, 0, 1) * (limit - strong)
    ) * np.clip((time - 0.5) / 2, 0, 1)


def squeeze_rollout(side, legacy=False):
    # For the pressure ablation, keep skins enabled in BOTH cases. This
    # isolates solver/contact response from the separate skin-mask regression.
    m = fixture_model(side, legacy=legacy)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    cube, support = m.geom("cube_geom").id, m.geom("support_geom").id
    support_start = d.mocap_pos[0].copy()
    skins = skin_geoms(m)
    assert skin_vertex_depth(m, d, cube, skins) == 0
    command_hash = hashlib.sha256()
    max_depth = max_skin = max_torque = 0.0
    final_contacts = []
    hand = {
        i
        for i in range(m.ngeom)
        if m.body(int(m.geom_bodyid[i])).name.startswith(side + "_")
    }
    for step in range(round(40 / m.opt.timestep)):
        t = step * m.opt.timestep
        d.ctrl[:] = progressive_commands(m, side, t)
        command_hash.update(np.asarray(d.ctrl, dtype="<f8").tobytes())
        d.mocap_pos[0] = support_start - [0, 0, 0.25 * np.clip((t - 3) / 0.5, 0, 1)]
        mujoco.mj_step(m, d)
        for c in d.contact:
            if (c.geom1 == cube and c.geom2 in hand) or (
                c.geom2 == cube and c.geom1 in hand
            ):
                max_depth = max(max_depth, -float(c.dist))
        max_torque = max(max_torque, float(np.max(np.abs(d.actuator_force))))
        if step % 10:
            continue
        mujoco.mj_forward(m, d)
        max_skin = max(max_skin, skin_vertex_depth(m, d, cube, skins))
        if t >= 39.8:
            count, digits, support_contacts = 0, set(), 0
            wrench = np.zeros(6)
            for index, contact in enumerate(d.contact):
                if cube not in (contact.geom1, contact.geom2):
                    continue
                other = contact.geom2 if contact.geom1 == cube else contact.geom1
                if other == support:
                    support_contacts += 1
                if other not in hand:
                    continue
                mujoco.mj_contactForce(m, d, index, wrench)
                if wrench[0] <= 1e-6:
                    continue
                count += 1
                body = int(m.geom_bodyid[other])
                while body:
                    for j in range(
                        m.body_jntadr[body], m.body_jntadr[body] + m.body_jntnum[body]
                    ):
                        name = m.joint(j).name
                        if name.startswith(side + "_") and "-" in name:
                            digits.add(name.split("_")[-1].split("-")[0])
                    body = int(m.body_parentid[body])
            final_contacts.append(
                count >= 2
                and "t" in digits
                and len(digits) >= 2
                and not support_contacts
                and d.body("cube").xpos[2] > 0.3549
            )
    assert m.neq == 0
    assert not np.any(d.xfrc_applied) and not np.any(d.qfrc_applied)
    assert np.isfinite(d.qpos).all() and np.isfinite(d.qvel).all()
    assert not any(w.number for w in d.warning)
    return dict(
        retained=all(final_contacts),
        max_depth=max_depth,
        max_skin=max_skin,
        max_torque=max_torque,
        commands=command_hash.hexdigest(),
    )


@pytest.mark.slow
@pytest.mark.parametrize("side", ["left", "right"])
def test_raw_increasing_closure_retains_cube_with_corrected_contacts(side):
    previous, fixed = squeeze_rollout(side, True), squeeze_rollout(side)
    assert previous["commands"] == fixed["commands"]
    assert not previous["retained"] and previous["max_skin"] > 0.002
    assert fixed["retained"]
    assert fixed["max_skin"] < 0.0001 and fixed["max_depth"] < 0.0001
    assert fixed["max_torque"] <= 1
