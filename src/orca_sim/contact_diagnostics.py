"""Mesh-based contact diagnostics, independent of any grasp controller.

Vertex containment measures a lower bound, not a complete mesh intersection.
"""

import mujoco
import numpy as np


def skin_geoms(model):
    return [
        i
        for i in range(model.ngeom)
        if model.geom_dataid[i] >= 0
        and model.geom_type[i] == mujoco.mjtGeom.mjGEOM_MESH
        and "Skin" in model.mesh(int(model.geom_dataid[i])).name
    ]


def skin_vertex_depth(model, data, cube_id, skins):
    """Lower bound: deepest rendered skin mesh vertex inside the actual cube."""
    maximum = 0.0
    cube_rotation = data.geom_xmat[cube_id].reshape(3, 3)
    for geom in skins:
        mesh = int(model.geom_dataid[geom])
        adr, num = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
        verts = model.mesh_vert[adr : adr + num]
        world = verts @ data.geom_xmat[geom].reshape(3, 3).T + data.geom_xpos[geom]
        local = (world - data.geom_xpos[cube_id]) @ cube_rotation
        depth = np.min(model.geom_size[cube_id] - np.abs(local), axis=1)
        maximum = max(maximum, float(np.max(depth, initial=0)))
    return maximum
