"""Serializable, environment-owned distributions for tabletop cube centres."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

CUBES = ("red_cube", "blue_cube")


def spawn_configuration(value, xy_limits):
    """Validate uniform world-XY rectangles; None retains fixed placement."""
    if value is None:
        return None
    if isinstance(value, (str, Path)):
        value = json.loads(Path(value).read_text())
    if not isinstance(value, dict) or set(value) != {
        "version",
        "red_cube",
        "blue_cube",
        "min_gap_m",
        "max_attempts",
    }:
        raise ValueError(
            "Expected version, both cube regions, min_gap_m and max_attempts"
        )
    if type(value["version"]) is not int or value["version"] != 1:
        raise ValueError("Unsupported cube spawn region version")
    gap = np.asarray(value["min_gap_m"], dtype=float)
    if gap.shape != () or not np.isfinite(gap) or gap < 0:
        raise ValueError("min_gap_m must be finite and nonnegative")
    attempts = value["max_attempts"]
    if type(attempts) is not int or not 1 <= attempts <= 10000:
        raise ValueError("max_attempts must be an integer in [1, 10000]")
    result = dict(version=1, min_gap_m=float(gap), max_attempts=attempts)
    for name in CUBES:
        region = value[name]
        if not isinstance(region, dict) or set(region) != {"x", "y"}:
            raise ValueError(f"{name} requires x and y bounds")
        result[name] = {}
        for axis, limits in zip(("x", "y"), xy_limits):
            bounds = np.asarray(region[axis], dtype=float)
            if (
                bounds.shape != (2,)
                or not np.isfinite(bounds).all()
                or bounds[0] > bounds[1]
                or bounds[0] < limits[0]
                or bounds[1] > limits[1]
            ):
                raise ValueError(
                    f"{name}.{axis} needs ordered finite bounds within {limits}"
                )
            result[name][axis] = bounds.tolist()
    return result


def sample_xy(rng, config):
    """One joint proposal: independently uniform within each cube's rectangle."""
    return {
        name: np.array([rng.uniform(*config[name][axis]) for axis in ("x", "y")])
        for name in CUBES
    }


def separated(xy, half_sizes, gap):
    """Reject overlap; require the specified Euclidean edge-to-edge XY gap."""
    delta = np.abs(xy[CUBES[0]] - xy[CUBES[1]])
    edges = delta - half_sizes[CUBES[0]][:2] - half_sizes[CUBES[1]][:2]
    return bool(np.any(edges > 0) and np.linalg.norm(np.maximum(edges, 0)) >= gap)
