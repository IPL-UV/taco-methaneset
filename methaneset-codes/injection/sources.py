"""
sources.py

Generate random emission source positions within an EMIT patch.

Each source is placed at least ``margin`` pixels from the patch edge
and at least ``min_separation`` pixels from every other source.
"""

from __future__ import annotations

import numpy as np


def generate_source_positions(
    n_plumes: int | None,
    patch_h: int,
    patch_w: int,
    margin: int = 10,
    min_separation: int = 15,
    max_attempts: int = 1000,
) -> list[tuple[int, int]]:
    """Return a list of (row, col) source positions inside the patch.

    Parameters
    ----------
    n_plumes:
        Number of plumes to place.  ``None`` draws a random count in [1, 10].
    patch_h, patch_w:
        Patch dimensions in pixels.
    margin:
        Minimum distance from the patch edge for each source.
    min_separation:
        Minimum pixel distance between any two sources.
    max_attempts:
        Maximum random trials per source before giving up.

    Returns
    -------
    list of (row, col) tuples.  May be shorter than *n_plumes* if
    the patch is too small to fit all sources with the requested
    separation.
    """
    if n_plumes is None:
        n_plumes = int(np.random.randint(1, 11))

    positions: list[tuple[int, int]] = []
    for _ in range(n_plumes):
        for _attempt in range(max_attempts):
            r = int(np.random.randint(margin, patch_h - margin))
            c = int(np.random.randint(margin, patch_w - margin))
            if all(
                np.hypot(r - pr, c - pc) >= min_separation
                for pr, pc in positions
            ):
                positions.append((r, c))
                break

    return positions
