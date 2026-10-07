"""
bank.py

Stateful nearest-neighbour index for the methane plume bank.
Loads and flattens the TACO catalogue once; all subsequent
lookups run entirely in-memory.

Matching strategy (sequential filtering):
    1. wind_speed -- snap to nearest grid value, filter exact
    2. sza        -- snap to nearest grid value, filter exact
    3. raa        -- snap to nearest value among remaining candidates

Steps 1 and 2 use regular grids (all plumes exist at all values),
so the intersection is never empty.  RAA varies per plume, so
step 3 picks the closest available value(s).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import tacoreader


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _nearest(value: float, grid: list) -> float:
    """Return the grid point closest to *value* (linear distance)."""
    arr = np.asarray(grid, dtype=float)
    return grid[int(np.argmin(np.abs(arr - value)))]


def _circular_diff(a: float, b: np.ndarray) -> np.ndarray:
    """Absolute angular difference on a 360 degree circle."""
    diff = np.abs((b % 360) - (a % 360))
    return np.minimum(diff, 360 - diff)


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class MethaneBank:
    """Sequential nearest-neighbour lookup over the methane plume bank.

    Parameters
    ----------
    bank_dir:
        Root directory of the TACO bank.
    """

    def __init__(self, bank_dir: str | Path) -> None:
        tacoreader.use("pandas")
        ds = tacoreader.load(str(bank_dir))
        self._df = ds.flatten()

        # Extract grid values from the bank metadata
        self.sza_values = sorted(self._df["l0:sun:sza"].unique().tolist())
        self.raa_values = sorted(self._df["l0:sun:raa"].unique().tolist())
        self.wind_speeds = sorted(self._df["l0:methane:wind_speed"].unique().tolist())

    def query(
        self,
        *,
        sza: float | None = None,
        raa: float | None = None,
        wind_speed: float | None = None,
        source_type: str | None = None,
        exclude_ids: list[str] | None = None,
        meta: bool = False,
        n: int = 1,
    ) -> str | dict | list[str] | list[dict]:
        """Return GDAL VSI path(s) for the plume(s) closest to the given params.

        Parameters
        ----------
        sza, raa, wind_speed, source_type:
            Filtering / matching parameters.
        exclude_ids : list[str] | None
            Plume IDs to exclude (avoid duplicates within one injection).
        meta : bool
            If ``True`` return dict(s) with path + all plume parameters.
        n : int
            Number of distinct plumes to return.

        Returns
        -------
        str | dict | list[str] | list[dict]
        """
        df = self._df

        # --- 1. Categorical filter ---
        if source_type is not None:
            df = df[df["l0:methane:sim_type"] == source_type]

        if exclude_ids:
            df = df[~df["l0:id"].isin(exclude_ids)]

        # --- 2. Wind speed: snap to nearest grid value, filter exact ---
        if wind_speed is not None:
            s = _nearest(wind_speed, self.wind_speeds)
            df = df[df["l0:methane:wind_speed"] == s]

        # --- 3. SZA: snap to nearest grid value, filter exact ---
        if sza is not None:
            s = _nearest(sza, self.sza_values)
            df = df[df["l0:sun:sza"] == s]

        # --- 4. RAA: select the n closest among remaining candidates ---
        if raa is not None and len(df) > 0:
            raa_diff = _circular_diff(raa, df["l0:sun:raa"].values)
            k = min(n, len(df))
            cutoff = np.partition(raa_diff, k - 1)[k - 1]
            pool = np.where(raa_diff <= cutoff + 1e-10)[0]
            chosen = np.random.choice(pool, size=k, replace=False)
            df = df.iloc[chosen]

        if df.empty:
            raise ValueError(
                f"No plumes found for sza={sza}, raa={raa}, "
                f"wind_speed={wind_speed}, source_type={source_type}"
            )

        # --- 5. Random selection (if RAA was None or n < len) ---
        k = min(n, len(df))
        indices = np.random.choice(len(df), size=k, replace=False)
        rows = [df.iloc[i] for i in indices]

        def _to_result(row):
            if not meta:
                return row["gdal_vsi"]
            return {
                "id":           row["l0:id"],
                "gdal_vsi":     row["gdal_vsi"],
                "sim_type":     row["l0:methane:sim_type"],
                "plume_uid":    row["l0:methane:plume_uid"],
                "wind_speed":   row["l0:methane:wind_speed"],
                "sza":          row["l0:sun:sza"],
                "raa":          row["l0:sun:raa"],
                "snapshot":     row["l0:methane:snapshot_index"],
                "source_row":   row["l0:array:source_row"],
                "source_col":   row["l0:array:source_col"],
                "peak_ppb":     row["l0:array:peak_ppb"],
            }

        if n == 1:
            return _to_result(rows[0])

        return [_to_result(r) for r in rows]
