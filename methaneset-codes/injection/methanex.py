"""
methanex.py

Methane column enhancement retrieval from imaging spectrometer data.

    eng = MethaneRetrieval(device="cuda")

    mf, R, meta = eng.retrieve(rad, MFConfig())
    mf, R, meta = eng.retrieve(rad, RMFConfig())
    mf, R, meta = eng.retrieve(rad, MAG1CConfig())
    mf, R, meta = eng.retrieve(rad, MAG1CFastConfig(tol=1e-5))

rad must be a torch.Tensor of shape (downtrack, crosstrack, bands).
retrieve() returns a RetrievalResult that supports both named access
and tuple unpacking (mf, albedo, metadata).

The CH4 absorption template for EMIT is loaded automatically from:
    https://data.source.coop/taco/methaneset/ch4_emit.safetensors
"""

import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Any, Literal

import torch
import torch.nn.functional as F
from pydantic import BaseModel, Field
from safetensors.torch import load_file
from tqdm import tqdm


# Constants --------------------

SAFETENSORS_URL  = "https://data.source.coop/taco/methaneset/ch4_emit.safetensors"
SAFETENSORS_PATH = Path(__file__).resolve().parent / "ch4_emit.safetensors"
WAVELENGTH_RANGE = (2122, 2488)
EPSILON          = 1e-9
NODATA           = -9999


# Result container --------------------

@dataclass
class RetrievalResult:
    """Output of MethaneRetrieval.retrieve().

    Supports both named access and tuple unpacking:
        result = eng.retrieve(rad, config)
        result.mf, result.albedo, result.metadata

        mf, albedo, meta = eng.retrieve(rad, config)
        mf, albedo, _    = eng.retrieve(rad, config)
    """
    mf: torch.Tensor
    albedo: torch.Tensor
    metadata: dict[str, Any] = field(default_factory=dict)

    def __iter__(self):
        return iter((self.mf, self.albedo, self.metadata))

    def __getitem__(self, idx):
        return (self.mf, self.albedo, self.metadata)[idx]

    def __len__(self):
        return 3


# Config models --------------------

class BaseRetrievalConfig(BaseModel):
    """Parameters shared by all retrieval methods."""

    column_step: Annotated[int, Field(ge=1, description=(
        "Number of crosstrack columns processed together per engine call. "
        "column_step=1 is physically correct for push-broom sensors. "
        "Ignored when batch_size is set (forced to 1)."
    ))] = 1

    batch_size: Annotated[int | None, Field(ge=1, description=(
        "Number of independent columns processed in parallel. "
        "When set, column_step is forced to 1. "
        "NODATA pixels are filled with per-column mean and re-masked on output. "
        "None = legacy path with explicit NODATA filtering."
    ))] = None

    alpha: Annotated[float, Field(ge=0.0, le=1.0, description=(
        "Covariance diagonal regularisation weight. "
        "Small values (~1e-4) stabilise the Cholesky decomposition."
    ))] = 1e-4

    scaling: Annotated[float, Field(gt=0.0, description=(
        "Multiplicative factor applied to raw matched filter output. "
        "Default 1e5 maps EMIT radiance-scale retrievals to ~ppm*m."
    ))] = 1e5


class MFConfig(BaseRetrievalConfig):
    """Basic matched filter (no albedo correction)."""
    pass


class RMFConfig(BaseRetrievalConfig):
    """Albedo-corrected single-pass matched filter."""
    pass


class MAG1CConfig(BaseRetrievalConfig):
    """Iterative sparse matched filter (reference implementation)."""

    num_iter: Annotated[int, Field(ge=1, description=(
        "Number of reweighted-L1 iterations."
    ))] = 30

    covariance_update_scaling: Annotated[float, Field(ge=0.0, le=1.0, description=(
        "Fraction of estimated CH4 signal subtracted at each step. "
        "1.0 = full removal, 0.0 = no update."
    ))] = 1.0


class MAG1CFastConfig(MAG1CConfig):
    """Rank-2 accelerated iterative filter."""

    tol: Annotated[float | None, Field(ge=0.0, description=(
        "Early stopping on max per-column relative mf change. "
        "None = disabled, runs exactly num_iter steps."
    ))] = None

    outer_mode: Annotated[Literal["auto", "bmm", "einsum"], Field(description=(
        "How to compute the rank-2 correction. "
        "'auto' picks bmm when N < 2000 or device is CPU."
    ))] = "auto"


RetrievalConfig = MFConfig | RMFConfig | MAG1CConfig | MAG1CFastConfig

_DISPATCH: dict[type, str] = {
    MFConfig:        "mf",
    RMFConfig:       "rmf",
    MAG1CConfig:     "mag1c",
    MAG1CFastConfig: "mag1c_fast",
}


# Template helpers --------------------

def _download_template(url: str, dest: Path) -> None:
    print(f"Downloading CH4 template to {dest} ...")
    dest.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, dest)
    print("Download complete.")


def _load_template(path, wavelength_range, device, dtype):
    if not path.exists():
        _download_template(SAFETENSORS_URL, path)
    tensors   = load_file(str(path), device=device)
    centers   = tensors["centers"]
    template  = tensors["template"]
    band_mask = (centers >= wavelength_range[0]) & (centers <= wavelength_range[1])
    return template[band_mask].to(dtype), band_mask.cpu()


# NODATA helpers --------------------

def _fill_nodata(rad):
    """Replace NODATA pixels with per-column mean. Fully vectorized.

    Returns:
        filled:      Clone with NODATA replaced by per-column mean.
        nodata_mask: Bool (D, C), True where any band was NODATA.
        empty_cols:  Bool (C,), True for entirely-NODATA columns.
    """
    filled      = rad.clone()
    nodata_mask = ((filled == NODATA) | torch.isnan(filled)).any(dim=-1)
    filled[nodata_mask] = 0.0

    counts     = (~nodata_mask).sum(dim=0)
    empty_cols = counts == 0

    col_sum  = filled.sum(dim=0, keepdim=True)
    col_mean = col_sum / counts.unsqueeze(0).clamp(min=1).unsqueeze(-1)
    filled[nodata_mask] = col_mean.expand_as(filled)[nodata_mask]

    return filled, nodata_mask, empty_cols


# Retrieval engine --------------------

class MethaneRetrieval:
    """Methane column enhancement retrieval engine."""

    def __init__(
        self,
        device: str | None = None,
        dtype: torch.dtype = torch.float32,
        wavelength_range: tuple[float, float] = WAVELENGTH_RANGE,
        safetensor_path: Path = SAFETENSORS_PATH,
        alpha: float = 1e-4,
        num_iter: int = 30,
    ):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if dtype not in (torch.float32, torch.float64):
            raise ValueError("dtype must be torch.float32 or torch.float64")

        self.device   = device
        self.dtype    = dtype
        self.alpha    = alpha
        self.num_iter = num_iter
        self.template, self.band_mask = _load_template(
            safetensor_path, wavelength_range, device, dtype
        )

    # Internal helpers --------------------

    def _prepare(self, x):
        x = x.to(device=self.device, dtype=self.dtype)
        n_expected = int(self.band_mask.sum())
        if x.shape[-1] != n_expected:
            raise ValueError(
                f"x has {x.shape[-1]} bands but engine expects {n_expected}."
            )
        return x

    def _regularize(self, C, alpha):
        return C.lerp_(torch.diag_embed(torch.diagonal(C, dim1=-2, dim2=-1)), alpha)

    def _covariance(self, m_mu, N, alpha):
        C = torch.bmm(m_mu.transpose(1, 2), m_mu) / N
        return self._regularize(C, alpha)

    def _outer_correction(self, C_u, A, tp, beta, N, mode):
        if mode == "auto":
            mode = "bmm" if self.device == "cpu" or N < 2000 else "einsum"
        if mode == "bmm":
            P = torch.stack([A,   tp], dim=2)
            Q = torch.stack([-tp, beta.unsqueeze(1) * tp - A], dim=2)
            return C_u + torch.bmm(P, Q.transpose(1, 2))
        return (C_u
                - torch.einsum("bi,bj->bij", A,  tp)
                - torch.einsum("bi,bj->bij", tp, A)
                + beta[:, None, None] * torch.einsum("bi,bj->bij", tp, tp))

    def _core_filter(self, x, alpha):
        """Shared MF/RMF core: mean, covariance, Cholesky solve, albedo.

        Returns (x_mu, Cit, norm, R, mu, target).
        """
        N = x.shape[1]
        t = self.template.unsqueeze(0).unsqueeze(0)

        mu     = x.mean(dim=1, keepdim=True)
        target = t * mu
        x_mu   = x - mu

        C     = self._covariance(x_mu, N, alpha)
        cholC = torch.linalg.cholesky(C)
        Cit   = torch.cholesky_solve(target.transpose(1, 2), cholC)
        norm  = torch.bmm(target, Cit)
        R     = torch.bmm(x, mu.transpose(1, 2)) / torch.bmm(mu, mu.transpose(1, 2))

        return x_mu, Cit, norm, R, mu, target

    # Retrieval methods --------------------

    @torch.no_grad()
    def mf(self, x, alpha=None, scaling=1e5):
        """Basic matched filter (no albedo correction)."""
        x     = self._prepare(x)
        alpha = alpha if alpha is not None else self.alpha

        x_mu, Cit, norm, _, _, _ = self._core_filter(x, alpha)

        R  = torch.ones(x.shape[0], x.shape[1], 1,
                        device=self.device, dtype=self.dtype)
        mf = torch.bmm(x_mu, Cit) / norm.clamp_(min=EPSILON)
        return F.relu(mf) * scaling, R

    @torch.no_grad()
    def rmf(self, x, alpha=None, scaling=1e5):
        """Albedo-corrected single-pass matched filter."""
        x     = self._prepare(x)
        alpha = alpha if alpha is not None else self.alpha

        x_mu, Cit, norm, R, _, _ = self._core_filter(x, alpha)

        mf = torch.bmm(x_mu, Cit) / (R * norm.clamp_(min=EPSILON))
        return F.relu(mf) * scaling, R

    @torch.no_grad()
    def mag1c(self, x, num_iter=None, alpha=None,
              covariance_update_scaling=1.0, scaling=1e5):
        """Iterative sparse matched filter (reference MAG1C)."""
        x        = self._prepare(x)
        num_iter = num_iter if num_iter is not None else self.num_iter
        alpha    = alpha if alpha is not None else self.alpha
        N        = x.shape[1]
        t        = self.template.unsqueeze(0).unsqueeze(0)

        mf, R  = self.rmf(x, alpha=alpha, scaling=1.0)
        mu     = x.mean(dim=1, keepdim=True)
        target = t * mu

        for _ in range(num_iter):
            modx    = x - covariance_update_scaling * R * mf * target
            mu      = modx.mean(dim=1, keepdim=True)
            target  = t * mu
            modx_mu = modx - mu
            x_mu    = x - mu

            C     = self._covariance(modx_mu, N, alpha)
            cholC = torch.linalg.cholesky(C)
            Cit   = torch.cholesky_solve(target.transpose(1, 2), cholC)

            reg  = 1.0 / (R * (mf + EPSILON))
            norm = torch.bmm(target, Cit).clamp_(min=EPSILON)
            mf   = (torch.bmm(x_mu, Cit) - reg) / (R * norm)
            F.relu_(mf)

        return mf * scaling, R

    @torch.no_grad()
    def mag1c_fast(self, x, num_iter=None, alpha=None,
                   covariance_update_scaling=1.0, tol=None,
                   outer_mode="auto", scaling=1e5, _run_info=None):
        """Rank-2 accelerated iterative matched filter.

        When _run_info is a dict, convergence diagnostics are written into it:
            iterations_run, converged, final_rel_change.
        """
        x        = self._prepare(x)
        num_iter = num_iter if num_iter is not None else self.num_iter
        alpha    = alpha if alpha is not None else self.alpha
        N        = x.shape[1]
        t        = self.template.unsqueeze(0).unsqueeze(0)

        # base covariance -- O(Ns^2), paid once
        mu_src = x.mean(dim=1, keepdim=True)
        u      = x - mu_src
        C_u    = torch.bmm(u.transpose(1, 2), u) / N

        # init with single-pass RMF (raw scale)
        mf, R  = self.rmf(x, alpha=alpha, scaling=1.0)
        target = t * mu_src

        iters_run   = num_iter
        converged   = False
        rel_per_col = None

        for k in range(num_iter):
            d     = covariance_update_scaling * R * mf
            d_bar = d.mean(dim=1, keepdim=True)
            delta = d - d_bar

            # rank-2 correction -- O(Ns) + O(s^2)
            tp   = target.squeeze(1)
            A    = torch.bmm(delta.transpose(1, 2), u).squeeze(1) / N
            beta = (delta * delta).sum(dim=1).squeeze(-1) / N

            C = self._outer_correction(C_u, A, tp, beta, N, outer_mode)
            C = self._regularize(C, alpha)

            # mean update -- O(s), shift computed before target update
            shift  = d_bar * target
            mu     = mu_src - shift
            target = t * mu

            cholC = torch.linalg.cholesky(C)
            Cit   = torch.cholesky_solve(target.transpose(1, 2), cholC)

            reg  = 1.0 / (R * (mf + EPSILON))
            norm = torch.bmm(target, Cit).clamp_(min=EPSILON)

            mf_prev = mf
            mf = (torch.bmm(u + shift, Cit) - reg) / (R * norm)
            F.relu_(mf)

            if tol is not None:
                rel_per_col = (mf - mf_prev).norm(dim=1) / (mf.norm(dim=1) + EPSILON)
                if rel_per_col.max() < tol:
                    iters_run = k + 1
                    converged = True
                    break

        if _run_info is not None:
            _run_info["iterations_run"]   = iters_run
            _run_info["converged"]        = converged
            if rel_per_col is not None:
                _run_info["final_rel_change"] = rel_per_col.detach()

        return mf * scaling, R

    # retrieve() public API --------------------

    def retrieve(self, rad, config, display_pbar=True):
        """Run retrieval over a full scene.

        Args:
            rad:    (downtrack, crosstrack, bands) radiance tensor.
            config: MFConfig | RMFConfig | MAG1CConfig | MAG1CFastConfig.

        Returns:
            RetrievalResult -- supports tuple unpacking: mf, albedo, meta.
        """
        if not isinstance(rad, torch.Tensor):
            raise TypeError(f"rad must be a torch.Tensor, got {type(rad).__name__}")

        method_name = _DISPATCH[type(config)]
        method      = getattr(self, method_name)

        base_keys     = set(BaseRetrievalConfig.model_fields)
        method_kwargs = {k: v for k, v in config.model_dump().items() if k not in base_keys}
        method_kwargs["scaling"] = config.scaling

        if config.batch_size is not None:
            return self._retrieve_batched(
                rad, method, method_kwargs, config.batch_size,
                display_pbar, method_name
            )
        return self._retrieve_legacy(
            rad, method, method_kwargs, config.column_step,
            display_pbar, method_name
        )

    # Batched path --------------------

    def _retrieve_batched(self, rad, method, method_kwargs, batch_size,
                          display_pbar, method_name):
        rad = rad.to(device=self.device, dtype=self.dtype)
        D, C, s = rad.shape

        filled, nodata_mask, empty_cols = _fill_nodata(rad)
        data = filled.permute(1, 0, 2)

        mf_out     = torch.full((C, D), NODATA, device=self.device, dtype=self.dtype)
        albedo_out = torch.full((C, D), NODATA, device=self.device, dtype=self.dtype)

        valid_counts    = (~nodata_mask).sum(dim=0)
        batch_run_infos = []

        for col_start in tqdm(range(0, C, batch_size),
                              desc=f"retrieve [{method_name}]",
                              disable=not display_pbar):
            col_end = min(col_start + batch_size, C)

            if empty_cols[col_start:col_end].all():
                continue

            x = data[col_start:col_end]

            run_info = {} if method_name == "mag1c_fast" else None
            if run_info is not None:
                mf, R = method(x, **method_kwargs, _run_info=run_info)
                batch_run_infos.append(run_info)
            else:
                mf, R = method(x, **method_kwargs)

            mf_out    [col_start:col_end] = mf[:, :, 0]
            albedo_out[col_start:col_end] = R [:, :, 0]

        mf_out     = mf_out.T
        albedo_out = albedo_out.T
        mf_out    [nodata_mask] = NODATA
        albedo_out[nodata_mask] = NODATA

        n_bands  = int(self.band_mask.sum())
        metadata = {
            "method":       method_name,
            "n_downtrack":  D,
            "n_crosstrack": C,
            "n_bands":      n_bands,
            "device":       self.device,
            "dtype":        str(self.dtype),
            "valid_pixel_counts":      valid_counts.cpu(),
            "empty_columns":           empty_cols.nonzero(as_tuple=False).squeeze(-1).cpu(),
            "underdetermined_columns": (valid_counts < n_bands).nonzero(as_tuple=False).squeeze(-1).cpu(),
        }
        if batch_run_infos:
            metadata["batch_convergence"] = batch_run_infos

        return RetrievalResult(mf=mf_out, albedo=albedo_out, metadata=metadata)

    # Legacy path --------------------

    def _retrieve_legacy(self, rad, method, method_kwargs, column_step,
                         display_pbar, method_name):
        rad = rad.to(device=self.device, dtype=self.dtype)
        D, C, _ = rad.shape

        mf_out     = torch.full((D, C), NODATA, device=self.device, dtype=self.dtype)
        albedo_out = torch.full((D, C), NODATA, device=self.device, dtype=self.dtype)

        for col_start in tqdm(range(0, C, column_step),
                              desc=f"retrieve [{method_name}]",
                              disable=not display_pbar):
            col_end    = min(col_start + column_step, C)
            slice_data = rad[:, col_start:col_end, :]
            valid      = (slice_data != NODATA).all(dim=-1)

            if not valid.any():
                continue

            x     = slice_data[valid].unsqueeze(0)
            mf, R = method(x, **method_kwargs)

            mf_out    [:, col_start:col_end][valid] = mf[0, :, 0]
            albedo_out[:, col_start:col_end][valid] = R [0, :, 0]

        metadata = {
            "method":       method_name,
            "n_downtrack":  D,
            "n_crosstrack": C,
            "n_bands":      int(self.band_mask.sum()),
            "device":       self.device,
            "dtype":        str(self.dtype),
        }
        return RetrievalResult(mf=mf_out, albedo=albedo_out, metadata=metadata)
