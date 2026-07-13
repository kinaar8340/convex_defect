r"""Multi-scale convex defect field ρ(·, s).

Evolves a density distribution over discrete scale bins \(s_k\) instead of
evaluating instantaneous \(\rho(x,f,\kappa,s)\) at a single \(s\).

Two layouts
-----------
1. **Spectral only** — shape ``(n_scales,)``: global multi-scale state for
   pointer + holonomy demos (no spatial grid).
2. **Spatio-spectral** — shape ``(*grid_shape, n_scales)``: each spatial
   cell holds a full \(\rho(s)\) spectrum (phase-screen / turbulence use).

Seeding
-------
Each bin is initialized from the static formula:

    ρ_k = A(f,κ) · exp(−x²/σ²) · s_k^{−δ(κ)}

Evolution (per bin, independent by default)
-------------------------------------------
Discrete survival **without** re-multiplying \(s^{−δ}\) each step (the fractal
weight is already baked into the bin state at seed). That matches the
intended multi-scale reading: each scale carries its own density and heals
via λ(κ) and the misalignment clamp.

Optional **scale coupling** mixes neighboring bins (weak cascade / roughness
redistribution across scales).

Opacity / screens
-----------------
    τ ≈ Σ_k ρ_k · w_k     with bin weights w_k from linear or log measure

Phase screen (spatial):

    screen_ij = Σ_k ρ_ij,k · w_k
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .defect_density import (
    KAPPA_STAR_DEFAULT,
    DefectParams,
    defect_density,
    fractal_exponent,
)
from .holonomy_accumulator import HolonomyParams, gamma_H
from .relaxation_dynamics import (
    RelaxationParams,
    discrete_step,
    eta_source,
    lambda_rate,
)

Measure = Literal["linear", "log"]


# ---------------------------------------------------------------------------
# Scale bins
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ScaleBins:
    """Discrete scale axis with integration weights.

    Parameters
    ----------
    s :
        Positive scale centers, shape (n_scales,). Prefer log-spaced.
    weights :
        Quadrature weights w_k for ∫ g(s) ds ≈ Σ g_k w_k.
    measure :
        How bins were built (documentation / diagnostics).
    """

    s: NDArray[np.floating]
    weights: NDArray[np.floating]
    measure: Measure = "log"

    def __post_init__(self) -> None:
        s = np.asarray(self.s, dtype=float)
        w = np.asarray(self.weights, dtype=float)
        if s.ndim != 1 or w.ndim != 1 or s.shape != w.shape:
            raise ValueError("s and weights must be 1-D and same length")
        if np.any(s <= 0):
            raise ValueError("all scale centers must be positive")
        if np.any(w < 0):
            raise ValueError("weights must be non-negative")
        object.__setattr__(self, "s", s)
        object.__setattr__(self, "weights", w)

    @property
    def n_scales(self) -> int:
        return int(self.s.size)

    @classmethod
    def logspace(
        cls,
        s_min: float = 1e-3,
        s_max: float = 1.0,
        n_scales: int = 16,
    ) -> ScaleBins:
        """Log-spaced bins with trapezoid weights in log-s (ds = s d(ln s))."""
        if not (0 < s_min < s_max) or n_scales < 2:
            raise ValueError("require 0 < s_min < s_max and n_scales >= 2")
        s = np.geomspace(s_min, s_max, n_scales)
        # ∫ f(s) ds with f = ρ; for log grid use mid-log spacing * s
        ln_s = np.log(s)
        dln = np.empty(n_scales)
        dln[0] = ln_s[1] - ln_s[0]
        dln[-1] = ln_s[-1] - ln_s[-2]
        dln[1:-1] = 0.5 * (ln_s[2:] - ln_s[:-2])
        weights = s * dln  # ds = s dln s
        return cls(s=s, weights=weights, measure="log")

    @classmethod
    def linspace(
        cls,
        s_min: float = 1e-3,
        s_max: float = 1.0,
        n_scales: int = 16,
    ) -> ScaleBins:
        if not (0 < s_min < s_max) or n_scales < 2:
            raise ValueError("require 0 < s_min < s_max and n_scales >= 2")
        s = np.linspace(s_min, s_max, n_scales)
        ds = (s_max - s_min) / (n_scales - 1)
        weights = np.full(n_scales, ds)
        weights[0] *= 0.5
        weights[-1] *= 0.5
        return cls(s=s, weights=weights, measure="linear")

    @classmethod
    def from_params(
        cls,
        params: DefectParams | None = None,
        n_scales: int = 16,
        measure: Measure = "log",
    ) -> ScaleBins:
        p = params or DefectParams()
        if measure == "log":
            return cls.logspace(p.s_min, p.s_max, n_scales)
        return cls.linspace(p.s_min, p.s_max, n_scales)


@dataclass(frozen=True)
class MultiScaleParams:
    """Configuration for multi-scale field evolution."""

    n_scales: int = 16
    measure: Measure = "log"
    # scale coupling: mix with neighbors each step (0 = independent bins)
    scale_coupling: float = 0.0
    # when True, re-apply s^{-δ} in discrete_step (usually False for multi-scale)
    apply_scale_in_discrete: bool = False
    # continuous source injects into each bin weighted by s^{-δ}
    source_mode: Literal["uniform", "fractal"] = "fractal"

    def __post_init__(self) -> None:
        if self.n_scales < 2:
            raise ValueError("n_scales must be >= 2")
        if not (0.0 <= self.scale_coupling < 1.0):
            raise ValueError("scale_coupling must lie in [0, 1)")

    def with_updates(self, **kwargs: Any) -> MultiScaleParams:
        return replace(self, **kwargs)


# ---------------------------------------------------------------------------
# Field
# ---------------------------------------------------------------------------


@dataclass
class MultiScaleDefectField:
    """Dynamical multi-scale defect density field.

    Attributes
    ----------
    bins :
        Scale centers and quadrature weights.
    rho :
        Density state. Shape ``(n_scales,)`` or ``(*spatial, n_scales)``.
    """

    bins: ScaleBins
    rho: NDArray[np.floating]
    defect_params: DefectParams = field(default_factory=DefectParams)
    relaxation_params: RelaxationParams = field(default_factory=RelaxationParams)
    multi_params: MultiScaleParams = field(default_factory=MultiScaleParams)
    f: float = 1.0
    kappa: float = KAPPA_STAR_DEFAULT
    t: float = 0.0

    def __post_init__(self) -> None:
        self.rho = np.asarray(self.rho, dtype=float)
        if self.rho.shape[-1] != self.bins.n_scales:
            raise ValueError(
                f"rho last axis must be n_scales={self.bins.n_scales}, "
                f"got shape {self.rho.shape}"
            )
        # multi-scale evolution should not re-multiply fractal each step by default
        if self.multi_params.apply_scale_in_discrete != self.relaxation_params.apply_scale_in_discrete:
            self.relaxation_params = self.relaxation_params.with_updates(
                apply_scale_in_discrete=self.multi_params.apply_scale_in_discrete
            )

    # --- constructors -------------------------------------------------------

    @classmethod
    def from_misalignment(
        cls,
        x: ArrayLike,
        f: float = 1.0,
        kappa: float | None = None,
        *,
        bins: ScaleBins | None = None,
        defect_params: DefectParams | None = None,
        relaxation_params: RelaxationParams | None = None,
        multi_params: MultiScaleParams | None = None,
        spatial_shape: tuple[int, ...] | None = None,
        grid: ArrayLike | None = None,
    ) -> MultiScaleDefectField:
        """Seed ρ from the static density formula at each scale bin.

        Parameters
        ----------
        x :
            Global scalar misalignment (used when ``grid`` is None).
        grid :
            Optional local misalignment array; seeds shape ``(*grid.shape, n_s)``.
        spatial_shape :
            If set (and grid is None), seed a uniform field of that shape
            using scalar ``x``.
        """
        dp = defect_params or DefectParams()
        mp = multi_params or MultiScaleParams()
        rp = relaxation_params or RelaxationParams(
            apply_scale_in_discrete=mp.apply_scale_in_discrete
        )
        k = float(dp.kappa_star if kappa is None else kappa)
        b = bins or ScaleBins.from_params(dp, n_scales=mp.n_scales, measure=mp.measure)

        if grid is not None:
            g = np.asarray(grid, dtype=float)
            # rho[..., k] = density(g, f, k, s_k)
            parts = [
                defect_density(g, f, k, float(s), params=dp) for s in b.s
            ]
            rho = np.stack(parts, axis=-1)
        elif spatial_shape is not None:
            g = np.full(spatial_shape, float(x), dtype=float)
            parts = [
                defect_density(g, f, k, float(s), params=dp) for s in b.s
            ]
            rho = np.stack(parts, axis=-1)
        else:
            x0 = float(np.asarray(x, dtype=float))
            rho = np.array(
                [float(defect_density(x0, f, k, float(s), params=dp)) for s in b.s],
                dtype=float,
            )

        return cls(
            bins=b,
            rho=rho,
            defect_params=dp,
            relaxation_params=rp,
            multi_params=mp,
            f=float(f),
            kappa=k,
            t=0.0,
        )

    # --- properties ---------------------------------------------------------

    @property
    def n_scales(self) -> int:
        return self.bins.n_scales

    @property
    def spatial_shape(self) -> tuple[int, ...]:
        if self.rho.ndim == 1:
            return ()
        return tuple(self.rho.shape[:-1])

    @property
    def is_spatial(self) -> bool:
        return self.rho.ndim > 1

    # --- reduce over scales -------------------------------------------------

    def integrate_scales(self, field: NDArray[np.floating] | None = None) -> NDArray[np.floating]:
        """Σ_k field[..., k] · w_k  → opacity-like reduction."""
        arr = self.rho if field is None else np.asarray(field, dtype=float)
        w = self.bins.weights
        return np.tensordot(arr, w, axes=([-1], [0]))

    def opacity(self) -> NDArray[np.floating] | float:
        """Topological opacity τ from current multi-scale state."""
        out = self.integrate_scales()
        if out.ndim == 0:
            return float(out)
        return out

    def mean_density(self) -> float:
        """Scalar mean of integrated opacity (or spectral integral)."""
        tau = self.opacity()
        if isinstance(tau, float):
            return tau
        return float(np.mean(tau))

    def spectrum(self, spatial_reduce: str = "mean") -> NDArray[np.floating]:
        """Return ρ(s) spectrum, averaging over space if needed."""
        if not self.is_spatial:
            return self.rho.copy()
        axes = tuple(range(self.rho.ndim - 1))
        if spatial_reduce == "mean":
            return np.mean(self.rho, axis=axes)
        if spatial_reduce == "max":
            return np.max(self.rho, axis=axes)
        if spatial_reduce == "sum":
            return np.sum(self.rho, axis=axes)
        raise ValueError("spatial_reduce must be mean|max|sum")

    def phase_screen(self, gain: float = 1.0) -> NDArray[np.floating]:
        """Integrated multi-scale screen (spatial only): gain · Σ ρ_k w_k."""
        if not self.is_spatial:
            raise ValueError("phase_screen requires a spatial multi-scale field")
        return gain * np.asarray(self.integrate_scales(), dtype=float)

    def holonomy_rate(
        self,
        x: ArrayLike | None = None,
        *,
        holonomy_params: HolonomyParams | None = None,
    ) -> float:
        """Scale-integrated holonomy rate (global / mean over space).

        Uses the **evolved** multi-scale ρ (not the instantaneous formula).
        Extra fractal weight (s^{−δ})^m is applied on top of the stored ρ,
        matching HolonomyParams.holonomy_fractal_multiplier.
        """
        hp = holonomy_params or HolonomyParams()
        g = float(gamma_H(self.f, holonomy_params=hp, defect_params=self.defect_params))
        m = hp.holonomy_fractal_multiplier
        delta = float(fractal_exponent(self.kappa, params=self.defect_params))
        # per-bin extra weight
        if m == 0.0:
            extra = np.ones(self.n_scales)
        else:
            extra = self.bins.s ** (-delta * m)
        # ρ weighted: mean over space of Σ ρ_k * extra_k * w_k
        weighted = self.rho * extra  # broadcast on last axis
        integrated = self.integrate_scales(weighted)
        if isinstance(integrated, float) or np.ndim(integrated) == 0:
            return g * float(integrated)
        return g * float(np.mean(integrated))

    # --- evolution ----------------------------------------------------------

    def _apply_scale_coupling(self) -> None:
        """Weak diffusion along the scale axis (neighbor mix)."""
        c = self.multi_params.scale_coupling
        if c <= 0.0 or self.n_scales < 2:
            return
        r = self.rho
        # pad edges with edge values
        left = np.concatenate([r[..., :1], r[..., :-1]], axis=-1)
        right = np.concatenate([r[..., 1:], r[..., -1:]], axis=-1)
        neighbor = 0.5 * (left + right)
        self.rho = (1.0 - c) * r + c * neighbor

    def step_discrete(
        self,
        x: ArrayLike,
        dt: float = 1.0,
        *,
        grid: ArrayLike | None = None,
    ) -> NDArray[np.floating]:
        """One discrete survival step for every scale bin.

        Parameters
        ----------
        x :
            Global misalignment (scalar) when not using ``grid``.
        grid :
            Local misalignment field matching spatial_shape; if provided,
            each cell evolves with its own x_ij.
        """
        dp = self.defect_params
        rp = self.relaxation_params
        f, k = self.f, self.kappa

        if grid is not None:
            g = np.asarray(grid, dtype=float)
            if g.shape != self.spatial_shape:
                raise ValueError(
                    f"grid shape {g.shape} != spatial_shape {self.spatial_shape}"
                )
            # vectorized over scales: discrete_step broadcasts on x
            for i, s in enumerate(self.bins.s):
                self.rho[..., i] = discrete_step(
                    self.rho[..., i],
                    g,
                    f,
                    k,
                    float(s),
                    dt=dt,
                    defect_params=dp,
                    relaxation_params=rp,
                )
        else:
            x0 = float(np.asarray(x, dtype=float))
            for i, s in enumerate(self.bins.s):
                self.rho[..., i] = discrete_step(
                    self.rho[..., i],
                    x0,
                    f,
                    k,
                    float(s),
                    dt=dt,
                    defect_params=dp,
                    relaxation_params=rp,
                )

        self._apply_scale_coupling()
        self.t += dt
        return self.rho

    def step_euler(
        self,
        x: ArrayLike,
        dt: float = 0.01,
        *,
        grid: ArrayLike | None = None,
    ) -> NDArray[np.floating]:
        """Explicit Euler of continuous ODE per bin (+ optional fractal source weight).

        dρ_k/dt = −λ ρ_k + η x² · σ_k
        with σ_k = s_k^{−δ} (source_mode='fractal') or 1 (uniform).
        """
        dp = self.defect_params
        rp = self.relaxation_params
        k = self.kappa
        lam = float(lambda_rate(k, defect_params=dp, relaxation_params=rp))
        eta = float(eta_source(self.f, k, defect_params=dp, relaxation_params=rp))
        delta = float(fractal_exponent(k, params=dp))
        if self.multi_params.source_mode == "fractal":
            src_w = self.bins.s ** (-delta)
        else:
            src_w = np.ones(self.n_scales)

        if grid is not None:
            g = np.asarray(grid, dtype=float)
            if g.shape != self.spatial_shape:
                raise ValueError(
                    f"grid shape {g.shape} != spatial_shape {self.spatial_shape}"
                )
            x2 = g**2
            dr = -lam * self.rho + eta * x2[..., None] * src_w
            self.rho = np.maximum(0.0, self.rho + dt * dr)
        else:
            x0 = float(np.asarray(x, dtype=float))
            x2 = x0**2
            if self.rho.ndim == 1:
                dr = -lam * self.rho + eta * x2 * src_w
                self.rho = np.maximum(0.0, self.rho + dt * dr)
            else:
                dr = -lam * self.rho + eta * x2 * src_w
                self.rho = np.maximum(0.0, self.rho + dt * dr)

        self._apply_scale_coupling()
        self.t += dt
        return self.rho

    def reseed_from_misalignment(
        self,
        x: ArrayLike,
        *,
        grid: ArrayLike | None = None,
        blend: float = 1.0,
    ) -> None:
        """Blend field toward instantaneous static multi-scale density.

        ``blend=1`` replaces; ``blend∈(0,1)`` soft-resets (useful for driven channels).
        """
        if not (0.0 <= blend <= 1.0):
            raise ValueError("blend must lie in [0, 1]")
        kwargs: dict[str, Any] = dict(
            f=self.f,
            kappa=self.kappa,
            bins=self.bins,
            defect_params=self.defect_params,
            relaxation_params=self.relaxation_params,
            multi_params=self.multi_params,
        )
        if grid is not None:
            target = MultiScaleDefectField.from_misalignment(x, grid=grid, **kwargs)
        elif self.is_spatial:
            target = MultiScaleDefectField.from_misalignment(
                x, spatial_shape=self.spatial_shape, **kwargs
            )
        else:
            target = MultiScaleDefectField.from_misalignment(x, **kwargs)
        self.rho = (1.0 - blend) * self.rho + blend * target.rho

    def copy(self) -> MultiScaleDefectField:
        return MultiScaleDefectField(
            bins=self.bins,
            rho=self.rho.copy(),
            defect_params=self.defect_params,
            relaxation_params=self.relaxation_params,
            multi_params=self.multi_params,
            f=self.f,
            kappa=self.kappa,
            t=self.t,
        )


def multi_scale_phase_screen(
    grid: ArrayLike,
    f: float,
    kappa: float,
    *,
    bins: ScaleBins | None = None,
    defect_params: DefectParams | None = None,
    multi_params: MultiScaleParams | None = None,
    gain: float = 1.0,
) -> NDArray[np.floating]:
    """Instantaneous multi-scale screen (no dynamics): seed + integrate."""
    field = MultiScaleDefectField.from_misalignment(
        0.0,
        f=f,
        kappa=kappa,
        bins=bins,
        defect_params=defect_params,
        multi_params=multi_params,
        grid=grid,
    )
    return field.phase_screen(gain=gain)
