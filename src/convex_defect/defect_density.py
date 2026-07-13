"""Frequency- and gauge-dependent convex defect density.

Implements the refined equation set:

    ρ(x, f, κ, s) = A(f, κ) · exp(−x² / σ(f, κ)²) · s^{−δ(κ)}

with σ, A, and δ carrying frequency and |κ − κ*| detuning. Topological
opacity τ is the integral of ρ over a finite scale window [s_min, s_max].

Conceptual links
----------------
- Holonomy gap / κ*  — mystery residual κ sweeps, oam_flux.LatticeConstants
- Turbulence screens — flux_trajectoid.propagation (structured alternative
  to pure Kolmogorov phase screens)
- OAM fidelity       — high-f modes couple to fine-scale s (quicksand)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

# Residual R = φ² + e² − π² (same family as oam_flux.constants)
_PHI = (1.0 + 5.0**0.5) / 2.0
_RESIDUAL_R = _PHI**2 + math.e**2 - math.pi**2

# κ that nulls B(κ) = π²(e/π − κ) against residual R  →  ≈ 0.8513
KAPPA_STAR_DEFAULT: float = math.e / math.pi - _RESIDUAL_R / math.pi**2


@dataclass(frozen=True)
class DefectParams:
    """Tunable constants for the convex defect model.

    Defaults match the refined equation set discussed with mystery /
    flux_trajectoid infrastructure (κ* ≈ 0.8513).
    """

    kappa_star: float = KAPPA_STAR_DEFAULT
    sigma0: float = 1.0
    f0: float = 1.0
    A0: float = 1.0
    alpha: float = 0.5  # frequency → width  σ ∝ (f0/f)^α
    beta: float = 0.25  # frequency → amp   A ∝ (f/f0)^β
    gamma: float = 1.0  # κ-detuning → width
    mu: float = 0.5  # κ-detuning → amplitude
    nu: float = 0.3  # κ-detuning → fractal exponent
    delta0: float = 0.2  # baseline fractal exponent
    s_min: float = 1e-3  # scale window for opacity integral
    s_max: float = 1.0
    eps: float = 1e-30  # numerical floor for σ² in Gaussian core

    def __post_init__(self) -> None:
        if self.sigma0 <= 0 or self.f0 <= 0 or self.A0 < 0:
            raise ValueError("require sigma0 > 0, f0 > 0, A0 >= 0")
        if self.s_min <= 0 or self.s_max <= 0 or self.s_min >= self.s_max:
            raise ValueError("require 0 < s_min < s_max")
        if self.eps <= 0:
            raise ValueError("eps must be positive")
        if self.delta0 < 0 or self.nu < 0:
            raise ValueError("fractal exponents delta0, nu should be non-negative")
        if self.gamma < 0 or self.mu < 0:
            raise ValueError("detuning strengths gamma, mu should be non-negative")

    def with_updates(self, **kwargs: Any) -> DefectParams:
        """Return a copy with selected fields replaced."""
        return replace(self, **kwargs)


def _as_array(x: ArrayLike) -> NDArray[np.floating]:
    return np.asarray(x, dtype=float)


def _detune(kappa: ArrayLike, kappa_star: float) -> NDArray[np.floating]:
    """Absolute gauge detuning |κ − κ*| (broadcastable)."""
    return np.abs(_as_array(kappa) - kappa_star)


def sigma_width(
    f: ArrayLike,
    kappa: ArrayLike = KAPPA_STAR_DEFAULT,
    *,
    params: DefectParams | None = None,
) -> NDArray[np.floating]:
    """Defect core width σ(f, κ).

    σ(f, κ) = σ0 · (f0 / f)^α · (1 + γ |κ − κ*|)
    """
    p = params or DefectParams()
    f_arr = _as_array(f)
    if np.any(f_arr <= 0):
        raise ValueError("frequency f must be positive")
    det = _detune(kappa, p.kappa_star)
    return p.sigma0 * (p.f0 / f_arr) ** p.alpha * (1.0 + p.gamma * det)


def amplitude(
    f: ArrayLike,
    kappa: ArrayLike = KAPPA_STAR_DEFAULT,
    *,
    params: DefectParams | None = None,
) -> NDArray[np.floating]:
    """Defect amplitude A(f, κ).

    A(f, κ) = A0 · (f / f0)^β · (1 + μ |κ − κ*|)
    """
    p = params or DefectParams()
    f_arr = _as_array(f)
    if np.any(f_arr <= 0):
        raise ValueError("frequency f must be positive")
    det = _detune(kappa, p.kappa_star)
    return p.A0 * (f_arr / p.f0) ** p.beta * (1.0 + p.mu * det)


def fractal_exponent(
    kappa: ArrayLike = KAPPA_STAR_DEFAULT,
    *,
    params: DefectParams | None = None,
) -> NDArray[np.floating]:
    """Fractal scaling exponent δ(κ) = δ0 + ν |κ − κ*|."""
    p = params or DefectParams()
    return p.delta0 + p.nu * _detune(kappa, p.kappa_star)


def scale_factor(
    s: ArrayLike,
    kappa: ArrayLike = KAPPA_STAR_DEFAULT,
    *,
    params: DefectParams | None = None,
) -> NDArray[np.floating]:
    """Multi-scale weight s^{−δ(κ)}."""
    s_arr = _as_array(s)
    if np.any(s_arr <= 0):
        raise ValueError("scale s must be positive")
    delta = fractal_exponent(kappa, params=params)
    return s_arr ** (-delta)


def defect_density(
    x: ArrayLike,
    f: ArrayLike,
    kappa: ArrayLike = KAPPA_STAR_DEFAULT,
    s: ArrayLike = 1.0,
    *,
    params: DefectParams | None = None,
) -> NDArray[np.floating]:
    """Convex defect density ρ(x, f, κ, s).

    ρ = A(f, κ) · exp(−x² / σ(f, κ)²) · s^{−δ(κ)}

    All arguments broadcast with numpy rules.
    """
    p = params or DefectParams()
    x_arr = _as_array(x)
    sig = sigma_width(f, kappa, params=p)
    amp = amplitude(f, kappa, params=p)
    frac = scale_factor(s, kappa, params=p)
    # guard against zero width (should not occur for f > 0, finite params)
    gauss = np.exp(-(x_arr**2) / (sig**2 + p.eps))
    return amp * gauss * frac


def scale_integral(
    kappa: ArrayLike = KAPPA_STAR_DEFAULT,
    *,
    params: DefectParams | None = None,
    s_min: float | None = None,
    s_max: float | None = None,
) -> NDArray[np.floating]:
    """Analytic ∫_{s_min}^{s_max} s^{−δ(κ)} ds.

    For δ ≠ 1: (s_max^{1−δ} − s_min^{1−δ}) / (1 − δ)
    For δ = 1: ln(s_max / s_min)
    """
    p = params or DefectParams()
    lo = float(p.s_min if s_min is None else s_min)
    hi = float(p.s_max if s_max is None else s_max)
    if lo <= 0 or hi <= 0 or hi <= lo:
        raise ValueError("require 0 < s_min < s_max")

    delta = fractal_exponent(kappa, params=p)
    delta = _as_array(delta)
    out = np.empty_like(delta, dtype=float)

    near_one = np.isclose(delta, 1.0, rtol=0.0, atol=1e-12)
    if np.any(~near_one):
        d = delta[~near_one]
        out[~near_one] = (hi ** (1.0 - d) - lo ** (1.0 - d)) / (1.0 - d)
    if np.any(near_one):
        out[near_one] = math.log(hi / lo)
    return out


def opacity(
    x: ArrayLike,
    f: ArrayLike,
    kappa: ArrayLike = KAPPA_STAR_DEFAULT,
    *,
    params: DefectParams | None = None,
    s_min: float | None = None,
    s_max: float | None = None,
) -> NDArray[np.floating]:
    """Topological opacity τ(x, f, κ) = ∫ ρ ds over the scale window.

    τ = A(f, κ) · exp(−x² / σ²) · I_δ(s_min, s_max)
    """
    p = params or DefectParams()
    x_arr = _as_array(x)
    sig = sigma_width(f, kappa, params=p)
    amp = amplitude(f, kappa, params=p)
    gauss = np.exp(-(x_arr**2) / (sig**2 + p.eps))
    i_delta = scale_integral(kappa, params=p, s_min=s_min, s_max=s_max)
    return amp * gauss * i_delta


def gaussian_area_opacity(
    f: ArrayLike,
    kappa: ArrayLike = KAPPA_STAR_DEFAULT,
    s: ArrayLike = 1.0,
    *,
    params: DefectParams | None = None,
) -> NDArray[np.floating]:
    """Misalignment-integrated opacity at fixed scale: ∫ ρ dx.

    τ_x = A(f, κ) · σ(f, κ) · √π · s^{−δ(κ)}
    """
    p = params or DefectParams()
    sig = sigma_width(f, kappa, params=p)
    amp = amplitude(f, kappa, params=p)
    frac = scale_factor(s, kappa, params=p)
    return amp * sig * math.sqrt(math.pi) * frac


def holonomy_gap(
    kappa: ArrayLike,
    *,
    params: DefectParams | None = None,
) -> NDArray[np.floating]:
    """Skyrme-style holonomy gap proxy B(κ) = π² (e/π − κ).

    Related (not identical) to residual-corrected κ* used for λ(κ).
    Provided for conceptual mapping to mystery residual_κ sweeps.
    """
    del params  # reserved for future residual-aware B(κ)
    return (math.pi**2) * (math.e / math.pi - _as_array(kappa))


@dataclass
class DefectModel:
    """Convenience object binding :class:`DefectParams` to density helpers.

    Example
    -------
    >>> m = DefectModel()
    >>> float(m.rho(0.0, 1.0, m.params.kappa_star, 1.0))
    1.0  # approximately, for default A0 and s=1
    """

    params: DefectParams

    def __init__(self, params: DefectParams | None = None, **overrides: Any) -> None:
        base = params or DefectParams()
        self.params = base.with_updates(**overrides) if overrides else base

    @property
    def kappa_star(self) -> float:
        return self.params.kappa_star

    def sigma(self, f: ArrayLike, kappa: ArrayLike | None = None) -> NDArray[np.floating]:
        k = self.params.kappa_star if kappa is None else kappa
        return sigma_width(f, k, params=self.params)

    def A(self, f: ArrayLike, kappa: ArrayLike | None = None) -> NDArray[np.floating]:
        k = self.params.kappa_star if kappa is None else kappa
        return amplitude(f, k, params=self.params)

    def delta(self, kappa: ArrayLike | None = None) -> NDArray[np.floating]:
        k = self.params.kappa_star if kappa is None else kappa
        return fractal_exponent(k, params=self.params)

    def rho(
        self,
        x: ArrayLike,
        f: ArrayLike,
        kappa: ArrayLike | None = None,
        s: ArrayLike = 1.0,
    ) -> NDArray[np.floating]:
        k = self.params.kappa_star if kappa is None else kappa
        return defect_density(x, f, k, s, params=self.params)

    def opacity(
        self,
        x: ArrayLike,
        f: ArrayLike,
        kappa: ArrayLike | None = None,
        *,
        s_min: float | None = None,
        s_max: float | None = None,
    ) -> NDArray[np.floating]:
        k = self.params.kappa_star if kappa is None else kappa
        return opacity(x, f, k, params=self.params, s_min=s_min, s_max=s_max)

    def gaussian_area(
        self,
        f: ArrayLike,
        kappa: ArrayLike | None = None,
        s: ArrayLike = 1.0,
    ) -> NDArray[np.floating]:
        k = self.params.kappa_star if kappa is None else kappa
        return gaussian_area_opacity(f, k, s, params=self.params)

    def holonomy_gap(self, kappa: ArrayLike) -> NDArray[np.floating]:
        return holonomy_gap(kappa, params=self.params)

    def profile(
        self,
        x: ArrayLike,
        f: ArrayLike,
        kappa: ArrayLike | None = None,
        s: ArrayLike = 1.0,
    ) -> dict[str, NDArray[np.floating] | float]:
        """Bundle ρ, σ, A, δ, τ for a parameter point (handy for demos)."""
        k = self.params.kappa_star if kappa is None else kappa
        return {
            "rho": self.rho(x, f, k, s),
            "sigma": self.sigma(f, k),
            "A": self.A(f, k),
            "delta": self.delta(k),
            "opacity": self.opacity(x, f, k),
            "kappa": np.asarray(k, dtype=float),
            "kappa_star": self.params.kappa_star,
        }
