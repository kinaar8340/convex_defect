"""Fractal holonomy accumulation along geodesics / trajectories.

Implements:

    H(t) = H0 + ∫₀ᵗ γ_H(f) · ρ(x(t'), f, κ, s) · s^{−δ(κ)} dt'

Because ρ already carries one factor of s^{−δ}, the integrand scales as
s^{−2δ} when ``holonomy_fractal_multiplier=1`` (default). That double
fractal weight is intentional: finer scales imprint richer topological
memory under misalignment. If simulations later find this too aggressive,
lower ``holonomy_fractal_multiplier`` (e.g. 0.0 → single fractal only).

Conceptual links
----------------
- Trajectoid geodesics — flux_trajectoid shell rolling paths
- Holonomy gaps         — mystery residual_κ / Skyrme reductions
- OAM path memory       — oam_flux lattice twist accumulation
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .defect_density import (
    KAPPA_STAR_DEFAULT,
    DefectParams,
    defect_density,
    scale_factor,
)


@dataclass(frozen=True)
class HolonomyParams:
    """Coupling constants for holonomy accumulation.

    Parameters
    ----------
    gamma_H0 :
        Base holonomy coupling strength.
    gamma_H_beta :
        Frequency boost: γ_H(f) = gamma_H0 · (f / f0)^{gamma_H_beta}.
        Higher frequencies accumulate memory faster (quicksand).
    holonomy_fractal_multiplier :
        Extra power of s^{−δ} on top of the one already inside ρ.
        Default 1.0 → integrand ∝ ρ · s^{−δ} ∝ s^{−2δ}.
        Set to 0.0 to accumulate with ρ alone (single fractal).
    H0 :
        Initial holonomy.
    """

    gamma_H0: float = 1.0
    gamma_H_beta: float = 0.5
    holonomy_fractal_multiplier: float = 1.0
    H0: float = 0.0

    def __post_init__(self) -> None:
        if self.gamma_H0 < 0:
            raise ValueError("gamma_H0 should be non-negative")
        if self.holonomy_fractal_multiplier < 0:
            raise ValueError("holonomy_fractal_multiplier should be non-negative")

    def with_updates(self, **kwargs: Any) -> HolonomyParams:
        return replace(self, **kwargs)


def gamma_H(
    f: ArrayLike,
    *,
    holonomy_params: HolonomyParams | None = None,
    defect_params: DefectParams | None = None,
) -> NDArray[np.floating]:
    """Frequency-dependent holonomy coupling γ_H(f)."""
    hp = holonomy_params or HolonomyParams()
    dp = defect_params or DefectParams()
    f_arr = np.asarray(f, dtype=float)
    if np.any(f_arr <= 0):
        raise ValueError("frequency f must be positive")
    return hp.gamma_H0 * (f_arr / dp.f0) ** hp.gamma_H_beta


def holonomy_integrand(
    x: ArrayLike,
    f: ArrayLike,
    kappa: ArrayLike = KAPPA_STAR_DEFAULT,
    s: ArrayLike = 1.0,
    *,
    defect_params: DefectParams | None = None,
    holonomy_params: HolonomyParams | None = None,
) -> NDArray[np.floating]:
    """Instantaneous holonomy rate: γ_H(f) · ρ · (s^{−δ})^m.

    m = holonomy_fractal_multiplier (default 1 → double fractal).
    """
    hp = holonomy_params or HolonomyParams()
    dp = defect_params or DefectParams()
    rho = defect_density(x, f, kappa, s, params=dp)
    g = gamma_H(f, holonomy_params=hp, defect_params=dp)
    m = hp.holonomy_fractal_multiplier
    if m == 0.0:
        extra = 1.0
    elif m == 1.0:
        extra = scale_factor(s, kappa, params=dp)
    else:
        # general power: (s^{−δ})^m
        extra = scale_factor(s, kappa, params=dp) ** m
    return g * rho * extra


def accumulate_holonomy(
    x_trace: ArrayLike,
    f: ArrayLike,
    kappa: ArrayLike = KAPPA_STAR_DEFAULT,
    s: ArrayLike = 1.0,
    *,
    dt: float | ArrayLike = 1.0,
    defect_params: DefectParams | None = None,
    holonomy_params: HolonomyParams | None = None,
) -> NDArray[np.floating]:
    """Discrete cumulative holonomy along a misalignment history.

    H_{n+1} = H_n + γ_H(f) · ρ(x_n, …) · s^{−m δ} · Δt

    Parameters
    ----------
    x_trace :
        Sequence of misalignment values (time along axis 0).
    f, kappa, s :
        Scalar or broadcastable against each x sample.
    dt :
        Time step (scalar) or per-step array of length len(x_trace).

    Returns
    -------
    H : ndarray, shape (len(x_trace),)
        Holonomy after each step (includes H0 at the first update after step 0,
        i.e. H[i] is the value at the end of step i).
    """
    hp = holonomy_params or HolonomyParams()
    x_arr = np.asarray(x_trace, dtype=float)
    if x_arr.ndim != 1:
        raise ValueError("x_trace must be 1-D (time series)")

    rates = holonomy_integrand(
        x_arr,
        f,
        kappa,
        s,
        defect_params=defect_params,
        holonomy_params=hp,
    )
    dt_arr = np.asarray(dt, dtype=float)
    if dt_arr.ndim == 0:
        increments = rates * float(dt_arr)
    else:
        if dt_arr.shape != x_arr.shape:
            raise ValueError("dt array must match x_trace shape")
        increments = rates * dt_arr

    return hp.H0 + np.cumsum(increments)


@dataclass
class HolonomyAccumulator:
    """Stateful holonomy integrator for online / simulation loops.

    Example
    -------
    >>> acc = HolonomyAccumulator()
    >>> acc.step(x=0.2, f=1.0, kappa=0.85, s=0.1, dt=0.01)
    >>> acc.H
    """

    defect_params: DefectParams = field(default_factory=DefectParams)
    holonomy_params: HolonomyParams = field(default_factory=HolonomyParams)
    H: float = 0.0
    t: float = 0.0
    history: list[float] = field(default_factory=list)
    time_history: list[float] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.history:
            self.H = float(self.holonomy_params.H0)
            self.history = [self.H]
            self.time_history = [self.t]

    def reset(self, H0: float | None = None) -> None:
        self.H = float(self.holonomy_params.H0 if H0 is None else H0)
        self.t = 0.0
        self.history = [self.H]
        self.time_history = [0.0]

    def rate(
        self,
        x: float,
        f: float,
        kappa: float | None = None,
        s: float = 1.0,
    ) -> float:
        k = self.defect_params.kappa_star if kappa is None else kappa
        return float(
            holonomy_integrand(
                x,
                f,
                k,
                s,
                defect_params=self.defect_params,
                holonomy_params=self.holonomy_params,
            )
        )

    def step(
        self,
        x: float,
        f: float,
        kappa: float | None = None,
        s: float = 1.0,
        dt: float = 1.0,
    ) -> float:
        """Advance holonomy by one time step; return new H."""
        self.H = self.H + self.rate(x, f, kappa, s) * dt
        self.t += dt
        self.history.append(self.H)
        self.time_history.append(self.t)
        return self.H

    def run(
        self,
        x_trace: Sequence[float] | ArrayLike,
        f: float,
        kappa: float | None = None,
        s: float = 1.0,
        dt: float = 1.0,
        *,
        reset: bool = True,
    ) -> NDArray[np.floating]:
        """Integrate a full misalignment trace; return H history (excl. optional reset H0)."""
        if reset:
            self.reset()
        x_arr = np.asarray(x_trace, dtype=float)
        k = self.defect_params.kappa_star if kappa is None else kappa
        for x in x_arr:
            self.step(float(x), f, k, s, dt)
        # history[0] is the reset/initial H0; return post-step values
        return np.asarray(self.history[1:], dtype=float)

    def as_arrays(self) -> dict[str, NDArray[np.floating]]:
        return {
            "t": np.asarray(self.time_history, dtype=float),
            "H": np.asarray(self.history, dtype=float),
        }
