"""Survival-eigenstructure relaxation / healing of convex defects.

Continuous form:

    dρ/dt = −λ(κ) · ρ + η(f, κ) · x²

Discrete survival form (compatible with mystery PDE probes):

    ρ(t+1) = ρ(t) · exp(−λ(κ) Δt) · clamp(1 − x²/σ²) · s^{−δ(κ)}

The misalignment factor (1 − x²/σ²) is clamped with a soft floor so large
misalignment cannot drive the multiplier negative or hard-zero the density
in a single step. Physically: defects are created as fast as misalignment
allows; the floor keeps the discrete map stable and nearly monotonic.

Relaxation rate peaks at resonant κ*:

    λ(κ) = λ0 · exp(−|κ − κ*|² / ε)

Conceptual links
----------------
- Survival eigenstructure — mystery pde_survival_eigenstructure / kappa sweeps
- Holonomy-gap attractor  — λ peaks where B(κ) is minimized near κ*
- Discrete probes         — exponential_survival_probe style multipliers
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Callable, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.integrate import solve_ivp

from .defect_density import (
    KAPPA_STAR_DEFAULT,
    DefectParams,
    scale_factor,
    sigma_width,
)


@dataclass(frozen=True)
class RelaxationParams:
    """Parameters for λ(κ), source η, and discrete survival clamp.

    Parameters
    ----------
    lambda0 :
        Peak healing rate at κ = κ*.
    epsilon :
        Sharpness of the λ(κ) Gaussian peak (smaller → narrower attractor).
    eta0 :
        Base misalignment source strength for continuous ODE.
    eta_beta :
        Frequency scaling of η: η ∝ (f/f0)^{eta_beta} (defaults to defect β).
        If None, uses DefectParams.beta.
    misalignment_floor :
        Soft floor for clamp(1 − x²/σ², floor). Prevents a hard kill of ρ
        in one discrete step. Use 0.0 for a pure max(0, ·) clamp.
    apply_scale_in_discrete :
        If True (default), discrete step multiplies by s^{−δ(κ)} as in the
        refined equation set. Set False to evolve a scale-stripped amplitude.
    """

    lambda0: float = 1.0
    epsilon: float = 0.05
    eta0: float = 0.1
    eta_beta: float | None = None
    misalignment_floor: float = 0.01
    apply_scale_in_discrete: bool = True

    def __post_init__(self) -> None:
        if self.lambda0 < 0:
            raise ValueError("lambda0 should be non-negative")
        if self.epsilon <= 0:
            raise ValueError("epsilon must be positive")
        if self.eta0 < 0:
            raise ValueError("eta0 should be non-negative")
        if not (0.0 <= self.misalignment_floor <= 1.0):
            raise ValueError("misalignment_floor must lie in [0, 1]")

    def with_updates(self, **kwargs: Any) -> RelaxationParams:
        return replace(self, **kwargs)


def lambda_rate(
    kappa: ArrayLike,
    *,
    defect_params: DefectParams | None = None,
    relaxation_params: RelaxationParams | None = None,
) -> NDArray[np.floating]:
    """λ(κ) = λ0 · exp(−|κ − κ*|² / ε)."""
    dp = defect_params or DefectParams()
    rp = relaxation_params or RelaxationParams()
    det2 = (np.asarray(kappa, dtype=float) - dp.kappa_star) ** 2
    return rp.lambda0 * np.exp(-det2 / rp.epsilon)


def eta_source(
    f: ArrayLike,
    kappa: ArrayLike = KAPPA_STAR_DEFAULT,
    *,
    defect_params: DefectParams | None = None,
    relaxation_params: RelaxationParams | None = None,
) -> NDArray[np.floating]:
    """Misalignment source amplitude η(f, κ).

    η = η0 · (f/f0)^β_η · (1 + μ |κ − κ*|), with β_η from RelaxationParams
    or DefectParams.beta.
    """
    dp = defect_params or DefectParams()
    rp = relaxation_params or RelaxationParams()
    f_arr = np.asarray(f, dtype=float)
    if np.any(f_arr <= 0):
        raise ValueError("frequency f must be positive")
    beta = dp.beta if rp.eta_beta is None else rp.eta_beta
    det = np.abs(np.asarray(kappa, dtype=float) - dp.kappa_star)
    return rp.eta0 * (f_arr / dp.f0) ** beta * (1.0 + dp.mu * det)


def misalignment_multiplier(
    x: ArrayLike,
    f: ArrayLike,
    kappa: ArrayLike = KAPPA_STAR_DEFAULT,
    *,
    defect_params: DefectParams | None = None,
    relaxation_params: RelaxationParams | None = None,
) -> NDArray[np.floating]:
    """Clamped discrete survival factor: max(floor, 1 − x²/σ(f,κ)²).

    Large |x| saturates at ``misalignment_floor`` rather than going negative.
    """
    dp = defect_params or DefectParams()
    rp = relaxation_params or RelaxationParams()
    sig = sigma_width(f, kappa, params=dp)
    raw = 1.0 - np.asarray(x, dtype=float) ** 2 / (sig**2 + dp.eps)
    return np.maximum(rp.misalignment_floor, raw)


def discrete_step(
    rho: ArrayLike,
    x: ArrayLike,
    f: ArrayLike,
    kappa: ArrayLike = KAPPA_STAR_DEFAULT,
    s: ArrayLike = 1.0,
    *,
    dt: float = 1.0,
    defect_params: DefectParams | None = None,
    relaxation_params: RelaxationParams | None = None,
) -> NDArray[np.floating]:
    """One discrete survival step.

    ρ ← ρ · e^{−λ Δt} · clamp(1 − x²/σ²) · [s^{−δ}]

    The scale factor is included when ``apply_scale_in_discrete`` is True.
    Result is non-negative.
    """
    dp = defect_params or DefectParams()
    rp = relaxation_params or RelaxationParams()
    lam = lambda_rate(kappa, defect_params=dp, relaxation_params=rp)
    survival = np.exp(-lam * dt)
    mis = misalignment_multiplier(
        x, f, kappa, defect_params=dp, relaxation_params=rp
    )
    out = np.asarray(rho, dtype=float) * survival * mis
    if rp.apply_scale_in_discrete:
        out = out * scale_factor(s, kappa, params=dp)
    return np.maximum(0.0, out)


def continuous_rhs(
    rho: float,
    x: float,
    f: float,
    kappa: float,
    *,
    defect_params: DefectParams | None = None,
    relaxation_params: RelaxationParams | None = None,
) -> float:
    """dρ/dt = −λ(κ) ρ + η(f,κ) x²  (scalar helper)."""
    dp = defect_params or DefectParams()
    rp = relaxation_params or RelaxationParams()
    lam = float(lambda_rate(kappa, defect_params=dp, relaxation_params=rp))
    eta = float(eta_source(f, kappa, defect_params=dp, relaxation_params=rp))
    return -lam * rho + eta * (x**2)


def integrate_ode(
    rho0: float,
    t_span: tuple[float, float] | Sequence[float],
    x_of_t: Callable[[float], float] | float,
    f: float,
    kappa: float = KAPPA_STAR_DEFAULT,
    *,
    t_eval: ArrayLike | None = None,
    defect_params: DefectParams | None = None,
    relaxation_params: RelaxationParams | None = None,
    **solve_ivp_kwargs: Any,
) -> dict[str, NDArray[np.floating]]:
    """Integrate continuous relaxation with scipy.solve_ivp.

    Parameters
    ----------
    rho0 :
        Initial defect density (scale-stripped or absolute — caller chooses).
    t_span :
        (t0, tf) for the integrator.
    x_of_t :
        Callable x(t) or constant misalignment.
    f, kappa :
        Frequency and gauge (held fixed over the interval).
    """
    dp = defect_params or DefectParams()
    rp = relaxation_params or RelaxationParams()

    if callable(x_of_t):
        x_fn: Callable[[float], float] = x_of_t  # type: ignore[assignment]
    else:
        x_const = float(x_of_t)

        def x_fn(t: float, _xc: float = x_const) -> float:
            return _xc

    def rhs(t: float, y: NDArray[np.floating]) -> list[float]:
        return [
            continuous_rhs(
                float(y[0]),
                x_fn(t),
                f,
                kappa,
                defect_params=dp,
                relaxation_params=rp,
            )
        ]

    span = (float(t_span[0]), float(t_span[1]))
    sol = solve_ivp(
        rhs,
        span,
        [float(rho0)],
        t_eval=None if t_eval is None else np.asarray(t_eval, dtype=float),
        dense_output=False,
        **solve_ivp_kwargs,
    )
    if not sol.success:
        raise RuntimeError(f"solve_ivp failed: {sol.message}")
    return {"t": sol.t, "rho": sol.y[0]}


def discrete_trajectory(
    rho0: float,
    x_trace: ArrayLike,
    f: ArrayLike,
    kappa: ArrayLike = KAPPA_STAR_DEFAULT,
    s: ArrayLike = 1.0,
    *,
    dt: float = 1.0,
    defect_params: DefectParams | None = None,
    relaxation_params: RelaxationParams | None = None,
) -> NDArray[np.floating]:
    """Evolve ρ over a misalignment history with discrete survival steps.

    Returns array of length len(x_trace)+1 including the initial ρ0.
    """
    x_arr = np.asarray(x_trace, dtype=float)
    if x_arr.ndim != 1:
        raise ValueError("x_trace must be 1-D")
    rho = float(rho0)
    out = [rho]
    for x in x_arr:
        rho = float(
            discrete_step(
                rho,
                x,
                f,
                kappa,
                s,
                dt=dt,
                defect_params=defect_params,
                relaxation_params=relaxation_params,
            )
        )
        out.append(rho)
    return np.asarray(out, dtype=float)


@dataclass
class RelaxationDynamics:
    """Stateful relaxation engine (ODE snapshot + discrete steps)."""

    defect_params: DefectParams = field(default_factory=DefectParams)
    relaxation_params: RelaxationParams = field(default_factory=RelaxationParams)
    rho: float = 0.0
    t: float = 0.0
    history: list[float] = field(default_factory=list)
    time_history: list[float] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.history:
            self.history = [float(self.rho)]
            self.time_history = [float(self.t)]

    def reset(self, rho0: float = 0.0, t0: float = 0.0) -> None:
        self.rho = float(rho0)
        self.t = float(t0)
        self.history = [self.rho]
        self.time_history = [self.t]

    def lambda_at(self, kappa: float | None = None) -> float:
        k = self.defect_params.kappa_star if kappa is None else kappa
        return float(
            lambda_rate(
                k,
                defect_params=self.defect_params,
                relaxation_params=self.relaxation_params,
            )
        )

    def step_discrete(
        self,
        x: float,
        f: float,
        kappa: float | None = None,
        s: float = 1.0,
        dt: float = 1.0,
    ) -> float:
        k = self.defect_params.kappa_star if kappa is None else kappa
        self.rho = float(
            discrete_step(
                self.rho,
                x,
                f,
                k,
                s,
                dt=dt,
                defect_params=self.defect_params,
                relaxation_params=self.relaxation_params,
            )
        )
        self.t += dt
        self.history.append(self.rho)
        self.time_history.append(self.t)
        return self.rho

    def step_euler(
        self,
        x: float,
        f: float,
        kappa: float | None = None,
        dt: float = 0.01,
    ) -> float:
        """Explicit Euler step of the continuous ODE (no scale factor)."""
        k = self.defect_params.kappa_star if kappa is None else kappa
        dr = continuous_rhs(
            self.rho,
            x,
            f,
            k,
            defect_params=self.defect_params,
            relaxation_params=self.relaxation_params,
        )
        self.rho = max(0.0, self.rho + dt * dr)
        self.t += dt
        self.history.append(self.rho)
        self.time_history.append(self.t)
        return self.rho

    def run_discrete(
        self,
        x_trace: Sequence[float] | ArrayLike,
        f: float,
        kappa: float | None = None,
        s: float = 1.0,
        dt: float = 1.0,
        *,
        rho0: float | None = None,
        reset: bool = True,
    ) -> NDArray[np.floating]:
        if reset:
            self.reset(0.0 if rho0 is None else rho0)
        elif rho0 is not None:
            self.rho = float(rho0)
        k = self.defect_params.kappa_star if kappa is None else kappa
        for x in np.asarray(x_trace, dtype=float):
            self.step_discrete(float(x), f, k, s, dt)
        return np.asarray(self.history, dtype=float)

    def as_arrays(self) -> dict[str, NDArray[np.floating]]:
        return {
            "t": np.asarray(self.time_history, dtype=float),
            "rho": np.asarray(self.history, dtype=float),
        }


def lambda_kappa_curve(
    kappa: ArrayLike | None = None,
    *,
    defect_params: DefectParams | None = None,
    relaxation_params: RelaxationParams | None = None,
) -> dict[str, NDArray[np.floating]]:
    """Convenience sweep of λ(κ) around κ* (for demos / tests)."""
    dp = defect_params or DefectParams()
    if kappa is None:
        kappa = np.linspace(dp.kappa_star - 0.3, dp.kappa_star + 0.3, 201)
    k = np.asarray(kappa, dtype=float)
    return {
        "kappa": k,
        "lambda": lambda_rate(
            k, defect_params=dp, relaxation_params=relaxation_params
        ),
        "kappa_star": np.asarray(dp.kappa_star),
    }
