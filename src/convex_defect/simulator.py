r"""Toy simulator: pointer misalignment + defect evolution + holonomy.

Couples three pieces at each time step:

1. **Pointer** \(x(t)\) — relaxes toward resonant alignment (0) with optional
   noise and external drive.
2. **Defect density** \(\rho\) — discrete survival update (or continuous Euler)
   from :mod:`relaxation_dynamics`.
3. **Holonomy** \(H\) — pure accumulation (v0.1) from
   :mod:`holonomy_accumulator`.

Also tracks topological opacity \(\tau(x, f, \kappa)\) and supports an optional
1D/2D spatial grid of *local* misalignment for later turbulence-screen hooks
into ``flux_trajectoid`` propagation.

Conceptual links
----------------
- mystery PDE survival / κ sweeps — relaxation rate \(\lambda(\kappa)\)
- flux_trajectoid Kolmogorov screens — grid misalignment as structured texture
- oam_flux lattice twist — pointer as global gauge / alignment bias
- trajectoid geodesics — holonomy along the misalignment history
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .defect_density import DefectParams, defect_density, opacity
from .holonomy_accumulator import HolonomyParams, holonomy_integrand
from .multi_scale_field import (
    MultiScaleDefectField,
    MultiScaleParams,
    multi_scale_phase_screen,
)
from .relaxation_dynamics import (
    RelaxationParams,
    continuous_rhs,
    discrete_step,
    lambda_rate,
)

GridDim = Literal[1, 2]


# ---------------------------------------------------------------------------
# Pointer dynamics
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PointerParams:
    r"""Global pointer misalignment dynamics.

    Continuous model (Euler-discretized):

        dx/dt = -gamma_x * x + drive(t) + noise

    so that free evolution is exponential realignment toward x = 0.
    """

    gamma_x: float = 0.5  # realignment rate
    noise_std: float = 0.0  # Wiener-like increment scale per √dt
    drive_amplitude: float = 0.0  # A_d · sin(2π f_d t + φ)
    drive_frequency: float = 0.1
    drive_phase: float = 0.0
    x0: float = 0.3  # initial misalignment
    x_clip: float | None = 5.0  # optional |x| hard clip (None = none)

    def __post_init__(self) -> None:
        if self.gamma_x < 0:
            raise ValueError("gamma_x should be non-negative")
        if self.noise_std < 0:
            raise ValueError("noise_std should be non-negative")
        if self.x_clip is not None and self.x_clip <= 0:
            raise ValueError("x_clip must be positive when set")

    def with_updates(self, **kwargs: Any) -> PointerParams:
        return replace(self, **kwargs)


@dataclass
class PointerDynamics:
    """Stateful pointer integrator."""

    params: PointerParams = field(default_factory=PointerParams)
    x: float = 0.0
    t: float = 0.0
    rng: np.random.Generator = field(default_factory=lambda: np.random.default_rng())

    def __post_init__(self) -> None:
        # honour x0 on first construction if x left at default and x0 set
        if self.x == 0.0 and self.params.x0 != 0.0:
            self.x = float(self.params.x0)

    def reset(self, x0: float | None = None, t0: float = 0.0) -> None:
        self.x = float(self.params.x0 if x0 is None else x0)
        self.t = float(t0)

    def drive(self, t: float | None = None) -> float:
        tt = self.t if t is None else float(t)
        p = self.params
        if p.drive_amplitude == 0.0:
            return 0.0
        return p.drive_amplitude * np.sin(
            2.0 * np.pi * p.drive_frequency * tt + p.drive_phase
        )

    def step(self, dt: float = 0.01) -> float:
        """Advance pointer by one Euler–Maruyama step; return new x."""
        p = self.params
        noise = 0.0
        if p.noise_std > 0.0:
            noise = p.noise_std * np.sqrt(dt) * float(self.rng.normal())
        dx = (-p.gamma_x * self.x + self.drive()) * dt + noise
        self.x = self.x + dx
        if p.x_clip is not None:
            self.x = float(np.clip(self.x, -p.x_clip, p.x_clip))
        self.t += dt
        return self.x


def pointer_trajectory(
    n_steps: int,
    dt: float = 0.01,
    *,
    params: PointerParams | None = None,
    seed: int | None = None,
) -> dict[str, NDArray[np.floating]]:
    """Integrate pointer alone; return ``t`` and ``x`` arrays (length n_steps+1)."""
    p = params or PointerParams()
    rng = np.random.default_rng(seed)
    ptr = PointerDynamics(params=p, x=p.x0, t=0.0, rng=rng)
    t = [0.0]
    x = [ptr.x]
    for _ in range(n_steps):
        ptr.step(dt)
        t.append(ptr.t)
        x.append(ptr.x)
    return {"t": np.asarray(t, dtype=float), "x": np.asarray(x, dtype=float)}


# ---------------------------------------------------------------------------
# Simulation config & result
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SimConfig:
    """Top-level simulator configuration."""

    n_steps: int = 200
    dt: float = 0.05
    f: float = 1.0
    kappa: float | None = None  # None → DefectParams.kappa_star
    s: float = 1.0  # scale bin (use 1.0 for pure healing demos)
    rho0: float | None = None  # None → seed from defect_density(x0, …)
    seed: int | None = 0
    mode: Literal["discrete", "euler"] = "discrete"
    # optional spatial grid of local misalignment
    grid_shape: tuple[int, ...] | None = None  # e.g. (64,) or (32, 32)
    grid_correlation: float = 0.85  # AR(1) blend toward global x each step
    grid_noise: float = 0.05  # local noise std on the grid
    track_grid_stats: bool = True
    # multi-scale ρ(s) field
    multi_scale: bool = False
    n_scales: int = 16
    scale_coupling: float = 0.0
    track_spectrum: bool = True  # store ρ(s) history when multi_scale

    def __post_init__(self) -> None:
        if self.n_steps < 1:
            raise ValueError("n_steps must be >= 1")
        if self.dt <= 0:
            raise ValueError("dt must be positive")
        if self.f <= 0:
            raise ValueError("f must be positive")
        if self.s <= 0:
            raise ValueError("s must be positive")
        if self.mode not in ("discrete", "euler"):
            raise ValueError("mode must be 'discrete' or 'euler'")
        if self.grid_shape is not None:
            if len(self.grid_shape) not in (1, 2):
                raise ValueError("grid_shape must be 1-D or 2-D")
            if any(n < 1 for n in self.grid_shape):
                raise ValueError("grid_shape entries must be positive")
        if not (0.0 <= self.grid_correlation <= 1.0):
            raise ValueError("grid_correlation must lie in [0, 1]")
        if self.n_scales < 2:
            raise ValueError("n_scales must be >= 2")
        if not (0.0 <= self.scale_coupling < 1.0):
            raise ValueError("scale_coupling must lie in [0, 1)")

    def with_updates(self, **kwargs: Any) -> SimConfig:
        return replace(self, **kwargs)


@dataclass
class SimResult:
    """Time series and optional grid snapshots from a simulation run."""

    t: NDArray[np.floating]
    x: NDArray[np.floating]
    rho: NDArray[np.floating]
    H: NDArray[np.floating]
    tau: NDArray[np.floating]
    lambda_trace: NDArray[np.floating]
    f: float
    kappa: float
    s: float
    config: SimConfig
    # optional grid diagnostics
    grid_mean: NDArray[np.floating] | None = None
    grid_std: NDArray[np.floating] | None = None
    grid_final: NDArray[np.floating] | None = None
    # multi-scale diagnostics
    s_bins: NDArray[np.floating] | None = None
    rho_spectrum: NDArray[np.floating] | None = None  # (n_times, n_scales) mean spectrum
    rho_spectrum_final: NDArray[np.floating] | None = None  # last spectrum
    screen_final: NDArray[np.floating] | None = None  # integrated multi-scale screen
    metadata: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "t": self.t,
            "x": self.x,
            "rho": self.rho,
            "H": self.H,
            "tau": self.tau,
            "lambda": self.lambda_trace,
            "f": self.f,
            "kappa": self.kappa,
            "s": self.s,
        }
        if self.grid_mean is not None:
            out["grid_mean"] = self.grid_mean
            out["grid_std"] = self.grid_std
        if self.grid_final is not None:
            out["grid_final"] = self.grid_final
        if self.s_bins is not None:
            out["s_bins"] = self.s_bins
        if self.rho_spectrum is not None:
            out["rho_spectrum"] = self.rho_spectrum
        if self.rho_spectrum_final is not None:
            out["rho_spectrum_final"] = self.rho_spectrum_final
        if self.screen_final is not None:
            out["screen_final"] = self.screen_final
        out["metadata"] = self.metadata
        return out


# ---------------------------------------------------------------------------
# Grid helpers (local misalignment texture)
# ---------------------------------------------------------------------------


def _init_grid(
    shape: tuple[int, ...],
    x0: float,
    noise: float,
    rng: np.random.Generator,
) -> NDArray[np.floating]:
    g = np.full(shape, float(x0), dtype=float)
    if noise > 0.0:
        g = g + noise * rng.normal(size=shape)
    return g


def _step_grid(
    grid: NDArray[np.floating],
    x_global: float,
    *,
    correlation: float,
    noise: float,
    rng: np.random.Generator,
) -> NDArray[np.floating]:
    """Blend local misalignment toward global pointer + spatial noise.

    grid ← c · grid + (1−c) · x_global + noise
    """
    c = correlation
    out = c * grid + (1.0 - c) * x_global
    if noise > 0.0:
        out = out + noise * rng.normal(size=grid.shape)
    return out


def grid_to_phase_screen(
    grid: ArrayLike,
    f: float,
    kappa: float,
    s: float = 1.0,
    *,
    defect_params: DefectParams | None = None,
    gain: float = 1.0,
    multi_scale: bool = False,
    n_scales: int = 16,
    multi_params: MultiScaleParams | None = None,
) -> NDArray[np.floating]:
    """Map local misalignment grid → defect-density phase/amplitude screen.

    Suitable as a structured alternative to Kolmogorov screens in
    ``flux_trajectoid.propagation``.

    - Single-scale (default): ρ(x_ij, f, κ, s) · gain
    - Multi-scale: Σ_k ρ(x_ij, f, κ, s_k) · w_k · gain
    """
    g = np.asarray(grid, dtype=float)
    if multi_scale:
        mp = multi_params or MultiScaleParams(n_scales=n_scales)
        return multi_scale_phase_screen(
            g,
            f,
            kappa,
            defect_params=defect_params,
            multi_params=mp,
            gain=gain,
        )
    return gain * defect_density(g, f, kappa, s, params=defect_params)


# ---------------------------------------------------------------------------
# Main simulator
# ---------------------------------------------------------------------------


@dataclass
class ConvexDefectSimulator:
    """Coupled pointer + defect + holonomy simulator.

    Example
    -------
    >>> sim = ConvexDefectSimulator()
    >>> result = sim.run()
    >>> result.rho.shape == result.t.shape
    True
    """

    config: SimConfig = field(default_factory=SimConfig)
    defect_params: DefectParams = field(default_factory=DefectParams)
    pointer_params: PointerParams = field(default_factory=PointerParams)
    relaxation_params: RelaxationParams = field(default_factory=RelaxationParams)
    holonomy_params: HolonomyParams = field(default_factory=HolonomyParams)

    def resolved_kappa(self) -> float:
        if self.config.kappa is None:
            return float(self.defect_params.kappa_star)
        return float(self.config.kappa)

    def run(self, config: SimConfig | None = None) -> SimResult:
        """Run a full trajectory; return :class:`SimResult` traces."""
        cfg = config or self.config
        dp = self.defect_params
        pp = self.pointer_params
        rp = self.relaxation_params
        hp = self.holonomy_params
        kappa = float(dp.kappa_star if cfg.kappa is None else cfg.kappa)
        f = float(cfg.f)
        s = float(cfg.s)
        dt = float(cfg.dt)
        rng = np.random.default_rng(cfg.seed)

        # --- initial state ---
        ptr = PointerDynamics(params=pp, x=pp.x0, t=0.0, rng=rng)
        x0 = float(ptr.x)
        H = float(hp.H0)
        lam0 = float(lambda_rate(kappa, defect_params=dp, relaxation_params=rp))

        grid: NDArray[np.floating] | None = None
        g_mean: list[float] = []
        g_std: list[float] = []
        if cfg.grid_shape is not None:
            grid = _init_grid(cfg.grid_shape, x0, cfg.grid_noise, rng)
            if cfg.track_grid_stats:
                g_mean.append(float(grid.mean()))
                g_std.append(float(grid.std()))

        ms_field: MultiScaleDefectField | None = None
        spectrum_hist: list[NDArray[np.floating]] = []
        if cfg.multi_scale:
            mp = MultiScaleParams(
                n_scales=cfg.n_scales,
                scale_coupling=cfg.scale_coupling,
                apply_scale_in_discrete=False,
            )
            # multi-scale: don't re-multiply s^{-δ} each discrete step
            rp_ms = rp.with_updates(apply_scale_in_discrete=False)
            if grid is not None:
                ms_field = MultiScaleDefectField.from_misalignment(
                    x0,
                    f=f,
                    kappa=kappa,
                    defect_params=dp,
                    relaxation_params=rp_ms,
                    multi_params=mp,
                    grid=grid,
                )
            else:
                ms_field = MultiScaleDefectField.from_misalignment(
                    x0,
                    f=f,
                    kappa=kappa,
                    defect_params=dp,
                    relaxation_params=rp_ms,
                    multi_params=mp,
                )
            rho = ms_field.mean_density()
            tau0 = float(ms_field.opacity()) if not ms_field.is_spatial else float(
                np.mean(ms_field.opacity())
            )
            if cfg.track_spectrum:
                spectrum_hist.append(ms_field.spectrum().copy())
        else:
            if cfg.rho0 is None:
                rho = float(defect_density(x0, f, kappa, s, params=dp))
            else:
                rho = float(cfg.rho0)
            tau0 = float(opacity(x0, f, kappa, params=dp))

        t_hist = [0.0]
        x_hist = [x0]
        rho_hist = [rho]
        H_hist = [H]
        tau_hist = [tau0]
        lam_hist = [lam0]

        # --- time loop ---
        for _ in range(cfg.n_steps):
            # 1) pointer
            x = ptr.step(dt)

            # 2) optional local misalignment grid
            if grid is not None:
                grid = _step_grid(
                    grid,
                    x,
                    correlation=cfg.grid_correlation,
                    noise=cfg.grid_noise,
                    rng=rng,
                )
                if cfg.track_grid_stats:
                    g_mean.append(float(grid.mean()))
                    g_std.append(float(grid.std()))

            # 3) defect density
            if ms_field is not None:
                if cfg.mode == "discrete":
                    ms_field.step_discrete(x, dt=dt, grid=grid)
                else:
                    ms_field.step_euler(x, dt=dt, grid=grid)
                rho = ms_field.mean_density()
                tau = (
                    float(ms_field.opacity())
                    if not ms_field.is_spatial
                    else float(np.mean(ms_field.opacity()))
                )
                # multi-scale holonomy from evolved ρ(s)
                rate = ms_field.holonomy_rate(x, holonomy_params=hp)
                if cfg.track_spectrum:
                    spectrum_hist.append(ms_field.spectrum().copy())
            else:
                if cfg.mode == "discrete":
                    rho = float(
                        discrete_step(
                            rho,
                            x,
                            f,
                            kappa,
                            s,
                            dt=dt,
                            defect_params=dp,
                            relaxation_params=rp,
                        )
                    )
                else:
                    dr = continuous_rhs(
                        rho,
                        x,
                        f,
                        kappa,
                        defect_params=dp,
                        relaxation_params=rp,
                    )
                    rho = max(0.0, rho + dt * dr)
                # instantaneous holonomy (v0.1 formula)
                rate = float(
                    holonomy_integrand(
                        x,
                        f,
                        kappa,
                        s,
                        defect_params=dp,
                        holonomy_params=hp,
                    )
                )
                tau = float(opacity(x, f, kappa, params=dp))

            # 4) holonomy (pure accumulation)
            H = H + rate * dt

            # 5) lambda snapshot
            lam = float(lambda_rate(kappa, defect_params=dp, relaxation_params=rp))

            t_hist.append(ptr.t)
            x_hist.append(x)
            rho_hist.append(rho)
            H_hist.append(H)
            tau_hist.append(tau)
            lam_hist.append(lam)

        screen_final = None
        s_bins = None
        rho_spectrum = None
        rho_spectrum_final = None
        if ms_field is not None:
            s_bins = ms_field.bins.s.copy()
            if spectrum_hist:
                rho_spectrum = np.stack(spectrum_hist, axis=0)
                rho_spectrum_final = spectrum_hist[-1].copy()
            if ms_field.is_spatial:
                screen_final = ms_field.phase_screen()

        return SimResult(
            t=np.asarray(t_hist, dtype=float),
            x=np.asarray(x_hist, dtype=float),
            rho=np.asarray(rho_hist, dtype=float),
            H=np.asarray(H_hist, dtype=float),
            tau=np.asarray(tau_hist, dtype=float),
            lambda_trace=np.asarray(lam_hist, dtype=float),
            f=f,
            kappa=kappa,
            s=s,
            config=cfg,
            grid_mean=np.asarray(g_mean, dtype=float) if g_mean else None,
            grid_std=np.asarray(g_std, dtype=float) if g_std else None,
            grid_final=None if grid is None else np.asarray(grid, dtype=float),
            s_bins=s_bins,
            rho_spectrum=rho_spectrum,
            rho_spectrum_final=rho_spectrum_final,
            screen_final=screen_final,
            metadata={
                "mode": cfg.mode,
                "lambda0": float(lam0),
                "seed": cfg.seed,
                "grid_shape": cfg.grid_shape,
                "multi_scale": cfg.multi_scale,
                "n_scales": cfg.n_scales if cfg.multi_scale else None,
                "scale_coupling": cfg.scale_coupling if cfg.multi_scale else None,
                "holonomy_source": "evolved_multi_scale" if cfg.multi_scale else "instantaneous",
            },
        )


def run_simulation(
    *,
    n_steps: int = 200,
    dt: float = 0.05,
    f: float = 1.0,
    kappa: float | None = None,
    s: float = 1.0,
    x0: float = 0.3,
    seed: int | None = 0,
    mode: Literal["discrete", "euler"] = "discrete",
    grid_shape: tuple[int, ...] | None = None,
    multi_scale: bool = False,
    n_scales: int = 16,
    scale_coupling: float = 0.0,
    defect_params: DefectParams | None = None,
    pointer_params: PointerParams | None = None,
    relaxation_params: RelaxationParams | None = None,
    holonomy_params: HolonomyParams | None = None,
    **config_overrides: Any,
) -> SimResult:
    """One-shot convenience wrapper around :class:`ConvexDefectSimulator`."""
    pp = pointer_params or PointerParams(x0=x0)
    if pointer_params is None and x0 != PointerParams().x0:
        pp = PointerParams(x0=x0)
    elif pointer_params is not None and x0 != pointer_params.x0:
        pp = pointer_params.with_updates(x0=x0)

    cfg = SimConfig(
        n_steps=n_steps,
        dt=dt,
        f=f,
        kappa=kappa,
        s=s,
        seed=seed,
        mode=mode,
        grid_shape=grid_shape,
        multi_scale=multi_scale,
        n_scales=n_scales,
        scale_coupling=scale_coupling,
        **config_overrides,
    )
    sim = ConvexDefectSimulator(
        config=cfg,
        defect_params=defect_params or DefectParams(),
        pointer_params=pp,
        relaxation_params=relaxation_params or RelaxationParams(),
        holonomy_params=holonomy_params or HolonomyParams(),
    )
    return sim.run()


def sweep_frequency(
    frequencies: ArrayLike,
    **run_kwargs: Any,
) -> list[SimResult]:
    """Run the same scenario at multiple frequencies."""
    return [run_simulation(f=float(f), **run_kwargs) for f in np.asarray(frequencies)]


def sweep_kappa(
    kappas: ArrayLike,
    **run_kwargs: Any,
) -> list[SimResult]:
    """Run the same scenario at multiple κ values."""
    return [run_simulation(kappa=float(k), **run_kwargs) for k in np.asarray(kappas)]
