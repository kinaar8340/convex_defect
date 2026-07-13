"""Tests for convex_defect density, holonomy, relaxation, and simulator."""

from __future__ import annotations

import math

import numpy as np
import pytest

from convex_defect import (
    KAPPA_STAR_DEFAULT,
    ConvexDefectSimulator,
    DefectModel,
    DefectParams,
    HolonomyParams,
    PointerParams,
    RelaxationParams,
    SimConfig,
    accumulate_holonomy,
    defect_density,
    discrete_step,
    discrete_trajectory,
    fractal_exponent,
    gaussian_area_opacity,
    grid_to_phase_screen,
    holonomy_gap,
    integrate_ode,
    lambda_rate,
    misalignment_multiplier,
    opacity,
    pointer_trajectory,
    run_simulation,
    sigma_width,
    sweep_frequency,
    sweep_kappa,
)


# ---------------------------------------------------------------------------
# Density
# ---------------------------------------------------------------------------


def test_kappa_star_near_documented_value():
    assert abs(KAPPA_STAR_DEFAULT - 0.8513) < 5e-4


def test_rho_at_resonance_x0_s1_is_A0():
    m = DefectModel()
    r = float(m.rho(0.0, 1.0, m.kappa_star, 1.0))
    assert abs(r - 1.0) < 1e-12


def test_higher_frequency_narrows_sigma_and_raises_A():
    m = DefectModel()
    assert float(m.sigma(2.0)) < float(m.sigma(0.5))
    assert float(m.A(2.0)) > float(m.A(0.5))


def test_detuning_increases_amplitude_and_delta():
    m = DefectModel()
    k0, k1 = m.kappa_star, m.kappa_star + 0.2
    assert float(m.A(1.0, k1)) > float(m.A(1.0, k0))
    assert float(m.delta(k1)) > float(m.delta(k0))


def test_defect_density_broadcasting():
    x = np.linspace(-1, 1, 11)
    rho = defect_density(x, f=1.0, kappa=KAPPA_STAR_DEFAULT, s=1.0)
    assert rho.shape == (11,)
    assert np.all(rho > 0)
    assert rho[5] == pytest.approx(rho.max())


def test_opacity_positive_and_finite():
    tau = opacity(0.1, 1.0, KAPPA_STAR_DEFAULT)
    assert np.isfinite(tau)
    assert float(tau) > 0


def test_gaussian_area_matches_analytic_at_resonance():
    # ∫ ρ dx = A σ √π s^{-δ}; at κ*, f=f0, s=1 → A0 σ0 √π
    ga = float(gaussian_area_opacity(1.0, KAPPA_STAR_DEFAULT, 1.0))
    assert ga == pytest.approx(math.sqrt(math.pi), rel=1e-12)


def test_scale_factor_finer_scale_larger_when_delta_positive():
    m = DefectModel()
    rho_fine = float(m.rho(0.0, 1.0, m.kappa_star, s=0.1))
    rho_coarse = float(m.rho(0.0, 1.0, m.kappa_star, s=1.0))
    assert rho_fine > rho_coarse


def test_defect_params_rejects_bad_scale_window():
    with pytest.raises(ValueError):
        DefectParams(s_min=1.0, s_max=0.1)
    with pytest.raises(ValueError):
        DefectParams(eps=-1.0)


def test_rejects_nonpositive_frequency_and_scale():
    with pytest.raises(ValueError):
        sigma_width(0.0)
    with pytest.raises(ValueError):
        defect_density(0.0, 1.0, s=-0.1)


def test_holonomy_gap_sign_near_e_over_pi():
    # B(κ) = π²(e/π − κ); positive below e/π
    e_over_pi = math.e / math.pi
    assert float(holonomy_gap(e_over_pi - 0.1)) > 0
    assert float(holonomy_gap(e_over_pi + 0.1)) < 0


# ---------------------------------------------------------------------------
# Holonomy
# ---------------------------------------------------------------------------


def test_holonomy_nondecreasing_and_pure_accumulation():
    x = np.full(40, 0.25)
    H = accumulate_holonomy(x, f=1.0, kappa=KAPPA_STAR_DEFAULT, s=1.0, dt=0.05)
    assert H.shape == (40,)
    assert np.all(np.diff(H) >= -1e-15)
    assert H[-1] > H[0]


def test_finer_scale_richer_holonomy_double_fractal():
    x = np.full(30, 0.3)
    H_fine = accumulate_holonomy(x, f=1.0, s=0.1, dt=0.05)
    H_coarse = accumulate_holonomy(x, f=1.0, s=1.0, dt=0.05)
    assert H_fine[-1] > H_coarse[-1]


def test_holonomy_fractal_multiplier_zero_reduces_extra_weight():
    x = np.full(30, 0.3)
    H1 = accumulate_holonomy(
        x, f=1.0, s=0.1, dt=0.05, holonomy_params=HolonomyParams(holonomy_fractal_multiplier=1.0)
    )
    H0 = accumulate_holonomy(
        x, f=1.0, s=0.1, dt=0.05, holonomy_params=HolonomyParams(holonomy_fractal_multiplier=0.0)
    )
    assert H0[-1] < H1[-1]


# ---------------------------------------------------------------------------
# Relaxation
# ---------------------------------------------------------------------------


def test_lambda_peaks_at_kappa_star():
    ks = KAPPA_STAR_DEFAULT
    assert float(lambda_rate(ks)) > float(lambda_rate(ks + 0.2))
    assert float(lambda_rate(ks)) > float(lambda_rate(ks - 0.2))
    assert float(lambda_rate(ks)) == pytest.approx(1.0)


def test_misalignment_multiplier_clamp():
    rp = RelaxationParams(misalignment_floor=0.01)
    assert float(misalignment_multiplier(0.0, 1.0, relaxation_params=rp)) == pytest.approx(1.0)
    assert float(misalignment_multiplier(10.0, 1.0, relaxation_params=rp)) == pytest.approx(0.01)
    # never negative
    assert float(misalignment_multiplier(100.0, 1.0, relaxation_params=rp)) >= 0.0


def test_discrete_healing_at_x0_monotonic():
    traj = discrete_trajectory(
        1.0, np.zeros(25), f=1.0, kappa=KAPPA_STAR_DEFAULT, s=1.0, dt=0.1
    )
    assert traj[-1] < traj[0]
    assert np.all(np.diff(traj) <= 1e-12)


def test_detuned_heals_slower_than_resonant():
    x = np.zeros(40)
    t_star = discrete_trajectory(1.0, x, f=1.0, kappa=KAPPA_STAR_DEFAULT, s=1.0, dt=0.1)
    t_off = discrete_trajectory(
        1.0, x, f=1.0, kappa=KAPPA_STAR_DEFAULT + 0.25, s=1.0, dt=0.1
    )
    assert t_star[-1] < t_off[-1]


def test_ode_heals_when_aligned():
    sol = integrate_ode(
        1.0,
        (0.0, 4.0),
        0.0,
        f=1.0,
        kappa=KAPPA_STAR_DEFAULT,
        t_eval=np.linspace(0, 4, 40),
    )
    assert sol["rho"][-1] < sol["rho"][0]


def test_discrete_step_nonnegative():
    out = discrete_step(1.0, x=5.0, f=1.0, kappa=KAPPA_STAR_DEFAULT, s=1.0, dt=0.1)
    assert float(out) >= 0.0


# ---------------------------------------------------------------------------
# Simulator
# ---------------------------------------------------------------------------


def test_run_simulation_shapes_and_monotone_H():
    r = run_simulation(n_steps=50, dt=0.05, f=1.0, x0=0.4, seed=0, s=1.0)
    n = 51
    assert r.t.shape == (n,)
    assert r.x.shape == (n,)
    assert r.rho.shape == (n,)
    assert r.H.shape == (n,)
    assert r.tau.shape == (n,)
    assert np.all(r.rho >= 0)
    assert np.all(np.diff(r.H) >= -1e-14)
    assert r.x[-1] < r.x[0]  # pointer relaxes


def test_resonant_heals_faster_in_simulator():
    common = dict(n_steps=80, dt=0.05, f=1.0, x0=0.45, seed=2, s=1.0)
    r_star = run_simulation(kappa=KAPPA_STAR_DEFAULT, **common)
    r_off = run_simulation(kappa=KAPPA_STAR_DEFAULT + 0.25, **common)
    assert r_star.lambda_trace[0] > r_off.lambda_trace[0]
    assert r_star.rho[-1] < r_off.rho[-1]


def test_grid_2d_and_phase_screen():
    r = run_simulation(
        n_steps=20,
        grid_shape=(12, 10),
        seed=3,
        x0=0.3,
        f=1.2,
    )
    assert r.grid_final is not None
    assert r.grid_final.shape == (12, 10)
    assert r.grid_mean is not None
    assert len(r.grid_mean) == 21
    screen = grid_to_phase_screen(r.grid_final, r.f, r.kappa, r.s)
    assert screen.shape == (12, 10)
    assert np.all(screen >= 0)


def test_grid_1d():
    r = run_simulation(n_steps=10, grid_shape=(24,), seed=4)
    assert r.grid_final is not None
    assert r.grid_final.shape == (24,)


def test_euler_mode_runs():
    r = run_simulation(n_steps=30, mode="euler", x0=0.5, seed=0)
    assert r.rho[-1] >= 0
    assert r.metadata["mode"] == "euler"


def test_pointer_trajectory_with_noise_is_reproducible():
    p = PointerParams(x0=0.2, noise_std=0.05, gamma_x=0.3)
    a = pointer_trajectory(40, dt=0.05, params=p, seed=99)
    b = pointer_trajectory(40, dt=0.05, params=p, seed=99)
    assert np.allclose(a["x"], b["x"])


def test_sweeps():
    fs = sweep_frequency([0.5, 1.0, 2.0], n_steps=15, seed=0)
    ks = sweep_kappa([0.7, KAPPA_STAR_DEFAULT], n_steps=15, seed=0)
    assert len(fs) == 3
    assert len(ks) == 2
    # higher f → typically more holonomy coupling
    assert fs[-1].H[-1] > fs[0].H[-1]


def test_simulator_class_resolved_kappa():
    sim = ConvexDefectSimulator(config=SimConfig(n_steps=5, kappa=None))
    assert abs(sim.resolved_kappa() - KAPPA_STAR_DEFAULT) < 1e-12
    res = sim.run()
    assert res.kappa == pytest.approx(KAPPA_STAR_DEFAULT)


def test_sim_config_validation():
    with pytest.raises(ValueError):
        SimConfig(n_steps=0)
    with pytest.raises(ValueError):
        SimConfig(grid_shape=(4, 4, 4))
    with pytest.raises(ValueError):
        SimConfig(mode="bogus")  # type: ignore[arg-type]


def test_fractal_exponent_formula():
    p = DefectParams(delta0=0.2, nu=0.3)
    d = float(fractal_exponent(KAPPA_STAR_DEFAULT + 0.1, params=p))
    assert d == pytest.approx(0.2 + 0.3 * 0.1)


# ---------------------------------------------------------------------------
# Multi-scale ρ(s) field
# ---------------------------------------------------------------------------


def test_scale_bins_logspace():
    from convex_defect import ScaleBins

    b = ScaleBins.logspace(1e-3, 1.0, 12)
    assert b.n_scales == 12
    assert b.s[0] < b.s[-1]
    assert np.all(b.weights > 0)


def test_multi_scale_spectral_heal():
    from convex_defect import MultiScaleDefectField, MultiScaleParams

    field = MultiScaleDefectField.from_misalignment(
        0.5,
        f=1.0,
        kappa=KAPPA_STAR_DEFAULT,
        multi_params=MultiScaleParams(n_scales=10, apply_scale_in_discrete=False),
    )
    tau0 = field.mean_density()
    for _ in range(30):
        field.step_discrete(0.0, dt=0.1)
    assert field.mean_density() < tau0
    assert field.spectrum().shape == (10,)
    assert np.all(field.rho >= 0)


def test_multi_scale_spatial_screen():
    from convex_defect import MultiScaleDefectField, MultiScaleParams, multi_scale_phase_screen

    rng = np.random.default_rng(0)
    grid = 0.3 + 0.05 * rng.normal(size=(16, 16))
    field = MultiScaleDefectField.from_misalignment(
        0.3,
        f=1.2,
        kappa=KAPPA_STAR_DEFAULT,
        multi_params=MultiScaleParams(n_scales=8),
        grid=grid,
    )
    assert field.rho.shape == (16, 16, 8)
    screen = field.phase_screen()
    assert screen.shape == (16, 16)
    screen2 = multi_scale_phase_screen(grid, 1.2, KAPPA_STAR_DEFAULT, multi_params=MultiScaleParams(n_scales=8))
    assert screen2.shape == (16, 16)


def test_multi_scale_scale_coupling():
    from convex_defect import MultiScaleDefectField, MultiScaleParams

    a = MultiScaleDefectField.from_misalignment(
        0.4, multi_params=MultiScaleParams(n_scales=8, scale_coupling=0.0)
    )
    b = MultiScaleDefectField.from_misalignment(
        0.4, multi_params=MultiScaleParams(n_scales=8, scale_coupling=0.2)
    )
    # spike one bin then couple
    b.rho[0] *= 5.0
    a.rho[0] *= 5.0
    b.step_discrete(0.0, dt=0.05)
    a.step_discrete(0.0, dt=0.05)
    # coupling should smooth the spike relative to no coupling
    assert b.rho[0] < a.rho[0] or b.rho[1] > a.rho[1]


def test_simulator_multi_scale():
    r = run_simulation(
        n_steps=40,
        dt=0.05,
        x0=0.45,
        seed=0,
        multi_scale=True,
        n_scales=10,
        s=1.0,
    )
    assert r.metadata["multi_scale"] is True
    assert r.metadata["holonomy_source"] == "evolved_multi_scale"
    assert r.s_bins is not None and r.s_bins.shape == (10,)
    assert r.rho_spectrum is not None
    assert r.rho_spectrum.shape[0] == 41
    assert r.rho_spectrum.shape[1] == 10
    assert r.H[-1] >= r.H[0]
    assert r.rho[-1] < r.rho[0]


def test_simulator_multi_scale_with_grid():
    r = run_simulation(
        n_steps=15,
        grid_shape=(12, 12),
        multi_scale=True,
        n_scales=6,
        seed=1,
        x0=0.35,
    )
    assert r.screen_final is not None
    assert r.screen_final.shape == (12, 12)
    assert r.grid_final is not None


def test_grid_to_phase_screen_multi_scale():
    from convex_defect import grid_to_phase_screen

    g = np.ones((8, 8)) * 0.25
    s1 = grid_to_phase_screen(g, 1.0, KAPPA_STAR_DEFAULT, s=1.0, multi_scale=False)
    s2 = grid_to_phase_screen(g, 1.0, KAPPA_STAR_DEFAULT, multi_scale=True, n_scales=8)
    assert s1.shape == s2.shape == (8, 8)
    assert float(s2.mean()) > 0
