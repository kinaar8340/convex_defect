#!/usr/bin/env python3
"""Visualize convex defect density, relaxation, holonomy, and phase screens.

Saves figures under ``outputs/`` (created next to the repo root).

    PYTHONPATH=src python examples/convex_defect_demo.py
    # or after pip install -e .
    python examples/convex_defect_demo.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from convex_defect import (  # noqa: E402
    KAPPA_STAR_DEFAULT,
    ConvexDefectSimulator,
    DefectModel,
    PointerParams,
    SimConfig,
    grid_to_phase_screen,
    lambda_kappa_curve,
    run_simulation,
    sweep_frequency,
    sweep_kappa,
)

OUT = ROOT / "outputs"
OUT.mkdir(parents=True, exist_ok=True)


def _save(fig: plt.Figure, name: str) -> Path:
    path = OUT / name
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path}")
    return path


def plot_gaussian_vs_frequency() -> None:
    """ρ(x) Gaussian core at several frequencies (fixed κ*, s=1)."""
    model = DefectModel()
    x = np.linspace(-3, 3, 401)
    freqs = [0.5, 1.0, 2.0, 4.0]
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for f in freqs:
        rho = model.rho(x, f, model.kappa_star, s=1.0)
        ax.plot(x, rho, label=f"f={f:g}  σ={float(model.sigma(f)):.3f}")
    ax.set_xlabel("misalignment x")
    ax.set_ylabel(r"$\rho(x, f, \kappa^*, 1)$")
    ax.set_title("Gaussian defect core vs frequency (quicksand narrowing)")
    ax.legend(frameon=False)
    ax.grid(True, alpha=0.3)
    _save(fig, "gaussian_vs_frequency.png")


def plot_relaxation_curves() -> None:
    """ρ(t) healing at κ* vs detuned κ (same pointer history)."""
    kappas = {
        r"$\kappa^*$": KAPPA_STAR_DEFAULT,
        r"$\kappa^*+0.15$": KAPPA_STAR_DEFAULT + 0.15,
        r"$\kappa^*+0.30$": KAPPA_STAR_DEFAULT + 0.30,
    }
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), constrained_layout=True)

    for label, k in kappas.items():
        r = run_simulation(
            n_steps=160,
            dt=0.05,
            f=1.0,
            kappa=k,
            s=1.0,
            x0=0.5,
            seed=0,
            mode="discrete",
        )
        axes[0].plot(r.t, r.rho, label=label)
        axes[1].plot(r.t, r.x, label=label, alpha=0.85)

    axes[0].set_xlabel("t")
    axes[0].set_ylabel(r"$\rho(t)$")
    axes[0].set_title("Defect healing (discrete survival)")
    axes[0].set_yscale("log")
    axes[0].legend(frameon=False)
    axes[0].grid(True, alpha=0.3, which="both")

    axes[1].set_xlabel("t")
    axes[1].set_ylabel(r"$x(t)$")
    axes[1].set_title("Pointer realignment (shared seed)")
    axes[1].legend(frameon=False)
    axes[1].grid(True, alpha=0.3)
    _save(fig, "relaxation_curves.png")


def plot_holonomy_vs_kappa() -> None:
    """H(t) for several κ — lower total memory near κ* when healing is fast."""
    kappas = np.array(
        [
            KAPPA_STAR_DEFAULT - 0.2,
            KAPPA_STAR_DEFAULT,
            KAPPA_STAR_DEFAULT + 0.15,
            KAPPA_STAR_DEFAULT + 0.3,
        ]
    )
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for k in kappas:
        r = run_simulation(
            n_steps=160,
            dt=0.05,
            f=1.0,
            kappa=float(k),
            s=1.0,
            x0=0.5,
            seed=0,
        )
        tag = f"κ={k:.3f}"
        if abs(k - KAPPA_STAR_DEFAULT) < 1e-9:
            tag += " (κ*)"
        ax.plot(r.t, r.H, label=tag)
    ax.set_xlabel("t")
    ax.set_ylabel("H(t)")
    ax.set_title("Holonomy accumulation vs κ (pure memory, instantaneous ρ)")
    ax.legend(frameon=False)
    ax.grid(True, alpha=0.3)
    _save(fig, "holonomy_vs_kappa.png")


def plot_holonomy_vs_frequency() -> None:
    """Final H vs frequency sweep + time series for a few f."""
    freqs = np.linspace(0.4, 3.0, 14)
    results = sweep_frequency(
        freqs,
        n_steps=120,
        dt=0.05,
        s=1.0,
        x0=0.45,
        seed=1,
        kappa=KAPPA_STAR_DEFAULT,
    )
    H_final = np.array([r.H[-1] for r in results])

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), constrained_layout=True)
    axes[0].plot(freqs, H_final, "o-", color="C0")
    axes[0].set_xlabel("f")
    axes[0].set_ylabel(r"$H_{\mathrm{final}}$")
    axes[0].set_title("Final holonomy vs frequency")
    axes[0].grid(True, alpha=0.3)

    for f in [0.5, 1.0, 2.0]:
        r = run_simulation(
            n_steps=120,
            dt=0.05,
            f=f,
            s=1.0,
            x0=0.45,
            seed=1,
            kappa=KAPPA_STAR_DEFAULT,
        )
        axes[1].plot(r.t, r.H, label=f"f={f:g}")
    axes[1].set_xlabel("t")
    axes[1].set_ylabel("H(t)")
    axes[1].set_title("H(t) at selected frequencies")
    axes[1].legend(frameon=False)
    axes[1].grid(True, alpha=0.3)
    _save(fig, "holonomy_vs_frequency.png")


def plot_lambda_peak() -> None:
    """λ(κ) peak at κ* (survival eigenstructure)."""
    curve = lambda_kappa_curve()
    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    ax.plot(curve["kappa"], curve["lambda"], color="C3")
    ax.axvline(float(curve["kappa_star"]), color="k", ls="--", lw=1, label=r"$\kappa^*$")
    ax.set_xlabel(r"$\kappa$")
    ax.set_ylabel(r"$\lambda(\kappa)$")
    ax.set_title("Relaxation rate peaks at resonant κ*")
    ax.legend(frameon=False)
    ax.grid(True, alpha=0.3)
    _save(fig, "lambda_kappa.png")


def plot_phase_screen_before_after() -> None:
    """2D local-misalignment grid → phase screen at t≈0 and final."""
    cfg = SimConfig(
        n_steps=100,
        dt=0.05,
        f=1.5,
        s=0.5,
        seed=7,
        grid_shape=(48, 48),
        grid_correlation=0.9,
        grid_noise=0.08,
    )
    ptr = PointerParams(x0=0.6, gamma_x=0.4, noise_std=0.02)
    sim = ConvexDefectSimulator(config=cfg, pointer_params=ptr)

    # capture early grid by short run + full run
    early = sim.run(cfg.with_updates(n_steps=5))
    late = sim.run(cfg)

    screen0 = grid_to_phase_screen(
        early.grid_final, early.f, early.kappa, early.s
    )
    screen1 = grid_to_phase_screen(
        late.grid_final, late.f, late.kappa, late.s
    )

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6), constrained_layout=True)
    im0 = axes[0].imshow(screen0, origin="lower", cmap="magma")
    axes[0].set_title("Phase screen (early)")
    fig.colorbar(im0, ax=axes[0], fraction=0.046)

    im1 = axes[1].imshow(screen1, origin="lower", cmap="magma")
    axes[1].set_title("Phase screen (final)")
    fig.colorbar(im1, ax=axes[1], fraction=0.046)

    im2 = axes[2].imshow(late.grid_final, origin="lower", cmap="coolwarm")
    axes[2].set_title(r"Final local $x_{ij}$")
    fig.colorbar(im2, ax=axes[2], fraction=0.046)

    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(
        f"2D misalignment texture → ρ screen  "
        f"(f={late.f}, κ={late.kappa:.3f}, s={late.s})"
    )
    _save(fig, "phase_screen_before_after.png")


def plot_summary_dashboard() -> None:
    """Compact 2×2 dashboard for the default resonant run."""
    r = run_simulation(
        n_steps=150,
        dt=0.05,
        f=1.0,
        kappa=KAPPA_STAR_DEFAULT,
        s=1.0,
        x0=0.5,
        seed=0,
    )
    fig, axes = plt.subplots(2, 2, figsize=(9, 7), constrained_layout=True)
    axes[0, 0].plot(r.t, r.x, color="C0")
    axes[0, 0].set_title("Pointer x(t)")
    axes[0, 0].set_xlabel("t")
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].semilogy(r.t, np.maximum(r.rho, 1e-16), color="C1")
    axes[0, 1].set_title(r"Defect density $\rho(t)$")
    axes[0, 1].set_xlabel("t")
    axes[0, 1].grid(True, alpha=0.3, which="both")

    axes[1, 0].plot(r.t, r.H, color="C2")
    axes[1, 0].set_title("Holonomy H(t)")
    axes[1, 0].set_xlabel("t")
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(r.t, r.tau, color="C4")
    axes[1, 1].set_title(r"Opacity $\tau(t)$")
    axes[1, 1].set_xlabel("t")
    axes[1, 1].grid(True, alpha=0.3)

    fig.suptitle(
        f"convex_defect summary  κ*={KAPPA_STAR_DEFAULT:.4f}  f={r.f}  s={r.s}"
    )
    _save(fig, "summary_dashboard.png")


def main() -> None:
    print(f"convex_defect demo → {OUT}")
    print(f"  κ* ≈ {KAPPA_STAR_DEFAULT:.6f}")
    plot_gaussian_vs_frequency()
    plot_lambda_peak()
    plot_relaxation_curves()
    plot_holonomy_vs_kappa()
    plot_holonomy_vs_frequency()
    plot_phase_screen_before_after()
    plot_summary_dashboard()

    # quick numeric report
    results = sweep_kappa(
        [KAPPA_STAR_DEFAULT, KAPPA_STAR_DEFAULT + 0.25],
        n_steps=100,
        dt=0.05,
        f=1.0,
        x0=0.5,
        seed=0,
    )
    print("\nκ sweep (final ρ, H):")
    for r in results:
        print(f"  κ={r.kappa:.4f}  ρ_f={r.rho[-1]:.4e}  H_f={r.H[-1]:.4f}")
    print("done.")


if __name__ == "__main__":
    main()
