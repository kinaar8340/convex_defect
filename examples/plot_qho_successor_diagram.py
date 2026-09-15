#!/usr/bin/env python3
"""Generate the QHO-successor poster for topological convex defect density.

Uses live library formulas (not hand-drawn curves):

    PYTHONPATH=src python examples/plot_qho_successor_diagram.py

Outputs:
    outputs/topological_convex_defect_qho_successor.png
    outputs/topological_convex_defect_qho_successor.pdf
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from convex_defect import (  # noqa: E402
    KAPPA_STAR_DEFAULT,
    DefectModel,
    fractal_exponent,
    run_simulation,
)
from convex_defect.relaxation_dynamics import lambda_rate as lam_rate  # noqa: E402

OUT = ROOT / "outputs"
OUT.mkdir(parents=True, exist_ok=True)


def main() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "mathtext.fontset": "dejavusans",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "#f4efe4",
            "axes.facecolor": "#f4efe4",
            "savefig.facecolor": "#f4efe4",
            "text.color": "#1a1a1a",
            "axes.labelcolor": "#1a1a1a",
            "xtick.color": "#333",
            "ytick.color": "#333",
            "axes.edgecolor": "#333",
        }
    )

    kappa_star = float(KAPPA_STAR_DEFAULT)
    model = DefectModel()
    x = np.linspace(-3.4, 3.4, 700)
    freqs = [0.4, 0.7, 1.0, 1.5, 2.2, 3.5]
    offsets = np.arange(len(freqs)) * 1.05
    colors = ["#2e6eb5", "#3a8fc4", "#4a9e7e", "#c9a227", "#d97b2b", "#c0392b"]

    fig = plt.figure(figsize=(13.2, 11.0))
    gs = fig.add_gridspec(
        3,
        3,
        height_ratios=[1.35, 0.58, 0.82],
        width_ratios=[1.45, 1.0, 1.0],
        hspace=0.48,
        wspace=0.30,
        left=0.065,
        right=0.975,
        top=0.855,
        bottom=0.05,
    )
    ax_main = fig.add_subplot(gs[0, :2])
    ax_lambda = fig.add_subplot(gs[0, 2])
    ax_relax = fig.add_subplot(gs[1, 0])
    ax_sigma = fig.add_subplot(gs[1, 1])
    ax_delta = fig.add_subplot(gs[1, 2])
    ax_formula = fig.add_subplot(gs[2, :])
    ax_formula.axis("off")

    E_top = offsets[-1] + 0.95
    k_spring = 2 * E_top / (3.15**2)
    xx = np.linspace(-3.25, 3.25, 500)
    V = 0.5 * k_spring * xx**2
    ax_main.plot(xx, V, color="#111", lw=2.6, zorder=10, solid_capstyle="round")
    ax_main.fill_between(xx, V, E_top + 0.4, color="#ddd5c0", alpha=0.35, zorder=0)
    ax_main.axvline(0, color="#888", lw=0.7, ls=":", alpha=0.55, zorder=1)

    rho_ref = np.exp(-(x**2) / 2.0)
    ax_main.fill_between(
        x, offsets[0], offsets[0] + 0.55 * rho_ref, color="#6b9bd1", alpha=0.14, zorder=2
    )
    ax_main.plot(
        x, offsets[0] + 0.55 * rho_ref, color="#6b9bd1", lw=1.2, ls="--", alpha=0.55, zorder=3
    )

    for f, off, c in zip(freqs, offsets, colors):
        rho_star = np.asarray(model.rho(x, f, kappa_star, s=1.0), dtype=float)
        rho_det = np.asarray(model.rho(x, f, kappa_star + 0.25, s=1.0), dtype=float)
        h = 0.78
        y_star = off + h * (rho_star / max(float(rho_star.max()), 1e-12))
        y_det = off + h * (rho_det / max(float(rho_det.max()), 1e-12))
        for s_j, a_j in [(0.25, 0.18), (0.08, 0.12)]:
            rho_s = np.asarray(model.rho(x, f, kappa_star, s=s_j), dtype=float)
            ax_main.plot(
                x,
                off + h * (rho_s / max(float(rho_s.max()), 1e-12)),
                color=c,
                lw=0.65,
                alpha=a_j,
                zorder=4,
            )
        ax_main.fill_between(x, off, y_star, color=c, alpha=0.38, zorder=5)
        ax_main.plot(x, y_star, color=c, lw=2.0, zorder=6)
        ax_main.plot(x, y_det, color=c, lw=1.15, ls="--", alpha=0.8, zorder=6)
        ax_main.axhline(off, color="#999", lw=0.35, alpha=0.45, zorder=1)
        sig = float(model.sigma(f, kappa_star))
        ax_main.text(
            3.15,
            off + 0.32,
            rf"$f={f:g}$  $\sigma={sig:.2f}$",
            fontsize=8.5,
            color=c,
            va="center",
            ha="left",
            fontweight="semibold",
            bbox=dict(
                boxstyle="round,pad=0.18",
                facecolor="#f4efe4",
                edgecolor=c,
                alpha=0.92,
                lw=0.7,
            ),
            zorder=12,
        )

    ax_main.set_xlim(-3.5, 4.35)
    ax_main.set_ylim(-0.35, E_top + 0.35)
    ax_main.set_xlabel(r"pointer misalignment $x$", fontsize=11)
    ax_main.set_ylabel(r"stacked $\rho(x,f,\kappa,s)$  (frequency ladder)", fontsize=10.5)
    ax_main.set_yticks([])
    ax_main.set_title(
        r"solid $\kappa{=}\kappa^*$ · dashed detuned · faint multi-scale $s$", fontsize=10, pad=8
    )
    handles = [
        Line2D([0], [0], color="#111", lw=2.4, label=r"misalignment potential $\propto x^{2}$"),
        Line2D(
            [0],
            [0],
            color="#6b9bd1",
            lw=1.2,
            ls="--",
            label=r"ref. Gaussian (flux gaussian_defect / QHO $n{=}0$)",
        ),
        Line2D([0], [0], color="#444", lw=1.4, ls="--", label=r"detuned $\kappa$ (broader core)"),
    ]
    ax_main.legend(
        handles=handles,
        loc="upper left",
        fontsize=8,
        frameon=True,
        facecolor="#f4efe4",
        edgecolor="#c9b98a",
        framealpha=0.95,
    )

    kappas = np.linspace(0.55, 1.15, 320)
    lam = np.array([float(lam_rate(k)) for k in kappas])
    ax_lambda.plot(kappas, lam, color="#c0392b", lw=2.3)
    ax_lambda.axvline(
        kappa_star, color="#2c3e50", ls="--", lw=1.3, label=rf"$\kappa^*\approx {kappa_star:.4f}$"
    )
    ax_lambda.fill_between(kappas, 0, lam, color="#c0392b", alpha=0.14)
    ax_lambda.annotate(
        "resonant attractor\n(healing maximized)",
        xy=(kappa_star, 1.0),
        xytext=(0.62, 0.72),
        textcoords="data",
        fontsize=7.5,
        color="#2c3e50",
        ha="center",
        arrowprops=dict(arrowstyle="->", color="#2c3e50", lw=0.9),
    )
    ax_lambda.set_xlabel(r"$\kappa$")
    ax_lambda.set_ylabel(r"$\lambda(\kappa)$")
    ax_lambda.set_title("Relaxation rate\n(survival eigenstructure)", fontsize=10.5)
    ax_lambda.legend(fontsize=8, frameon=False, loc="upper right")
    ax_lambda.grid(True, alpha=0.25)
    ax_lambda.set_ylim(0, 1.12)

    for label, k, col in [
        (r"$\kappa^*$", kappa_star, "#1e8449"),
        (r"$\kappa^*{+}0.15$", kappa_star + 0.15, "#d35400"),
        (r"$\kappa^*{+}0.30$", kappa_star + 0.30, "#6c3483"),
    ]:
        r = run_simulation(
            n_steps=140, dt=0.05, f=1.0, kappa=k, s=1.0, x0=0.5, seed=0, mode="discrete"
        )
        ax_relax.plot(r.t, r.rho, label=label, color=col, lw=1.9)
    ax_relax.set_yscale("log")
    ax_relax.set_xlabel(r"$t$")
    ax_relax.set_ylabel(r"$\rho(t)$")
    ax_relax.set_title(r"$\kappa$-tuned fractal relaxation", fontsize=10.5)
    ax_relax.legend(fontsize=8, frameon=False)
    ax_relax.grid(True, alpha=0.25, which="both")

    f_sweep = np.linspace(0.3, 4.0, 100)
    ax_sigma.plot(
        f_sweep,
        [float(model.sigma(f, kappa_star)) for f in f_sweep],
        color="#2471a3",
        lw=2.1,
        label=r"$\kappa^*$",
    )
    ax_sigma.plot(
        f_sweep,
        [float(model.sigma(f, kappa_star + 0.25)) for f in f_sweep],
        color="#c0392b",
        lw=1.7,
        ls="--",
        label=r"$\kappa^*{+}0.25$",
    )
    ax_sigma.set_xlabel(r"$f$")
    ax_sigma.set_ylabel(r"$\sigma(f,\kappa)$")
    ax_sigma.set_title("Core width (quicksand narrowing)", fontsize=10.5)
    ax_sigma.legend(fontsize=8, frameon=False)
    ax_sigma.grid(True, alpha=0.25)

    k2 = np.linspace(0.55, 1.15, 220)
    ax_delta.plot(k2, [float(fractal_exponent(k)) for k in k2], color="#117a65", lw=2.2)
    ax_delta.axvline(kappa_star, color="#2c3e50", ls="--", lw=1.2)
    ax_delta.set_xlabel(r"$\kappa$")
    ax_delta.set_ylabel(r"$\delta(\kappa)$")
    ax_delta.set_title(r"Fractal exponent  $s^{-\delta(\kappa)}$", fontsize=10.5)
    ax_delta.grid(True, alpha=0.25)

    formula = (
        r"$\rho(x,f,\kappa,s)=A(f,\kappa)\,\exp\!\left(-\frac{x^{2}}{\sigma(f,\kappa)^{2}}\right)\,s^{-\delta(\kappa)}$"
        "\n"
        r"$\sigma=\sigma_{0}(f_{0}/f)^{\alpha}(1+\gamma|\kappa-\kappa^{*}|),\quad "
        r"A=A_{0}(f/f_{0})^{\beta}(1+\mu|\kappa-\kappa^{*}|),\quad "
        r"\delta=\delta_{0}+\nu|\kappa-\kappa^{*}|$"
        "\n"
        r"$d\rho/dt=-\lambda(\kappa)\,\rho+\eta(f,\kappa)\,x^{2},\quad "
        r"\lambda(\kappa)=\lambda_{0}\exp(-|\kappa-\kappa^{*}|^{2}/\varepsilon),\quad "
        rf"\kappa^{{*}}=e/\pi - R/\pi^{{2}}\approx {kappa_star:.4f}$"
        "\n"
        r"$\rho(t{+}1)=\rho(t)\cdot e^{-\lambda\Delta t}\cdot\mathrm{clamp}(1-x^{2}/\sigma^{2},\,\mathrm{floor})\cdot s^{-\delta(\kappa)}$"
    )
    ax_formula.text(
        0.5,
        0.58,
        formula,
        transform=ax_formula.transAxes,
        ha="center",
        va="center",
        fontsize=11.2,
        linespacing=1.55,
        bbox=dict(boxstyle="round,pad=0.65", facecolor="#fffef7", edgecolor="#c9b98a", lw=1.3),
    )
    ax_formula.text(
        0.5,
        0.08,
        r"$\rho$: convex defect density  ·  $x$: pointer misalignment  ·  $f$: frequency  ·  "
        r"$\kappa$: gauge (Hopf/flux)  ·  $s$: multi-scale  ·  $\lambda$: healing rate"
        "\n"
        r"flux_hopf_lib.gaussian_defect  →  2D isotropic Gaussian core    ·    "
        r"convex_defect  →  frequency, $\kappa$-detuning, fractal $s^{-\delta}$, survival relaxation",
        transform=ax_formula.transAxes,
        ha="center",
        va="center",
        fontsize=8.2,
        color="#333",
    )

    fig.patches.append(
        plt.Rectangle(
            (0.16, 0.905),
            0.68,
            0.055,
            transform=fig.transFigure,
            facecolor="#f0d86e",
            edgecolor="none",
            zorder=0,
            alpha=0.95,
        )
    )
    fig.text(
        0.5,
        0.932,
        "Topological Convex Defect Density",
        ha="center",
        va="center",
        fontsize=18,
        fontweight="bold",
        color="#1a1a1a",
        zorder=1,
    )
    fig.text(
        0.5,
        0.875,
        "Frequency-dependent cores with multi-scale fractal relaxation in a gauged Hopf/flux medium",
        ha="center",
        va="center",
        fontsize=10.5,
        color="#333",
    )
    fig.text(
        0.975,
        0.012,
        "convex_defect + flux_hopf_lib",
        ha="right",
        va="bottom",
        fontsize=7.5,
        color="#666",
        style="italic",
    )
    fig.text(
        0.025,
        0.012,
        "successor to classical QHO eigenstate poster",
        ha="left",
        va="bottom",
        fontsize=7.5,
        color="#666",
        style="italic",
    )

    png = OUT / "topological_convex_defect_qho_successor.png"
    pdf = OUT / "topological_convex_defect_qho_successor.pdf"
    fig.savefig(png, dpi=170, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {png}")
    print(f"wrote {pdf}")


if __name__ == "__main__":
    main()
