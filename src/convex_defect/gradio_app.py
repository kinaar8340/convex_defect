"""Optional Gradio UI for interactive convex_defect exploration.

Requires: pip install -e '.[demo]'
Launch:   convex-defect gradio
"""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np


def _run_bundle(
    n_steps: int,
    dt: float,
    f: float,
    kappa: float,
    s: float,
    x0: float,
    seed: int,
    use_grid: bool,
) -> tuple[Any, str]:
    from convex_defect import KAPPA_STAR_DEFAULT, grid_to_phase_screen, run_simulation

    grid_shape = (32, 32) if use_grid else None
    r = run_simulation(
        n_steps=int(n_steps),
        dt=float(dt),
        f=float(f),
        kappa=float(kappa),
        s=float(s),
        x0=float(x0),
        seed=int(seed),
        grid_shape=grid_shape,
    )

    fig, axes = plt.subplots(2, 2, figsize=(8.5, 6.5), constrained_layout=True)
    axes[0, 0].plot(r.t, r.x)
    axes[0, 0].set_title("x(t)")
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].semilogy(r.t, np.maximum(r.rho, 1e-16))
    axes[0, 1].set_title(r"$\rho(t)$")
    axes[0, 1].grid(True, alpha=0.3, which="both")

    axes[1, 0].plot(r.t, r.H)
    axes[1, 0].set_title("H(t)")
    axes[1, 0].grid(True, alpha=0.3)

    if r.grid_final is not None:
        screen = grid_to_phase_screen(r.grid_final, r.f, r.kappa, r.s)
        im = axes[1, 1].imshow(screen, origin="lower", cmap="magma")
        axes[1, 1].set_title("ρ phase screen")
        fig.colorbar(im, ax=axes[1, 1], fraction=0.046)
    else:
        axes[1, 1].plot(r.t, r.tau)
        axes[1, 1].set_title(r"$\tau(t)$")
        axes[1, 1].grid(True, alpha=0.3)

    note = (
        f"κ*={KAPPA_STAR_DEFAULT:.4f}  κ={r.kappa:.4f}  f={r.f}  s={r.s}\n"
        f"x: {r.x[0]:.3f}→{r.x[-1]:.3f}  "
        f"ρ: {r.rho[0]:.3e}→{r.rho[-1]:.3e}  "
        f"H_final={r.H[-1]:.4f}"
    )
    return fig, note


def launch(share: bool = False, server_name: str = "127.0.0.1", server_port: int = 7860) -> None:
    try:
        import gradio as gr
    except ImportError as exc:
        raise ImportError(
            "gradio is required for the UI. Install with: pip install -e '.[demo]'"
        ) from exc

    from convex_defect import KAPPA_STAR_DEFAULT

    with gr.Blocks(title="convex_defect") as demo:
        gr.Markdown(
            "# convex_defect\n"
            "Frequency-dependent topological convex defect density "
            "(pointer → ρ → holonomy). "
            f"Resonant **κ\\*** ≈ {KAPPA_STAR_DEFAULT:.4f}."
        )
        with gr.Row():
            n_steps = gr.Slider(20, 400, value=120, step=10, label="steps")
            dt = gr.Slider(0.01, 0.2, value=0.05, step=0.01, label="dt")
            f = gr.Slider(0.2, 4.0, value=1.0, step=0.1, label="frequency f")
        with gr.Row():
            kappa = gr.Slider(0.5, 1.2, value=float(KAPPA_STAR_DEFAULT), step=0.01, label="κ")
            s = gr.Slider(0.05, 1.0, value=1.0, step=0.05, label="scale s")
            x0 = gr.Slider(0.0, 1.5, value=0.4, step=0.05, label="x0")
        with gr.Row():
            seed = gr.Number(value=0, label="seed", precision=0)
            use_grid = gr.Checkbox(value=True, label="2D misalignment grid")
            btn = gr.Button("Run", variant="primary")
        plot = gr.Plot(label="traces")
        summary = gr.Textbox(label="summary", lines=2)

        btn.click(
            _run_bundle,
            inputs=[n_steps, dt, f, kappa, s, x0, seed, use_grid],
            outputs=[plot, summary],
        )
        demo.load(
            _run_bundle,
            inputs=[n_steps, dt, f, kappa, s, x0, seed, use_grid],
            outputs=[plot, summary],
        )

    demo.launch(share=share, server_name=server_name, server_port=server_port)


if __name__ == "__main__":
    launch()
