"""Thin CLI for convex_defect demos and sweeps.

Examples
--------
    convex-defect demo
    convex-defect run --steps 100 --freq 1.5 --kappa 0.85
    convex-defect sweep-f --fmin 0.5 --fmax 3 --n 10
    convex-defect gradio
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="convex-defect",
        description="Frequency-dependent topological convex defect density tools",
    )
    p.add_argument(
        "--version",
        action="store_true",
        help="Print package version and exit",
    )
    sub = p.add_subparsers(dest="cmd")

    demo = sub.add_parser("demo", help="Run matplotlib demo (writes outputs/)")
    demo.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output directory (default: <repo>/outputs or ./outputs)",
    )

    run = sub.add_parser("run", help="Single simulation; print summary JSON")
    run.add_argument("--steps", type=int, default=100)
    run.add_argument("--dt", type=float, default=0.05)
    run.add_argument("--freq", type=float, default=1.0)
    run.add_argument("--kappa", type=float, default=None)
    run.add_argument("--s", type=float, default=1.0)
    run.add_argument("--x0", type=float, default=0.3)
    run.add_argument("--seed", type=int, default=0)
    run.add_argument(
        "--mode",
        choices=("discrete", "euler"),
        default="discrete",
    )
    run.add_argument(
        "--grid",
        type=str,
        default=None,
        help="Grid shape as H,W or N (e.g. 32,32 or 64)",
    )
    run.add_argument("--json", action="store_true", help="Machine-readable summary")

    sf = sub.add_parser("sweep-f", help="Frequency sweep of final holonomy")
    sf.add_argument("--fmin", type=float, default=0.5)
    sf.add_argument("--fmax", type=float, default=3.0)
    sf.add_argument("--n", type=int, default=8)
    sf.add_argument("--steps", type=int, default=80)
    sf.add_argument("--seed", type=int, default=0)

    sk = sub.add_parser("sweep-k", help="κ sweep of final ρ and H")
    sk.add_argument("--kmin", type=float, default=0.6)
    sk.add_argument("--kmax", type=float, default=1.1)
    sk.add_argument("--n", type=int, default=8)
    sk.add_argument("--steps", type=int, default=80)
    sk.add_argument("--seed", type=int, default=0)

    sub.add_parser("gradio", help="Launch optional Gradio UI (requires [demo] extra)")

    return p


def _parse_grid(spec: str | None) -> tuple[int, ...] | None:
    if not spec:
        return None
    parts = [int(x.strip()) for x in spec.split(",")]
    if len(parts) not in (1, 2):
        raise SystemExit("--grid must be N or H,W")
    return tuple(parts)


def _cmd_demo(out: Path | None) -> int:
    # Import demo as script logic
    root = Path(__file__).resolve().parents[2]
    demo_path = root / "examples" / "convex_defect_demo.py"
    if not demo_path.is_file():
        # installed package: run inline minimal demo
        return _cmd_demo_inline(out)

    import runpy

    sys.path.insert(0, str(root / "src"))
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    # Example writes under <repo>/outputs; --out is reserved for installed fallback
    if out is not None:
        print(f"note: example demo writes to {root / 'outputs'} (ignoring --out={out})")
    g = runpy.run_path(str(demo_path), run_name="__main__")
    del g
    return 0


def _cmd_demo_inline(out: Path | None) -> int:
    """Fallback if examples/ is not next to the package."""
    from convex_defect import KAPPA_STAR_DEFAULT, run_simulation

    dest = out or Path("outputs")
    dest.mkdir(parents=True, exist_ok=True)
    r = run_simulation(n_steps=100, f=1.0, x0=0.4, seed=0)
    summary = {
        "kappa_star": KAPPA_STAR_DEFAULT,
        "rho_final": float(r.rho[-1]),
        "H_final": float(r.H[-1]),
        "x_final": float(r.x[-1]),
    }
    path = dest / "cli_demo_summary.json"
    path.write_text(json.dumps(summary, indent=2))
    print(f"wrote {path}")
    print(json.dumps(summary, indent=2))
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    from convex_defect import run_simulation

    grid = _parse_grid(args.grid)
    r = run_simulation(
        n_steps=args.steps,
        dt=args.dt,
        f=args.freq,
        kappa=args.kappa,
        s=args.s,
        x0=args.x0,
        seed=args.seed,
        mode=args.mode,
        grid_shape=grid,
    )
    summary: dict[str, Any] = {
        "f": r.f,
        "kappa": r.kappa,
        "s": r.s,
        "steps": args.steps,
        "x0": float(r.x[0]),
        "x_final": float(r.x[-1]),
        "rho0": float(r.rho[0]),
        "rho_final": float(r.rho[-1]),
        "H_final": float(r.H[-1]),
        "tau_final": float(r.tau[-1]),
        "lambda": float(r.lambda_trace[0]),
        "mode": r.metadata.get("mode"),
    }
    if r.grid_final is not None:
        summary["grid_shape"] = list(r.grid_final.shape)
        summary["grid_mean_final"] = float(r.grid_final.mean())
    if args.json:
        print(json.dumps(summary, indent=2))
    else:
        print(
            f"κ={summary['kappa']:.4f}  f={summary['f']}  s={summary['s']}\n"
            f"  x:   {summary['x0']:.4f} → {summary['x_final']:.4f}\n"
            f"  ρ:   {summary['rho0']:.4e} → {summary['rho_final']:.4e}\n"
            f"  H:   {summary['H_final']:.4f}\n"
            f"  τ:   {summary['tau_final']:.4f}\n"
            f"  λ:   {summary['lambda']:.4f}"
        )
    return 0


def _cmd_sweep_f(args: argparse.Namespace) -> int:
    import numpy as np

    from convex_defect import sweep_frequency

    freqs = np.linspace(args.fmin, args.fmax, args.n)
    results = sweep_frequency(freqs, n_steps=args.steps, seed=args.seed)
    print(f"{'f':>8}  {'H_final':>10}  {'rho_final':>12}")
    for f, r in zip(freqs, results):
        print(f"{float(f):8.3f}  {r.H[-1]:10.4f}  {r.rho[-1]:12.4e}")
    return 0


def _cmd_sweep_k(args: argparse.Namespace) -> int:
    import numpy as np

    from convex_defect import KAPPA_STAR_DEFAULT, sweep_kappa

    kappas = np.linspace(args.kmin, args.kmax, args.n)
    results = sweep_kappa(kappas, n_steps=args.steps, seed=args.seed)
    print(f"{'kappa':>8}  {'lambda':>8}  {'H_final':>10}  {'rho_final':>12}  note")
    for k, r in zip(kappas, results):
        note = "κ*" if abs(float(k) - KAPPA_STAR_DEFAULT) < 0.02 else ""
        print(
            f"{float(k):8.4f}  {r.lambda_trace[0]:8.4f}  "
            f"{r.H[-1]:10.4f}  {r.rho[-1]:12.4e}  {note}"
        )
    return 0


def _cmd_gradio() -> int:
    try:
        from convex_defect.gradio_app import launch
    except ImportError as exc:
        print(
            "Gradio UI requires the demo extra:\n"
            "  pip install -e '.[demo]'\n"
            f"({exc})",
            file=sys.stderr,
        )
        return 1
    launch()
    return 0


def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.version:
        from convex_defect import __version__

        print(__version__)
        return

    if args.cmd is None:
        parser.print_help()
        return

    if args.cmd == "demo":
        raise SystemExit(_cmd_demo(args.out))
    if args.cmd == "run":
        raise SystemExit(_cmd_run(args))
    if args.cmd == "sweep-f":
        raise SystemExit(_cmd_sweep_f(args))
    if args.cmd == "sweep-k":
        raise SystemExit(_cmd_sweep_k(args))
    if args.cmd == "gradio":
        raise SystemExit(_cmd_gradio())

    parser.print_help()


if __name__ == "__main__":
    main()
