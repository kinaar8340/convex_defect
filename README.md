# convex_defect

Frequency-dependent topological **convex defect density** with \(\kappa\)-tuned fractal relaxation.

Extends the `mystery` / `flux_trajectoid` research line: pointer misalignment produces Gaussian + multi-scale fractal defects that increase topological opacity, accumulate holonomy along geodesics, and heal under survival-eigenstructure rates.

## Install

```bash
cd ~/Projects/convex_defect
pip install -e ".[dev]"
# optional Gradio UI:
# pip install -e ".[demo]"
```

## Quick start

```python
from convex_defect import DefectModel, run_simulation

model = DefectModel()  # defaults: κ* ≈ 0.8513
rho = model.rho(x=0.1, f=1.0, kappa=0.85, s=1.0)
tau = model.opacity(x=0.1, f=1.0, kappa=0.85)

result = run_simulation(n_steps=100, f=1.0, x0=0.4, seed=0)
print(result.H[-1], result.rho[-1])

# multi-scale dynamical ρ(s)
ms = run_simulation(n_steps=80, multi_scale=True, n_scales=16, grid_shape=(32, 32), x0=0.5)
print(ms.rho_spectrum_final.shape, ms.screen_final.shape)
```

```bash
# plots → outputs/
python examples/convex_defect_demo.py

# CLI
convex-defect run --steps 100 --freq 1.5
convex-defect sweep-f --fmin 0.5 --fmax 3 --n 10
convex-defect sweep-k --kmin 0.6 --kmax 1.1
convex-defect demo
convex-defect gradio   # needs pip install -e '.[demo]'

pytest
```

## Layout

| Path | Role |
|------|------|
| `docs/convex_defect_theory.md` | Equations + mapping to oam_flux / mystery / trajectoids |
| `src/convex_defect/defect_density.py` | \(\rho(x,f,\kappa,s)\), \(\sigma\), \(A\), \(\tau\) |
| `src/convex_defect/multi_scale_field.py` | Dynamical multi-scale \(\rho(s)\) / spatial \(\rho(x_{ij},s)\) |
| `src/convex_defect/holonomy_accumulator.py` | Fractal holonomy \(H(t)\) (pure accumulation, double fractal) |
| `src/convex_defect/relaxation_dynamics.py` | Survival-eigenstructure relaxation + clamp |
| `src/convex_defect/simulator.py` | Coupled pointer + \(\rho\) + \(H\); optional 1D/2D grid |
| `src/convex_defect/cli.py` | Thin CLI |
| `src/convex_defect/gradio_app.py` | Optional interactive UI |
| `examples/convex_defect_demo.py` | Plots: Gaussian vs \(f\), relaxation, \(H(t)\), phase screens |
| `tests/test_convex_defect.py` | Unit tests |

## Conceptual links

- **Holonomy gaps** / \(\kappa^*\) — `mystery` residual \(\kappa\) sweeps
- **Survival eigenstructure** — `pde_survival_eigenstructure` relaxation rates
- **OAM flux / turbulence** — `oam_flux` + `flux_trajectoid` propagation screens (`grid_to_phase_screen`)
- **Trajectoid geodesics** — rolling paths on a fractally textured manifold

See `docs/convex_defect_theory.md` for the full equation set.
