# convex_defect

Frequency-dependent topological **convex defect density** with \(\kappa\)-tuned fractal relaxation.

Extends the `mystery` / `flux_trajectoid` research line: pointer misalignment produces Gaussian + multi-scale fractal defects that increase topological opacity, accumulate holonomy along geodesics, and heal under survival-eigenstructure rates.

## Install

```bash
cd ~/Projects/convex_defect
pip install -e ".[dev]"
# optional Gradio demo:
# pip install -e ".[demo]"
```

## Quick start

```python
from convex_defect import DefectModel, defect_density

model = DefectModel()  # defaults: κ* ≈ 0.8513
rho = model.rho(x=0.1, f=1.0, kappa=0.85, s=1.0)
tau = model.opacity(x=0.1, f=1.0, kappa=0.85)
```

```bash
python examples/convex_defect_demo.py
pytest
```

## Layout

| Path | Role |
|------|------|
| `docs/convex_defect_theory.md` | Equations + mapping to oam_flux / mystery / trajectoids |
| `src/convex_defect/defect_density.py` | \(\rho(x,f,\kappa,s)\), \(\sigma\), \(A\), \(\tau\) |
| `src/convex_defect/holonomy_accumulator.py` | Fractal holonomy \(H(t)\) |
| `src/convex_defect/relaxation_dynamics.py` | Survival-eigenstructure relaxation |
| `src/convex_defect/simulator.py` | Toy 1D/2D pointer + defect + holonomy sim |
| `examples/convex_defect_demo.py` | Plots: Gaussian vs \(f\), relaxation, \(H(t)\) |

## Conceptual links

- **Holonomy gaps** / \(\kappa^*\) — `mystery` residual \(\kappa\) sweeps
- **Survival eigenstructure** — `pde_survival_eigenstructure` relaxation rates
- **OAM flux / turbulence** — `oam_flux` + `flux_trajectoid` propagation screens
- **Trajectoid geodesics** — rolling paths on a fractally textured manifold

See `docs/convex_defect_theory.md` for the full equation set.
