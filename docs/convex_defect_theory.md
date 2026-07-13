# Convex Defect Theory: Frequency-Dependent Topological Texture in the Gauged Hopf Medium

**Status**: Working notes — speculative but tightly coupled to existing `mystery` and `flux_trajectoid` infrastructure.

## Overview

When the global pointer (z-axis / resonant alignment) drifts from the attractor state of the gauged Hopf lattice, the otherwise smooth resonant medium develops **convex topological defects**. These defects increase topological opacity, scatter propagating modes (especially high-frequency), and accumulate fractal holonomy along geodesics.

The model unifies:

- Misalignment-driven roughness (Gaussian core)
- Frequency-dependent “quicksand” behavior
- Gauge-parameter (\(\kappa\)) tuning via holonomy gaps
- Multi-scale fractal texture
- Relaxation/healing governed by survival eigenstructure

This framework maps naturally onto turbulence in `flux_trajectoid` (Kolmogorov screens, OAM fidelity degradation) and the holonomy/PDE dynamics in `mystery`.

## Variables

| Symbol | Meaning |
|--------|---------|
| \(x\) | Global pointer misalignment angle (from resonant alignment) |
| \(f\) | Normalized frequency (\(f > 0\)) |
| \(\kappa\) | Gauge parameter (resonant optimum \(\kappa^* \approx 0.8513\)) |
| \(s\) | Scale variable (normalized spatial or frequency scale, \(s > 0\)) |
| \(t\) | Time / relaxation step |

Default constants (overridable in `DefectParams`):

| Symbol | Default | Role |
|--------|---------|------|
| \(\kappa^*\) | \(e/\pi - R/\pi^2 \approx 0.8513\) | Holonomy-gap null (same family as `oam_flux.LatticeConstants`) |
| \(\sigma_0, f_0, A_0\) | \(1, 1, 1\) | Reference width, frequency, amplitude |
| \(\alpha, \beta\) | \(0.5, 0.25\) | Frequency exponents for width / amplitude |
| \(\gamma, \mu, \nu\) | \(1.0, 0.5, 0.3\) | Strength of \(\kappa\)-detuning effects |
| \(\delta_0, \epsilon\) | \(0.2, 0.05\) | Baseline fractal exponent; relaxation sharpness |
| \(\lambda_0, \eta_0\) | \(1.0, 0.1\) | Peak healing rate; misalignment source strength |

## Core Equations

### 1. Frequency- and gauge-dependent defect density

Gaussian core + fractal multi-scale modulation:

\[
\rho(x, f, \kappa, s)
  = A(f, \kappa)\,
    \exp\!\left(-\frac{x^2}{\sigma(f, \kappa)^2}\right)\,
    s^{-\delta(\kappa)}
\]

**Width** (higher \(f\) → narrower core; detuned \(\kappa\) → broader roughness):

\[
\sigma(f, \kappa)
  = \sigma_0 \left(\frac{f_0}{f}\right)^{\alpha}
    \bigl(1 + \gamma\,|\kappa - \kappa^*|\bigr)
\]

**Amplitude**:

\[
A(f, \kappa)
  = A_0 \left(\frac{f}{f_0}\right)^{\beta}
    \bigl(1 + \mu\,|\kappa - \kappa^*|\bigr)
\]

**Fractal exponent** (higher misalignment → stronger multi-scale roughness):

\[
\delta(\kappa) = \delta_0 + \nu\,|\kappa - \kappa^*|
\]

The factor \(s^{-\delta(\kappa)}\) introduces self-similar convex defects across scales. Higher frequencies couple preferentially to smaller \(s\) (quicksand / fine-scale texture).

### 2. Integrated topological opacity

\[
\tau(x, f, \kappa)
  = \int_{s_{\min}}^{s_{\max}}
      \rho(x, f, \kappa, s)\, ds
\]

The pure power-law \(\int_0^\infty s^{-\delta}\,ds\) diverges, so the implementation integrates over a finite scale window \([s_{\min}, s_{\max}]\) (analytic antiderivative when \(\delta \neq 1\)):

\[
\int_{s_{\min}}^{s_{\max}} s^{-\delta}\,ds
  =
  \begin{cases}
    \dfrac{s_{\max}^{1-\delta} - s_{\min}^{1-\delta}}{1-\delta}
      & \delta \neq 1 \\[0.6em]
    \ln(s_{\max}/s_{\min}) & \delta = 1
  \end{cases}
\]

Thus

\[
\tau(x, f, \kappa)
  = A(f, \kappa)\,
    \exp\!\left(-\frac{x^2}{\sigma(f, \kappa)^2}\right)\,
    I_\delta(s_{\min}, s_{\max}).
\]

A related **Gaussian area** (misalignment-integrated opacity at fixed scale) is

\[
\tau_x(f, \kappa; s)
  = \int_{-\infty}^{\infty} \rho(x, f, \kappa, s)\, dx
  = A(f, \kappa)\, \sigma(f, \kappa)\, \sqrt{\pi}\,
    s^{-\delta(\kappa)}.
\]

### 3. Holonomy accumulation (fractal topological memory)

Along a geodesic / trajectory with misalignment history \(x(t)\):

\[
H(t)
  = H_0
  + \int_0^t
      \gamma_H(f)\,
      \rho\bigl(x(t'), f, \kappa, s\bigr)\,
      s^{-\delta(\kappa)}\,
      dt'
\]

Note: \(\rho\) already carries one factor of \(s^{-\delta}\), so the integrand scales as \(s^{-2\delta}\) — intentional scale-dependent memory (finer scales accumulate richer holonomy when \(\kappa\) is detuned). The coupling \(\gamma_H(f)\) may grow with frequency (high-\(f\) modes “feel” the texture more).

**Tunable double fractal**: `HolonomyParams.holonomy_fractal_multiplier` (default \(1.0\)) multiplies the extra \(s^{-\delta}\) power. Set to \(0\) for single-fractal holonomy (\(H \propto \rho\) only) if the double exponent proves too aggressive in later sweeps.

In discrete form (compatible with survival probes):

\[
H_{n+1}
  = H_n
  + \gamma_H(f)\,
    \rho(x_n, f, \kappa, s)\,
    s^{-\delta(\kappa)}\,
    \Delta t.
\]

### 4. Relaxation / healing (survival eigenstructure)

When the pointer realigns (\(x \to 0\)), defects heal. Continuous form:

\[
\frac{d\rho}{dt}
  = -\lambda(\kappa)\,\rho(x, f, \kappa, s)
    + \eta(f, \kappa)\, x^2
\]

Discrete survival form (directly compatible with existing probes):

\[
\rho(t+1)
  = \rho(t)\,
    e^{-\lambda(\kappa)\,\Delta t}\,
    \mathrm{clamp}\!\left(
      1 - \frac{x(t)^2}{\sigma(f, \kappa)^2},\;
      \textit{floor}
    \right)\,
    s^{-\delta(\kappa)}
\]

**Clamp**: soft floor default `misalignment_floor ≈ 0.01` in `relaxation_dynamics.py` so large \(|x|\) cannot drive a negative multiplier or hard-zero \(\rho\) in one step. Floor \(= 0\) recovers pure \(\max(0,\cdot)\). Not applied inside `defect_density.py`.

**Relaxation rate** peaks sharply near resonant \(\kappa^*\):

\[
\lambda(\kappa)
  = \lambda_0\,
    \exp\!\left(
      -\frac{|\kappa - \kappa^*|^2}{\epsilon}
    \right)
\]

This matches survival eigenstructure and holonomy-gap minimization: near \(\kappa^*\), the holonomy gap

\[
B(\kappa) = \pi^2 \bigl(e/\pi - \kappa\bigr)
\]

is minimized (modulo residual \(R\)), and healing is fastest.

Optional source amplitude:

\[
\eta(f, \kappa)
  = \eta_0 \left(\frac{f}{f_0}\right)^{\beta}
    \bigl(1 + \mu\,|\kappa - \kappa^*|\bigr)
\]

so detuned / high-frequency channels source defects more strongly under residual misalignment.

## Physical & Topological Interpretation

- **Convex defects** emerge as geometric consequences of a non-zero holonomy gap when \(\kappa\) drifts from optimum.
- **Higher-frequency quicksand**: Smaller wavelengths couple preferentially to fine-scale (\(s\)) convex bumps — opacity and holonomy grow with \(f\) through \(A\) and \(\gamma_H\).
- **Fractal texture**: Self-similar defects across scales produce scale-dependent holonomy — waves carry “memory” of the misaligned phase that is richer at higher resolutions.
- **Healing**: When the pointer realigns, survival eigenstructure drives exponential decay of defects. Residual fractal scars may persist due to topological protection (Hopf winding / Skyrme-like structures).
- **Lake / Aether analogy**: Mirror-flat surface = perfect alignment, zero defects. Sun-driven ripples + convex bumps = misalignment-induced texture. Submersion (“fish on”) = strong internal coupling that populates high-\(f\) defect modes.

## Mapping to Existing Infrastructure

| Concept | Where it lives | Role in convex_defect |
|---------|----------------|------------------------|
| Holonomy gaps | `mystery/notes/skyrme_holonomy_derivation.md`, `residual_kappa_sweep.py` | Primary source of convex defect amplitude via \(\lvert\kappa - \kappa^*\rvert\) |
| Survival eigenstructure | `pde_survival_eigenstructure.py`, `kappa_survival_sweep.py` | Governs \(\lambda(\kappa)\) and exponential defect decay |
| PDE relaxation probes | `pde_relaxation_probe.py`, `pde_structured_ic_probe.py` | Natural environment for defect evolution and geodesic holonomy |
| OAM flux / helical lattice | `oam_flux` (`lattice`, `coupling`, `emergence`) | Convex defects as structured turbulence degrading OAM fidelity in a frequency- and holonomy-dependent way |
| Turbulence propagation | `flux_trajectoid.propagation.simulator` (Kolmogorov screens) | Replace / augment generic phase screens with \(\rho\)-driven structure |
| Trajectoid geodesics | `flux_trajectoid.shell` rolling paths | Paths become geodesics on a fractally textured manifold; accumulated \(H\) feeds recovery metrics |
| \(\kappa^*\), residual \(R\) | `oam_flux.constants.LatticeConstants` | Shared numerical anchor (\(\kappa^* \approx 0.8513\)) |

### Conceptual integration sketch

1. **Pointer** \(x(t)\) — global alignment error (or local mean phase / twist bias on the lattice).
2. **Defect field** \(\rho\) — multi-scale convex roughness parameterized by \(f, \kappa, s\).
3. **Opacity** \(\tau\) — effective scattering strength for a mode of frequency \(f\).
4. **Holonomy** \(H\) — path-ordered topological memory along a trajectoid geodesic or lattice trajectory.
5. **Relaxation** — when control / survival dynamics drive \(x \to 0\) and \(\kappa \to \kappa^*\), \(\rho\) decays at rate \(\lambda(\kappa)\).

For `flux_trajectoid` turbulence sweeps, a structured phase screen can be built by sampling \(\rho\) (or \(\tau\)) over a spatial grid and frequency band instead of pure Kolmogorov noise — preserving the same fidelity metrics (`overlap_fidelity`, `oam_fidelity`, etc.).

## Analytic limits (useful checks)

1. **Aligned pointer**: \(x = 0\) → \(\rho = A\, s^{-\delta}\) (scale-only texture; no Gaussian suppression).
2. **Resonant gauge**: \(\kappa = \kappa^*\) → \(\sigma = \sigma_0 (f_0/f)^\alpha\), \(A = A_0 (f/f_0)^\beta\), \(\delta = \delta_0\), \(\lambda = \lambda_0\).
3. **Gaussian area**: \(\int \rho\, dx = A\,\sigma\sqrt{\pi}\, s^{-\delta}\).
4. **Opacity vs \(f\)**: for fixed \(x, \kappa\), \(\tau \propto A(f)\, e^{-x^2/\sigma(f)^2}\, I_\delta\); high \(f\) narrows \(\sigma\) (core tightens) while \(A\) may grow — “quicksand” is the competition of these trends.
5. **Healing at \(x = 0\)**: continuous \(\dot\rho = -\lambda\rho\) → pure exponential; discrete form multiplies by \(e^{-\lambda\Delta t}\, s^{-\delta}\).

## Implementation map

| Module | Responsibility |
|--------|----------------|
| `defect_density.py` | `DefectParams`, \(\sigma, A, \delta, \rho, \tau, \tau_x\) |
| `holonomy_accumulator.py` | Discrete / integrated \(H(t)\) with \(\gamma_H(f)\) |
| `relaxation_dynamics.py` | \(\lambda(\kappa)\), ODE and discrete survival steps |
| `simulator.py` | Coupled pointer \(x(t)\) + \(\rho\) + \(H\) + \(\tau\); optional 1D/2D local-misalignment grid; `grid_to_phase_screen` for flux_trajectoid hooks |

## Next steps

1. ✅ Theory doc + `defect_density.py`
2. `relaxation_dynamics.py` + `holonomy_accumulator.py`
3. Minimal simulator coupling pointer dynamics to \(\rho\)
4. Frequency-sweep and fractal-holonomy visualizations
5. Conceptual hook into `flux_trajectoid` turbulence (structured screens from \(\rho\))
6. Optional: live coupling to `oam_flux` lattice twist fields

## References & related notes

- `mystery/notes/emergent_signatures.md`, `kappa_sim_interpretation.md`, `skyrme_holonomy_derivation.md`
- `mystery` scripts: `pde_survival_eigenstructure.py`, `kappa_survival_sweep.py`, `residual_kappa_sweep.py`
- `oam_flux` lattice / emergence / holonomy-gap constants
- `flux_trajectoid` architecture and turbulence metrics

*This document is living theory — equations and mappings will evolve with simulation results.*
