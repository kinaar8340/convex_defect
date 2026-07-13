"""convex_defect — frequency-dependent topological convex defect density.

Extends mystery / flux_trajectoid ideas: misalignment-driven Gaussian + fractal
defects, holonomy accumulation, survival-eigenstructure relaxation, and
multi-scale dynamical ρ(s) fields.
"""

from __future__ import annotations

from .defect_density import (
    KAPPA_STAR_DEFAULT,
    DefectModel,
    DefectParams,
    amplitude,
    defect_density,
    fractal_exponent,
    gaussian_area_opacity,
    holonomy_gap,
    opacity,
    sigma_width,
)
from .holonomy_accumulator import (
    HolonomyAccumulator,
    HolonomyParams,
    accumulate_holonomy,
    gamma_H,
    holonomy_integrand,
)
from .multi_scale_field import (
    MultiScaleDefectField,
    MultiScaleParams,
    ScaleBins,
    multi_scale_phase_screen,
)
from .relaxation_dynamics import (
    RelaxationDynamics,
    RelaxationParams,
    continuous_rhs,
    discrete_step,
    discrete_trajectory,
    eta_source,
    integrate_ode,
    lambda_kappa_curve,
    lambda_rate,
    misalignment_multiplier,
)
from .simulator import (
    ConvexDefectSimulator,
    PointerDynamics,
    PointerParams,
    SimConfig,
    SimResult,
    grid_to_phase_screen,
    pointer_trajectory,
    run_simulation,
    sweep_frequency,
    sweep_kappa,
)

__version__ = "0.2.0"

__all__ = [
    # density
    "KAPPA_STAR_DEFAULT",
    "DefectParams",
    "DefectModel",
    "sigma_width",
    "amplitude",
    "fractal_exponent",
    "defect_density",
    "opacity",
    "gaussian_area_opacity",
    "holonomy_gap",
    # multi-scale
    "ScaleBins",
    "MultiScaleParams",
    "MultiScaleDefectField",
    "multi_scale_phase_screen",
    # holonomy
    "HolonomyParams",
    "HolonomyAccumulator",
    "gamma_H",
    "holonomy_integrand",
    "accumulate_holonomy",
    # relaxation
    "RelaxationParams",
    "RelaxationDynamics",
    "lambda_rate",
    "eta_source",
    "misalignment_multiplier",
    "discrete_step",
    "discrete_trajectory",
    "continuous_rhs",
    "integrate_ode",
    "lambda_kappa_curve",
    # simulator
    "PointerParams",
    "PointerDynamics",
    "SimConfig",
    "SimResult",
    "ConvexDefectSimulator",
    "run_simulation",
    "pointer_trajectory",
    "grid_to_phase_screen",
    "sweep_frequency",
    "sweep_kappa",
    "__version__",
]
