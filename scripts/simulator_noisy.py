"""Simple SteMy iPSC-CM differentiation simulator v2.

Implements stage-score equations from the cleaned v1 simulator documentation:
parameters -> Q_seed -> Q_meso -> Q_prog -> Q_spec -> cTnT%.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import math

import numpy as np


# All biological priors live here so equations stay clean.
# confidence: confirmed / inferred / placeholder / simulator-only
PARAM_SPECS: dict[str, dict[str, Any]] = {
    # Stage 0-1: passaging / seeding context
    "cpsc_pct": {
        "default": 85.0,
        "optimum": 85.0,
        "tolerance": 15.0,
        "bounds": (70.0, 100.0),
        "confidence": "inferred",
        "optimizable": False,
    },
    "versene_dur_min": {
        "default": 4.0,
        "optimum": 4.0,
        "tolerance": 1.5,
        "bounds": (2.0, 6.0),
        "confidence": "inferred",
        "optimizable": False,
    },
    "seed_density_mcells_per_well": {
        "default": 1.9,
        "optimum": 1.9,
        "tolerance": 0.4,
        "bounds": (1.0, 2.8),
        "confidence": "confirmed",
        "optimizable": False,
    },
    # PLACEHOLDER: ROCK dose/timing should be confirmed by wet-lab team.
    "rock_conc_uM": {
        "default": 10.0,
        "optimum": 10.0,
        "tolerance": 5.0,
        "bounds": (0.0, 20.0),
        "confidence": "placeholder",
        "optimizable": True,
        "needs_confirmation": True,
    },
    # PLACEHOLDER: assumes ROCK is used for post-replating recovery.
    "rock_dur_hr": {
        "default": 24.0,
        "optimum": 24.0,
        "tolerance": 12.0,
        "bounds": (0.0, 48.0),
        "confidence": "placeholder",
        "optimizable": True,
        "needs_confirmation": True,
    },
    # PLACEHOLDER: D0 confluence target should be confirmed.
    "d0_confluence_pct": {
        "default": 95.0,
        "optimum": 95.0,
        "tolerance": 15.0,
        "bounds": (60.0, 120.0),
        "confidence": "placeholder",
        "optimizable": False,
        "needs_confirmation": True,
    },
    "d0_morphology_score": {
        "default": 1.0,
        "bounds": (0.0, 1.0),
        "confidence": "placeholder",
        "optimizable": False,
        "needs_confirmation": True,
    },

    # Stage 2: mesoderm induction
    "chir_start_day": {
        "default": 0.0,
        "optimum": 0.0,
        "tolerance": 0.25,
        "bounds": (-0.25, 0.25),
        "confidence": "confirmed",
        "optimizable": False,  # held constant in first version
    },
    "chir_conc_uM": {
        "default": 12.0,
        "optimum": 12.0,
        "tolerance": 4.0,
        "bounds": (8.0, 14.0),
        "confidence": "inferred",
        "optimizable": True,
    },
    "chir_dur_hr": {
        "default": 24.0,
        "optimum": 24.0,
        "tolerance": 4.0,
        "bounds": (18.0, 30.0),
        "confidence": "confirmed",
        "optimizable": True,
    },
    "insulin_start_day": {
        "default": 0.0,
        "optimum": 0.0,
        "tolerance": 0.25,
        "bounds": (-0.25, 0.25),
        "confidence": "confirmed",
        "optimizable": True,
    },
    # PLACEHOLDER: normalized insulin level, not absolute molar concentration.
    # 1.0 = standard insulin-containing medium; 0.0 = no insulin.
    "insulin_conc": {
        "default": 1.0,
        "optimum": 1.0,
        "tolerance": 0.5,
        "bounds": (0.0, 2.0),
        "confidence": "placeholder",
        "optimizable": True,
        "needs_confirmation": True,
    },
    "insulin_dur_hr": {
        "default": 24.0,
        "optimum": 24.0,
        "tolerance": 8.0,
        "bounds": (0.0, 48.0),
        "confidence": "confirmed",
        "optimizable": True,
    },

    # Stage 3: cardiac progenitor / SHF modulation
    "bfgf_start_day": {
        "default": 2.75,
        "optimum": 2.75,
        "tolerance": 0.25,
        "bounds": (2.25, 3.25),
        "confidence": "confirmed",
        "optimizable": True,
    },
    # PLACEHOLDER: bFGF concentration must be confirmed.
    "bfgf_conc_ng_ml": {
        "default": 10.0,
        "optimum": 10.0,
        "tolerance": 10.0,
        "bounds": (0.0, 50.0),
        "confidence": "placeholder",
        "optimizable": True,
        "needs_confirmation": True,
    },
    # PLACEHOLDER: assumes bFGF acts mainly over D2.75-D3.75.
    "bfgf_dur_hr": {
        "default": 24.0,
        "optimum": 24.0,
        "tolerance": 12.0,
        "bounds": (0.0, 72.0),
        "confidence": "placeholder",
        "optimizable": True,
        "needs_confirmation": True,
    },
    # PLACEHOLDER: hypoxia timing/severity should be confirmed.
    "hypoxia_start_day": {
        "default": 3.25,
        "optimum": 3.25,
        "tolerance": 0.75,
        "bounds": (2.75, 4.75),
        "confidence": "placeholder",
        "optimizable": True,
        "needs_confirmation": True,
    },
    "hypoxia_dur_hr": {
        "default": 24.0,
        "optimum": 24.0,
        "tolerance": 12.0,
        "bounds": (1.0, 48.0),
        "confidence": "placeholder",
        "optimizable": True,
        "needs_confirmation": True,
    },
    "hypoxia_pct": {
        "default": 3.0,
        "optimum": 3.0,
        "tolerance": 2.0,
        "bounds": (1.0, 5.0),
        "confidence": "placeholder",
        "optimizable": True,
        "needs_confirmation": True,
    },

    # Stage 4: cardiac specification / Wnt inhibition
    "iwp2_start_day": {
        "default": 3.75,
        "optimum": 3.75,
        "tolerance": 0.25,
        "bounds": (3.25, 4.25),
        "confidence": "confirmed",
        "optimizable": True,
    },
    "iwp2_conc_uM": {
        "default": 5.0,
        "optimum": 5.0,
        "tolerance": 2.0,
        "bounds": (3.0, 7.5),
        "confidence": "inferred",
        "optimizable": True,
    },
    "iwp2_dur_hr": {
        "default": 48.0,
        "optimum": 48.0,
        "tolerance": 12.0,
        "bounds": (24.0, 72.0),
        "confidence": "confirmed",
        "optimizable": True,
    },
}


@dataclass(frozen=True)
class StageResult:
    """Stage score plus component diagnostics."""

    score: float
    components: dict[str, float]


def sigmoid(z: float) -> float:
    """Numerically stable logistic transform."""
    if z >= 0:
        ez = math.exp(-z)
        return 1.0 / (1.0 + ez)
    ez = math.exp(z)
    return ez / (1.0 + ez)


def clip01(x: float) -> float:
    return float(np.clip(x, 0.0, 1.0))


def bounded_score(value: float, optimum: float, tolerance: float) -> float:
    """Bounded optimum-centered score: 1 near optimum, 0 far away."""
    if tolerance <= 0:
        raise ValueError("tolerance must be positive.")
    score = 1.0 - ((float(value) - float(optimum)) / float(tolerance)) ** 2
    return clip01(score)


def score_param(params: dict[str, float], name: str) -> float:
    """Compute S(x; x*, t) for one parameter using PARAM_SPECS."""
    spec = PARAM_SPECS[name]
    return bounded_score(
        value=params[name],
        optimum=spec["optimum"],
        tolerance=spec["tolerance"],
    )


def default_params() -> dict[str, float]:
    """Return simulator defaults for all defined parameters."""
    return {name: float(spec["default"]) for name, spec in PARAM_SPECS.items()}


def effective_params(params: dict[str, Any] | None = None) -> dict[str, float]:
    """Merge user-provided parameters onto simulator defaults."""
    merged = default_params()
    for key, value in (params or {}).items():
        if key not in PARAM_SPECS:
            raise KeyError(f"Unknown simulator parameter: {key}")
        merged[key] = float(value)
    return merged


def validate_bounds(params: dict[str, float]) -> None:
    """Raise an error if any parameter is outside its declared bounds."""
    for name, value in params.items():
        low, high = PARAM_SPECS[name]["bounds"]
        if not (low <= value <= high):
            raise ValueError(
                f"{name}={value} outside bounds [{low}, {high}]."
            )


def _variability_scale(name: str) -> float:
    """Return a parameter-specific scale for execution variability."""
    spec = PARAM_SPECS[name]
    if "tolerance" in spec:
        return float(spec["tolerance"])
    low, high = spec["bounds"]
    return float(high - low)


def apply_variability(
    planned_params: dict[str, float],
    *,
    seed: int | None = None,
    variability_strength: float = 0.05,
) -> dict[str, dict[str, float]]:
    """Simulate independent run-level execution noise.

    The function maps planned/target parameter values to actual executed
    values by adding independent zero-mean Gaussian noise to each parameter.
    Noise is scaled by each parameter's tolerance when available, otherwise by
    its allowed range. Values are clipped to the declared parameter bounds.

    Parameters marked ``optimizable=True`` in PARAM_SPECS are returned as:
        x_planned, x_actual, delta_x

    Parameters marked ``optimizable=False`` in PARAM_SPECS are returned as:
        z_target, z_actual, delta_z

    This function intentionally does not assign QC status, QC score, data
    eligibility flags, or observation noise. Those decisions belong to a
    separate SteMy QC/reproducibility module.
    """
    if variability_strength < 0:
        raise ValueError("variability_strength must be non-negative.")

    rng = np.random.default_rng(seed)

    planned_clean = {name: float(value) for name, value in planned_params.items()}
    actual_params: dict[str, float] = {}
    deviations: dict[str, float] = {}

    x_planned: dict[str, float] = {}
    x_actual: dict[str, float] = {}
    delta_x: dict[str, float] = {}

    z_target: dict[str, float] = {}
    z_actual: dict[str, float] = {}
    delta_z: dict[str, float] = {}

    for name, planned_value in planned_clean.items():
        if name not in PARAM_SPECS:
            raise KeyError(f"Unknown simulator parameter: {name}")

        spec = PARAM_SPECS[name]
        low, high = spec["bounds"]
        scale = _variability_scale(name)
        noise = rng.normal(0.0, variability_strength * scale)
        actual_value = float(np.clip(planned_value + noise, low, high))
        deviation = actual_value - planned_value

        actual_params[name] = actual_value
        deviations[name] = deviation

        if bool(spec.get("optimizable", False)):
            x_planned[name] = planned_value
            x_actual[name] = actual_value
            delta_x[name] = deviation
        else:
            z_target[name] = planned_value
            z_actual[name] = actual_value
            delta_z[name] = deviation

    return {
        "planned_params": planned_clean,
        "actual_params": actual_params,
        "parameter_deviations": deviations,
        "x_planned": x_planned,
        "x_actual": x_actual,
        "delta_x": delta_x,
        "z_target": z_target,
        "z_actual": z_actual,
        "delta_z": delta_z,
    }


def assumption_flags() -> dict[str, str]:
    """Return confidence labels for all simulator assumptions."""
    return {name: spec["confidence"] for name, spec in PARAM_SPECS.items()}


def sample_hidden_context(
    *,
    seed: int | None = None,
    context: dict[str, Any] | None = None,
) -> dict[str, float]:
    """Sample hidden variability; context overrides sampled values."""
    rng = np.random.default_rng(seed)

    hidden = {
        "cell_line_quality": float(rng.normal(0.0, 0.15)),
        "batch_effect": float(rng.normal(0.0, 0.10)),
        "temp_flux_min": float(np.clip(rng.gamma(shape=2.0, scale=1.0), 0.0, 20.0)),
        "o2_flux_min": float(np.clip(rng.gamma(shape=2.0, scale=1.0), 0.0, 20.0)),
    }

    if context:
        for key, value in context.items():
            hidden[key] = float(value)

    return hidden


def compute_seed_score(params: dict[str, float]) -> StageResult:
    """Stage 0-1: passaging/seeding quality and D0 readiness."""
    s_cpsc = score_param(params, "cpsc_pct")
    s_versene = score_param(params, "versene_dur_min")
    s_seed_density = score_param(params, "seed_density_mcells_per_well")

    s_rock = (
        0.5 * score_param(params, "rock_conc_uM")
        + 0.5 * score_param(params, "rock_dur_hr")
    )

    s_conf = score_param(params, "d0_confluence_pct")
    s_morph = clip01(params["d0_morphology_score"])

    s_cell_state = 0.6 * s_conf + 0.4 * s_morph

    q_seed = (
        0.15 * s_cpsc
        + 0.10 * s_versene
        + 0.20 * s_seed_density
        + 0.20 * s_rock
        + 0.35 * s_cell_state
    )

    components = {
        "S_cpsc": s_cpsc,
        "S_versene": s_versene,
        "S_seed_density": s_seed_density,
        "S_rock": s_rock,
        "S_d0_conf": s_conf,
        "S_d0_morph": s_morph,
        "S_cell_state": s_cell_state,
    }
    return StageResult(score=clip01(q_seed), components=components)


def compute_meso_score(params: dict[str, float], q_seed: float) -> StageResult:
    """Stage 2: CHIR/insulin-driven mesoderm induction."""
    s_chir_start = score_param(params, "chir_start_day")
    s_chir_conc = score_param(params, "chir_conc_uM")
    s_chir_dur = score_param(params, "chir_dur_hr")

    s_chir = (
        0.10 * s_chir_start
        + 0.45 * s_chir_conc
        + 0.45 * s_chir_dur
    )

    s_insulin_start = score_param(params, "insulin_start_day")
    s_insulin_conc = score_param(params, "insulin_conc")
    s_insulin_dur = score_param(params, "insulin_dur_hr")

    s_insulin = (
        0.25 * s_insulin_start
        + 0.40 * s_insulin_conc
        + 0.35 * s_insulin_dur
    )

    q_meso = (0.75 * s_chir + 0.25 * s_insulin) * (0.7 + 0.3 * q_seed)

    components = {
        "S_chir_start": s_chir_start,
        "S_chir_conc": s_chir_conc,
        "S_chir_dur": s_chir_dur,
        "S_chir": s_chir,
        "S_insulin_start": s_insulin_start,
        "S_insulin_conc": s_insulin_conc,
        "S_insulin_dur": s_insulin_dur,
        "S_insulin": s_insulin,
    }
    return StageResult(score=clip01(q_meso), components=components)


def compute_prog_score(params: dict[str, float], q_meso: float) -> StageResult:
    """Stage 3: bFGF/hypoxia progenitor-state modulation."""
    s_bfgf_start = score_param(params, "bfgf_start_day")
    s_bfgf_conc = score_param(params, "bfgf_conc_ng_ml")
    s_bfgf_dur = score_param(params, "bfgf_dur_hr")

    s_bfgf = (
        0.25 * s_bfgf_start
        + 0.45 * s_bfgf_conc
        + 0.30 * s_bfgf_dur
    )

    s_hypoxia_start = score_param(params, "hypoxia_start_day")
    s_hypoxia_dur = score_param(params, "hypoxia_dur_hr")
    s_hypoxia_pct = score_param(params, "hypoxia_pct")

    s_hypoxia = (
        0.30 * s_hypoxia_start
        + 0.30 * s_hypoxia_dur
        + 0.40 * s_hypoxia_pct
    )

    # Simple synergy: bFGF and hypoxia together can outperform either alone.
    s_prog_signal = (
        0.50 * s_bfgf
        + 0.30 * s_hypoxia
        + 0.20 * (s_bfgf * s_hypoxia)
    )

    q_prog = s_prog_signal * (0.7 + 0.3 * q_meso)

    components = {
        "S_bfgf_start": s_bfgf_start,
        "S_bfgf_conc": s_bfgf_conc,
        "S_bfgf_dur": s_bfgf_dur,
        "S_bfgf": s_bfgf,
        "S_hypoxia_start": s_hypoxia_start,
        "S_hypoxia_dur": s_hypoxia_dur,
        "S_hypoxia_pct": s_hypoxia_pct,
        "S_hypoxia": s_hypoxia,
        "S_prog_signal": s_prog_signal,
    }
    return StageResult(score=clip01(q_prog), components=components)


def compute_spec_score(params: dict[str, float], q_prog: float) -> StageResult:
    """Stage 4: IWP2-driven cardiac specification / Wnt inhibition."""
    s_iwp2_start = score_param(params, "iwp2_start_day")
    s_iwp2_conc = score_param(params, "iwp2_conc_uM")
    s_iwp2_dur = score_param(params, "iwp2_dur_hr")

    s_iwp2 = (
        0.30 * s_iwp2_start
        + 0.45 * s_iwp2_conc
        + 0.25 * s_iwp2_dur
    )

    q_spec = s_iwp2 * (0.65 + 0.35 * q_prog)

    components = {
        "S_iwp2_start": s_iwp2_start,
        "S_iwp2_conc": s_iwp2_conc,
        "S_iwp2_dur": s_iwp2_dur,
        "S_iwp2": s_iwp2,
    }
    return StageResult(score=clip01(q_spec), components=components)


def compute_environment_penalty(hidden: dict[str, float]) -> float:
    """Weak penalty for unintended handling/environmental stress."""
    return (
        -0.005 * float(hidden.get("temp_flux_min", 0.0))
        -0.003 * float(hidden.get("o2_flux_min", 0.0))
    )


def compute_final_ctnt(
    q_seed: float,
    q_meso: float,
    q_prog: float,
    q_spec: float,
    hidden: dict[str, float],
) -> dict[str, float]:
    """Combine stage scores and hidden context into bounded cTnT%."""
    q_final = (
        0.15 * q_seed
        + 0.25 * q_meso
        + 0.25 * q_prog
        + 0.35 * q_spec
    )

    e_env = compute_environment_penalty(hidden)

    z_ctnt = (
        2.0
        + 4.0 * (q_final - 0.85)
        + float(hidden.get("cell_line_quality", 0.0))
        + float(hidden.get("batch_effect", 0.0))
        + e_env
    )

    ctnt_pct = 100.0 * sigmoid(z_ctnt)

    return {
        "q_final": float(q_final),
        "z_ctnt": float(z_ctnt),
        "ctnt_pct": float(np.clip(ctnt_pct, 0.0, 100.0)),
        "environment_penalty": float(e_env),
    }


def simulate(
    params: dict[str, Any] | None = None,
    *,
    seed: int | None = None,
    context: dict[str, Any] | None = None,
    validate: bool = True,
    variability_strength: float = 0.05,
) -> dict[str, Any]:
    """Run one simulated iPSC-CM differentiation experiment."""
    planned_params = effective_params(params)

    if validate:
        validate_bounds(planned_params)

    variability = apply_variability(
        planned_params,
        seed=seed,
        variability_strength=variability_strength,
    )
    p = variability["actual_params"]

    if validate:
        validate_bounds(p)

    hidden = sample_hidden_context(seed=seed, context=context)

    seed_stage = compute_seed_score(p)
    meso_stage = compute_meso_score(p, seed_stage.score)
    prog_stage = compute_prog_score(p, meso_stage.score)
    spec_stage = compute_spec_score(p, prog_stage.score)

    final = compute_final_ctnt(
        q_seed=seed_stage.score,
        q_meso=meso_stage.score,
        q_prog=prog_stage.score,
        q_spec=spec_stage.score,
        hidden=hidden,
    )

    return {
        "ctnt_pct": final["ctnt_pct"],
        "q_final": final["q_final"],
        "z_ctnt": final["z_ctnt"],
        "stage_scores": {
            "q_seed": seed_stage.score,
            "q_meso": meso_stage.score,
            "q_prog": prog_stage.score,
            "q_spec": spec_stage.score,
        },
        "component_scores": {
            "seed": seed_stage.components,
            "meso": meso_stage.components,
            "prog": prog_stage.components,
            "spec": spec_stage.components,
        },
        "hidden_context": hidden,
        "environment_penalty": final["environment_penalty"],
        "planned_params": variability["planned_params"],
        "actual_params": variability["actual_params"],
        "parameter_deviations": variability["parameter_deviations"],
        "x_planned": variability["x_planned"],
        "x_actual": variability["x_actual"],
        "delta_x": variability["delta_x"],
        "z_target": variability["z_target"],
        "z_actual": variability["z_actual"],
        "delta_z": variability["delta_z"],
        "variability_strength": float(variability_strength),
        "effective_params": p,  # Backward-compatible alias for actual_params.
        "assumption_flags": assumption_flags(),
    }


def evaluate_for_ax(
    params: dict[str, Any],
    *,
    seed: int | None = None,
    context: dict[str, Any] | None = None,
    variability_strength: float = 0.05,
) -> dict[str, float]:
    """Minimal Ax-compatible wrapper."""
    out = simulate(
        params=params,
        seed=seed,
        context=context,
        variability_strength=variability_strength,
    )
    return {"ctnt_pct": out["ctnt_pct"]}


if __name__ == "__main__":
    result = simulate(seed=123)
    print("Simulated cTnT%:", round(result["ctnt_pct"], 2))
    print("Q_final:", round(result["q_final"], 3))
    print("Stage scores:", result["stage_scores"])