"""
SteMy Ax Bayesian optimization CLI, v1.1.0.

This script implements a persistent, human-in-the-loop optimizer workflow for
SteMy's iPSC-cardiomyocyte differentiation simulator and future wet-lab backend.
It separates proposed experimental configurations from actual executed runs:

    config TSV: proposed setups and edit history
    log TSV:    actual runs, outcomes, and Ax sync state

Core commands:
    create-study   Create a persistent Ax study.
    recommend      Generate a new auto-incremented batch of proposed configs.
    edit-config    Duplicate/edit one config while preserving history.
    run            Convert selected configs into actual run-log rows.
    simulate       Run an automated simulator loop: recommend -> run -> log -> sync.
    sync-results   Fallback command to sync completed run results into Ax.
    summary        Print and save a study-state summary.

Example:
    python bo_ax.py create-study --study-id test1 --seed 123
    python bo_ax.py recommend --study-id test1 --n-trials 6
    python bo_ax.py edit-config --study-id test1 --batch-id B01 --config-id C03 \
        --set chir_conc_uM=11.5 --notes "more conservative CHIR"
    python bo_ax.py run --study-id test1 --batch-id B01 --plate-id P1
    python bo_ax.py sync-results --study-id test1
    python bo_ax.py summary --study-id test1

Simulator dry run:
    python bo_ax_v1_1_0.py simulate --study-id sim1 --n-trials 30 --batch-size 6 --seed 123

Output files:
    results/<study_id>/<study_id>_ax_client_snapshot.json
    results/<study_id>/<study_id>_metadata.json
    results/<study_id>/<study_id>_summary.json
    results/<study_id>/<study_id>_config.tsv
    results/<study_id>/<study_id>_log.tsv

Objective:
    Maximize ctnt_pct.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

from differentiation_sim import PARAM_SPECS, simulate

try:
    from ax.api.client import Client
    from ax.api.configs import RangeParameterConfig
except ImportError as exc:
    raise ImportError("Install Ax with: pip install -U ax-platform") from exc


VERSION = "1.1.0"
OBJECTIVE_NAME = "ctnt_pct"
SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR.parent / "results"

CONFIG_BASE_COLUMNS = [
    "study_id",
    "batch_id",
    "config_id",
    "status",
    "trial_index",
    "created_at",
    "started_at",
    "finished_at",
    "created_by",
    "notes",
]

LOG_BASE_COLUMNS = [
    "study_id",
    "batch_id",
    "run_id",
    "config_id",
    "trial_index",
    "run_status",
    "ctnt_pct",
    "synced",
    "started_at",
    "finished_at",
    "notes",
]

DEFAULT_OPTIMIZER_PARAMS = [
    name for name, spec in PARAM_SPECS.items() if bool(spec.get("optimizable", False))
]


# -----------------------------------------------------------------------------
# Generic helpers
# -----------------------------------------------------------------------------


def now_iso() -> str:
    return datetime.now().replace(microsecond=0).isoformat()


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def is_blank(value: Any) -> bool:
    return value is None or str(value).strip() == ""


def safe_float(value: Any) -> float:
    if is_blank(value):
        return math.nan
    return float(value)


def parse_param_names(text: str | None) -> list[str]:
    """Parse comma-separated optimizer parameter names, or use defaults."""
    if not text:
        return list(DEFAULT_OPTIMIZER_PARAMS)
    names = [part.strip() for part in text.split(",") if part.strip()]
    unknown = [name for name in names if name not in PARAM_SPECS]
    if unknown:
        raise KeyError(f"Unknown parameter(s): {unknown}")
    return names


def parse_csv_arg(text: str | None) -> list[str]:
    """Parse a comma-separated CLI value into a list."""
    if not text:
        return []
    return [part.strip() for part in text.split(",") if part.strip()]


def parse_set_args(items: list[str]) -> dict[str, float]:
    """Parse repeated --set name=value arguments."""
    updates: dict[str, float] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Invalid --set value {item!r}; expected name=value")
        name, value = item.split("=", 1)
        name = name.strip()
        if name not in PARAM_SPECS:
            raise KeyError(f"Unknown parameter in --set: {name}")
        updates[name] = float(value)
    if not updates:
        raise ValueError("At least one --set name=value update is required")
    return updates


def format_batch_id(num: int) -> str:
    return f"B{num:02d}"


def format_config_id(num: int) -> str:
    return f"C{num:02d}"


def config_base(config_id: str) -> str:
    return config_id.split("-", 1)[0]


def config_version(config_id: str) -> int:
    if "-" not in config_id:
        return 0
    try:
        return int(config_id.split("-", 1)[1])
    except ValueError:
        return 0


def bool_str(value: bool) -> str:
    return "true" if value else "false"


# -----------------------------------------------------------------------------
# Paths and persistence
# -----------------------------------------------------------------------------


def study_dir(study_id: str) -> Path:
    return RESULTS_DIR / study_id


def study_paths(study_id: str) -> dict[str, Path]:
    root = study_dir(study_id)
    return {
        "root": root,
        "ax": root / f"{study_id}_ax_client_snapshot.json",
        "metadata": root / f"{study_id}_metadata.json",
        "summary": root / f"{study_id}_summary.json",
        "config": root / f"{study_id}_config.tsv",
        "log": root / f"{study_id}_log.tsv",
    }


def save_json(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_tsv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        return [dict(row) for row in reader]


def write_tsv(rows: list[dict[str, Any]], path: Path, fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def config_fieldnames(param_names: list[str]) -> list[str]:
    return CONFIG_BASE_COLUMNS + param_names


def log_fieldnames(param_names: list[str]) -> list[str]:
    return LOG_BASE_COLUMNS + [f"{name}_actual" for name in param_names]


def save_config_rows(study_id: str, rows: list[dict[str, Any]], param_names: list[str]) -> None:
    write_tsv(rows, study_paths(study_id)["config"], config_fieldnames(param_names))


def save_log_rows(study_id: str, rows: list[dict[str, Any]], param_names: list[str], path: Path | None = None) -> None:
    write_tsv(rows, path or study_paths(study_id)["log"], log_fieldnames(param_names))


def load_metadata(study_id: str) -> dict[str, Any]:
    path = study_paths(study_id)["metadata"]
    if not path.exists():
        raise FileNotFoundError(f"Study {study_id!r} does not exist. Run create-study first.")
    return load_json(path)


def save_metadata(study_id: str, metadata: dict[str, Any]) -> None:
    metadata["updated_at"] = now_iso()
    save_json(metadata, study_paths(study_id)["metadata"])


# -----------------------------------------------------------------------------
# Ax helpers
# -----------------------------------------------------------------------------


def build_search_space(param_names: list[str]) -> list[RangeParameterConfig]:
    """Convert PARAM_SPECS entries into Ax range parameters."""
    search_space: list[RangeParameterConfig] = []
    for name in param_names:
        spec = PARAM_SPECS[name]
        low, high = spec["bounds"]
        search_space.append(
            RangeParameterConfig(
                name=name,
                parameter_type="float",
                bounds=(float(low), float(high)),
            )
        )
    return search_space


def make_client(param_names: list[str], *, seed: int, study_id: str) -> Client:
    """Create an Ax Client experiment that maximizes cTnT%."""
    client = Client()
    client.configure_experiment(
        name=study_id,
        parameters=build_search_space(param_names),
    )
    client.configure_optimization(objective=OBJECTIVE_NAME)

    if hasattr(client, "configure_generation_strategy"):
        try:
            client.configure_generation_strategy(
                method="fast",
                initialization_random_seed=seed,
                use_existing_trials_for_initialization=True,
            )
        except TypeError:
            client.configure_generation_strategy(
                method="fast",
                initialization_random_seed=seed,
            )
        except Exception as exc:
            print(f"Warning: could not configure Ax generation seed: {exc}")

    return client


def save_client(study_id: str, client: Client) -> None:
    path = study_paths(study_id)["ax"]
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        client.save_to_json_file(filepath=str(path))
    except TypeError:
        client.save_to_json_file(str(path))


def load_client(study_id: str) -> Client:
    path = study_paths(study_id)["ax"]
    if not path.exists():
        raise FileNotFoundError(f"Missing Ax snapshot: {path}")
    try:
        return Client.load_from_json_file(filepath=str(path))
    except TypeError:
        return Client.load_from_json_file(str(path))


def get_next_trials(client: Client, *, n_trials: int) -> dict[int, dict[str, Any]]:
    """Request candidate parameterizations from Ax."""
    result = client.get_next_trials(max_trials=n_trials)
    trial_map = result[0] if isinstance(result, tuple) else result
    return {int(k): dict(v) for k, v in trial_map.items()}


def attach_trial(client: Client, params: dict[str, Any]) -> int:
    """Attach a user-defined config as a new Ax trial."""
    result = client.attach_trial(parameters={k: float(v) for k, v in params.items()})
    return int(result[0] if isinstance(result, tuple) else result)


def mark_trial_failed_if_possible(client: Client, trial_index: int) -> None:
    """Best-effort handling for Ax trials that were rejected/edited before running."""
    for method_name in ("mark_trial_failed", "log_trial_failure"):
        method = getattr(client, method_name, None)
        if method is None:
            continue
        try:
            method(trial_index=trial_index)
            return
        except TypeError:
            try:
                method(trial_index)
                return
            except Exception:
                return
        except Exception:
            return


def complete_trial(client: Client, trial_index: int, ctnt_pct: float) -> None:
    """Report one completed trial result back to Ax."""
    client.complete_trial(
        trial_index=trial_index,
        raw_data={OBJECTIVE_NAME: float(ctnt_pct)},
    )


def get_best_recommendation(client: Client) -> dict[str, Any]:
    """Return Ax's current best parameterization, if available."""
    try:
        best = client.get_best_parameterization()
    except Exception as exc:
        return {"error": str(exc)}

    if isinstance(best, tuple):
        params = best[0] if len(best) > 0 else None
        prediction = best[1] if len(best) > 1 else None
    else:
        params = best
        prediction = None

    return {"params": params, "prediction": prediction}


# -----------------------------------------------------------------------------
# Study creation
# -----------------------------------------------------------------------------


def create_study(*, study_id: str, params: str | None, seed: int, overwrite: bool) -> None:
    paths = study_paths(study_id)
    root = paths["root"]

    if root.exists():
        if not overwrite:
            raise FileExistsError(f"Study {study_id!r} already exists. Use --overwrite to recreate it.")
        shutil.rmtree(root)

    root.mkdir(parents=True, exist_ok=False)
    param_names = parse_param_names(params)
    if not param_names:
        raise ValueError("At least one optimizer parameter is required.")

    client = make_client(param_names, seed=seed, study_id=study_id)
    save_client(study_id, client)

    metadata = {
        "study_id": study_id,
        "version": VERSION,
        "objective_name": OBJECTIVE_NAME,
        "created_at": now_iso(),
        "updated_at": now_iso(),
        "seed": seed,
        "param_names": param_names,
        "next_batch_num": 1,
        "paths": {k: str(v) for k, v in paths.items() if k != "root"},
    }
    save_metadata(study_id, metadata)
    save_config_rows(study_id, [], param_names)
    save_log_rows(study_id, [], param_names)
    save_summary(study_id)

    print(f"Created study: {study_id}")
    print(f"Results directory: {root}")
    print("Optimizer parameters:")
    for name in param_names:
        spec = PARAM_SPECS[name]
        print(f"  {name}: bounds={spec['bounds']}, default={spec.get('default')}")


# -----------------------------------------------------------------------------
# Config table operations
# -----------------------------------------------------------------------------


def next_batch_id(metadata: dict[str, Any]) -> str:
    return format_batch_id(int(metadata.get("next_batch_num", 1)))


def increment_batch(metadata: dict[str, Any]) -> None:
    metadata["next_batch_num"] = int(metadata.get("next_batch_num", 1)) + 1


def configs_for_batch(config_rows: list[dict[str, str]], batch_id: str) -> list[dict[str, str]]:
    return [row for row in config_rows if row.get("batch_id") == batch_id]


def latest_config_rows(config_rows: list[dict[str, str]], batch_id: str) -> list[dict[str, str]]:
    """Return the latest version of each config base in a batch."""
    latest: dict[str, dict[str, str]] = {}
    for row in configs_for_batch(config_rows, batch_id):
        cid = row.get("config_id", "")
        base = config_base(cid)
        if base not in latest or config_version(cid) > config_version(latest[base].get("config_id", "")):
            latest[base] = row
    return [latest[key] for key in sorted(latest, key=lambda x: int(x[1:]) if x.startswith("C") and x[1:].isdigit() else x)]


def find_config(config_rows: list[dict[str, str]], *, batch_id: str, config_id: str) -> dict[str, str]:
    matches = [row for row in config_rows if row.get("batch_id") == batch_id and row.get("config_id") == config_id]
    if not matches:
        raise KeyError(f"No config found for batch_id={batch_id}, config_id={config_id}")
    return matches[-1]


def config_params(row: dict[str, str], param_names: list[str]) -> dict[str, float]:
    return {name: float(row[name]) for name in param_names}


def next_config_id_for_batch(config_rows: list[dict[str, str]], batch_id: str) -> str:
    nums: list[int] = []
    for row in configs_for_batch(config_rows, batch_id):
        base = config_base(row.get("config_id", ""))
        if base.startswith("C") and base[1:].isdigit():
            nums.append(int(base[1:]))
    return format_config_id((max(nums) if nums else 0) + 1)


def next_config_version(config_rows: list[dict[str, str]], batch_id: str, base_config_id: str) -> str:
    base = config_base(base_config_id)
    versions = [
        config_version(row.get("config_id", ""))
        for row in configs_for_batch(config_rows, batch_id)
        if config_base(row.get("config_id", "")) == base
    ]
    return f"{base}-{(max(versions) if versions else 0) + 1}"


def print_configs(rows: list[dict[str, str]], param_names: list[str], *, title: str | None = None) -> None:
    if title:
        print(f"\n{title}")
    if not rows:
        print("No configs.")
        return

    cols = ["batch_id", "config_id", "status", "trial_index", "created_by", "notes", *param_names]
    widths = {col: max(len(col), *(len(str(row.get(col, ""))) for row in rows)) for col in cols}
    header = "  ".join(col.ljust(widths[col]) for col in cols)
    print(header)
    print("  ".join("-" * widths[col] for col in cols))
    for row in rows:
        print("  ".join(str(row.get(col, "")).ljust(widths[col]) for col in cols))


def recommend_configs(
    *,
    study_id: str,
    n_trials: int,
    policy: str = "balanced",
    interactive: bool = True,
) -> str:
    """Generate a new batch of proposed configs and optionally prompt user action."""
    if n_trials < 1:
        raise ValueError("--n-trials must be >= 1")

    metadata = load_metadata(study_id)
    param_names = list(metadata["param_names"])
    batch_id = next_batch_id(metadata)
    client = load_client(study_id)
    config_rows = read_tsv(study_paths(study_id)["config"])

    if policy != "balanced":
        print(f"Policy {policy!r} is accepted as a placeholder, but v{VERSION} uses Ax default generation.")

    candidates = get_next_trials(client, n_trials=n_trials)
    created_at = now_iso()

    for i, (trial_index, params) in enumerate(candidates.items(), start=1):
        config_id = format_config_id(i)
        row: dict[str, Any] = {
            "study_id": study_id,
            "batch_id": batch_id,
            "config_id": config_id,
            "status": "proposed",
            "trial_index": trial_index,
            "created_at": created_at,
            "started_at": "",
            "finished_at": "",
            "created_by": "model",
            "notes": "",
        }
        for name in param_names:
            row[name] = params[name]
        config_rows.append(row)

    increment_batch(metadata)
    save_config_rows(study_id, config_rows, param_names)
    save_client(study_id, client)
    save_metadata(study_id, metadata)
    save_summary(study_id)

    print_configs(latest_config_rows(config_rows, batch_id), param_names, title=f"Recommended configs for {study_id} / {batch_id}")

    if interactive:
        recommend_interactive_loop(study_id=study_id, batch_id=batch_id)

    return batch_id


def edit_config(
    *,
    study_id: str,
    batch_id: str,
    config_id: str,
    updates: dict[str, float],
    notes: str = "",
    print_after: bool = True,
) -> str:
    """Create an edited config version and preserve the original row as rejected."""
    metadata = load_metadata(study_id)
    param_names = list(metadata["param_names"])
    unknown_updates = [name for name in updates if name not in param_names]
    if unknown_updates:
        raise KeyError(f"Cannot edit non-optimized parameter(s) in this study: {unknown_updates}")

    client = load_client(study_id)
    config_rows = read_tsv(study_paths(study_id)["config"])
    original = find_config(config_rows, batch_id=batch_id, config_id=config_id)

    original_status = original.get("status", "")
    if original_status in {"running", "completed"}:
        raise ValueError(f"Cannot edit config {config_id} with status={original_status!r}")

    for row in config_rows:
        if row is original:
            row["status"] = "rejected"
            row["finished_at"] = now_iso()
            if notes:
                row["notes"] = f"replaced by edited version; {row.get('notes', '')}".strip("; ")
            break

    old_trial = original.get("trial_index")
    if old_trial and old_trial.isdigit():
        mark_trial_failed_if_possible(client, int(old_trial))

    new_config_id = next_config_version(config_rows, batch_id, config_id)
    new_params = config_params(original, param_names)
    new_params.update(updates)
    new_trial_index = attach_trial(client, new_params)

    new_row: dict[str, Any] = {
        "study_id": study_id,
        "batch_id": batch_id,
        "config_id": new_config_id,
        "status": "proposed",
        "trial_index": new_trial_index,
        "created_at": now_iso(),
        "started_at": "",
        "finished_at": "",
        "created_by": "user",
        "notes": notes,
    }
    for name in param_names:
        new_row[name] = new_params[name]
    config_rows.append(new_row)

    save_config_rows(study_id, config_rows, param_names)
    save_client(study_id, client)
    save_metadata(study_id, metadata)
    save_summary(study_id)

    if print_after:
        print_configs(latest_config_rows(config_rows, batch_id), param_names, title=f"Latest configs for {study_id} / {batch_id}")

    return new_config_id


def recommend_interactive_loop(*, study_id: str, batch_id: str) -> None:
    """Prompt user to confirm or edit configs after recommend."""
    metadata = load_metadata(study_id)
    param_names = list(metadata["param_names"])

    while True:
        action = input("\nAction: confirm or edit? [confirm/edit]: ").strip().lower()
        if action in {"", "confirm", "c"}:
            next_action = input("Confirm configs. Run now or save and quit? [run-now/save]: ").strip().lower()
            if next_action in {"run-now", "run", "r"}:
                run_configs(study_id=study_id, batch_id=batch_id, config_ids=None, plate_id="P1", run_ids=None)
            else:
                print("Saved proposed configs. Run later with:")
                print(f"  python bo_ax_v1_1_0.py run --study-id {study_id} --batch-id {batch_id}")
            return

        if action in {"edit", "e"}:
            config_id = input("Config ID to edit, e.g. C03: ").strip()
            raw_updates = input("Updates, comma-separated name=value pairs: ").strip()
            notes = input("Notes (optional): ").strip()
            updates = parse_set_args([part.strip() for part in raw_updates.split(",") if part.strip()])
            edit_config(
                study_id=study_id,
                batch_id=batch_id,
                config_id=config_id,
                updates=updates,
                notes=notes,
                print_after=True,
            )
            continue

        print("Invalid action. Enter confirm or edit.")
        config_rows = read_tsv(study_paths(study_id)["config"])
        print_configs(latest_config_rows(config_rows, batch_id), param_names, title=f"Latest configs for {study_id} / {batch_id}")


# -----------------------------------------------------------------------------
# Run/log operations
# -----------------------------------------------------------------------------


def resolve_configs_to_run(
    *,
    config_rows: list[dict[str, str]],
    batch_id: str,
    config_ids: list[str] | None,
) -> list[dict[str, str]]:
    if config_ids:
        return [find_config(config_rows, batch_id=batch_id, config_id=cid) for cid in config_ids]

    latest = latest_config_rows(config_rows, batch_id)
    return [row for row in latest if row.get("status") == "proposed"]


def auto_run_ids(n: int, plate_id: str) -> list[str]:
    return [f"{plate_id}-W{i}" for i in range(1, n + 1)]


def run_configs(
    *,
    study_id: str,
    batch_id: str,
    config_ids: list[str] | None,
    plate_id: str,
    run_ids: list[str] | None,
) -> list[str]:
    """Convert selected proposed configs into actual run-log rows."""
    metadata = load_metadata(study_id)
    param_names = list(metadata["param_names"])
    paths = study_paths(study_id)
    config_rows = read_tsv(paths["config"])
    log_rows = read_tsv(paths["log"])

    selected = resolve_configs_to_run(
        config_rows=config_rows,
        batch_id=batch_id,
        config_ids=config_ids,
    )
    if not selected:
        raise ValueError(f"No proposed configs found to run for batch {batch_id}.")

    if run_ids is None:
        run_ids = auto_run_ids(len(selected), plate_id)
    if len(run_ids) != len(selected):
        raise ValueError("Number of --run-ids must match number of selected configs.")

    existing_keys = {(row.get("batch_id"), row.get("run_id")) for row in log_rows}
    started_at = now_iso()
    created_run_ids: list[str] = []

    selected_ids = {row["config_id"] for row in selected}
    for row in config_rows:
        if row.get("batch_id") == batch_id and row.get("config_id") in selected_ids:
            if row.get("status") not in {"proposed", "running"}:
                raise ValueError(
                    f"Config {row.get('config_id')} has status={row.get('status')!r}; cannot run."
                )
            row["status"] = "running"
            row["started_at"] = row.get("started_at") or started_at

    for cfg, run_id in zip(selected, run_ids):
        key = (batch_id, run_id)
        if key in existing_keys:
            raise ValueError(f"Run already exists for batch_id={batch_id}, run_id={run_id}")

        run_row: dict[str, Any] = {
            "study_id": study_id,
            "batch_id": batch_id,
            "run_id": run_id,
            "config_id": cfg["config_id"],
            "trial_index": cfg["trial_index"],
            "run_status": "running",
            "ctnt_pct": "",
            "synced": "false",
            "started_at": started_at,
            "finished_at": "",
            "notes": "",
        }
        for name in param_names:
            run_row[f"{name}_actual"] = cfg[name]
        log_rows.append(run_row)
        created_run_ids.append(run_id)

    save_config_rows(study_id, config_rows, param_names)
    save_log_rows(study_id, log_rows, param_names)
    save_metadata(study_id, metadata)
    save_summary(study_id)

    print(f"Created {len(created_run_ids)} run(s) for {study_id} / {batch_id}:")
    for run_id in created_run_ids:
        print(f"  {run_id}")
    return created_run_ids


def params_from_log_row(row: dict[str, str], param_names: list[str]) -> dict[str, float]:
    return {name: float(row[f"{name}_actual"]) for name in param_names}


def simulate_completed_runs(*, study_id: str, batch_id: str, seed: int) -> None:
    """Evaluate running rows in the simulator and write outcomes to the log."""
    metadata = load_metadata(study_id)
    param_names = list(metadata["param_names"])
    paths = study_paths(study_id)
    log_rows = read_tsv(paths["log"])
    config_rows = read_tsv(paths["config"])
    finished_at = now_iso()

    simulated = 0
    for row in log_rows:
        if row.get("batch_id") != batch_id or row.get("run_status") != "running":
            continue
        trial_index = int(row["trial_index"])
        params = params_from_log_row(row, param_names)
        out = simulate(params=params, seed=seed + trial_index, validate=True)
        row["ctnt_pct"] = out["ctnt_pct"]
        row["run_status"] = "completed"
        row["finished_at"] = finished_at
        simulated += 1

        for cfg in config_rows:
            if cfg.get("batch_id") == batch_id and cfg.get("config_id") == row.get("config_id"):
                cfg["status"] = "completed"
                cfg["finished_at"] = finished_at
                break

    save_log_rows(study_id, log_rows, param_names)
    save_config_rows(study_id, config_rows, param_names)
    save_summary(study_id)
    print(f"Simulated {simulated} completed run(s) for {study_id} / {batch_id}.")


# -----------------------------------------------------------------------------
# Sync and summary
# -----------------------------------------------------------------------------


def sync_results(*, study_id: str, results_file: str | None = None) -> int:
    """Sync completed, unsynced run outcomes into Ax."""
    metadata = load_metadata(study_id)
    param_names = list(metadata["param_names"])
    paths = study_paths(study_id)
    log_path = Path(results_file) if results_file else paths["log"]

    client = load_client(study_id)
    log_rows = read_tsv(log_path)
    config_rows = read_tsv(paths["config"])
    synced_count = 0

    for row in log_rows:
        if row.get("run_status") != "completed":
            continue
        if parse_bool(row.get("synced", "false")):
            continue
        ctnt = safe_float(row.get("ctnt_pct"))
        if math.isnan(ctnt):
            continue
        trial_index = int(row["trial_index"])
        complete_trial(client, trial_index, ctnt)
        row["synced"] = "true"
        synced_count += 1

        for cfg in config_rows:
            if cfg.get("batch_id") == row.get("batch_id") and cfg.get("config_id") == row.get("config_id"):
                cfg["status"] = "completed"
                cfg["finished_at"] = cfg.get("finished_at") or row.get("finished_at", "")
                break

    save_client(study_id, client)
    save_log_rows(study_id, log_rows, param_names, path=log_path)
    if log_path != paths["log"]:
        print(f"Synced from alternate log path: {log_path}")
    save_config_rows(study_id, config_rows, param_names)
    save_metadata(study_id, metadata)
    save_summary(study_id)

    print(f"Synced {synced_count} completed run(s) into Ax.")
    return synced_count


def compute_summary(study_id: str) -> dict[str, Any]:
    metadata = load_metadata(study_id)
    paths = study_paths(study_id)
    config_rows = read_tsv(paths["config"])
    log_rows = read_tsv(paths["log"])

    batches = sorted({row.get("batch_id", "") for row in config_rows if row.get("batch_id")})
    latest_batch = batches[-1] if batches else None

    completed_rows = [
        row for row in log_rows
        if row.get("run_status") == "completed" and not math.isnan(safe_float(row.get("ctnt_pct")))
    ]
    best_row = max(completed_rows, key=lambda row: safe_float(row.get("ctnt_pct")), default=None)

    counts = {
        "n_batches": len(batches),
        "n_configs": len(config_rows),
        "n_runs": len(log_rows),
        "n_completed_runs": sum(1 for row in log_rows if row.get("run_status") == "completed"),
        "n_synced_runs": sum(1 for row in log_rows if parse_bool(row.get("synced", "false"))),
        "n_running_runs": sum(1 for row in log_rows if row.get("run_status") == "running"),
        "n_failed_runs": sum(1 for row in log_rows if row.get("run_status") == "failed"),
        "n_unsynced_completed_runs": sum(
            1 for row in log_rows
            if row.get("run_status") == "completed" and not parse_bool(row.get("synced", "false"))
        ),
    }

    batch_summaries = []
    for batch_id in batches:
        b_configs = [row for row in config_rows if row.get("batch_id") == batch_id]
        b_logs = [row for row in log_rows if row.get("batch_id") == batch_id]
        batch_summaries.append(
            {
                "batch_id": batch_id,
                "n_configs": len(b_configs),
                "n_proposed": sum(1 for row in b_configs if row.get("status") == "proposed"),
                "n_rejected": sum(1 for row in b_configs if row.get("status") == "rejected"),
                "n_running": sum(1 for row in b_logs if row.get("run_status") == "running"),
                "n_completed": sum(1 for row in b_logs if row.get("run_status") == "completed"),
                "n_synced": sum(1 for row in b_logs if parse_bool(row.get("synced", "false"))),
            }
        )

    warnings: list[str] = []
    if counts["n_unsynced_completed_runs"]:
        warnings.append(f"{counts['n_unsynced_completed_runs']} completed run(s) have synced=false")
    if counts["n_running_runs"]:
        warnings.append(f"{counts['n_running_runs']} run(s) are still marked running")

    if counts["n_synced_runs"] > 0:
        try:
            current_recommendation = get_best_recommendation(load_client(study_id))
        except Exception as exc:
            current_recommendation = {"error": str(exc)}
    else:
        current_recommendation = {
            "params": None,
            "prediction": None,
            "note": "No synced results yet; Ax recommendation unavailable.",
        }

    summary = {
        "study_id": study_id,
        "version": VERSION,
        "objective_name": OBJECTIVE_NAME,
        "updated_at": now_iso(),
        "counts": counts,
        "batches": batch_summaries,
        "latest_batch": latest_batch,
        "best_observed": None,
        "current_recommendation": current_recommendation,
        "warnings": warnings,
        "paths": metadata.get("paths", {}),
    }

    if best_row:
        param_names = list(metadata["param_names"])
        summary["best_observed"] = {
            "batch_id": best_row.get("batch_id"),
            "run_id": best_row.get("run_id"),
            "config_id": best_row.get("config_id"),
            "trial_index": best_row.get("trial_index"),
            "ctnt_pct": safe_float(best_row.get("ctnt_pct")),
            "params": {f"{name}_actual": safe_float(best_row.get(f"{name}_actual")) for name in param_names},
        }

    return summary


def save_summary(study_id: str) -> dict[str, Any]:
    summary = compute_summary(study_id)
    save_json(summary, study_paths(study_id)["summary"])
    return summary


def print_summary(study_id: str, *, as_json: bool = False) -> None:
    summary = save_summary(study_id)
    if as_json:
        print(json.dumps(summary, indent=2, default=str))
        return

    print(f"Study: {study_id}")
    print(f"Objective: maximize {OBJECTIVE_NAME}")
    print("\nCounts:")
    for key, value in summary["counts"].items():
        print(f"  {key}: {value}")

    print("\nBatches:")
    for batch in summary["batches"]:
        print(
            f"  {batch['batch_id']}: "
            f"configs={batch['n_configs']}, proposed={batch['n_proposed']}, rejected={batch['n_rejected']}, "
            f"running={batch['n_running']}, completed={batch['n_completed']}, synced={batch['n_synced']}"
        )

    print("\nBest observed:")
    if summary["best_observed"]:
        best = summary["best_observed"]
        print(f"  batch_id:    {best['batch_id']}")
        print(f"  run_id:      {best['run_id']}")
        print(f"  config_id:   {best['config_id']}")
        print(f"  trial_index: {best['trial_index']}")
        print(f"  ctnt_pct:    {best['ctnt_pct']:.2f}")
    else:
        print("  none yet")

    print("\nWarnings:")
    if summary["warnings"]:
        for warning in summary["warnings"]:
            print(f"  {warning}")
    else:
        print("  none")


# -----------------------------------------------------------------------------
# Simulation loop
# -----------------------------------------------------------------------------


def ensure_study_for_simulation(study_id: str, seed: int) -> None:
    if not study_dir(study_id).exists():
        create_study(study_id=study_id, params=None, seed=seed, overwrite=False)


def simulate_loop(
    *,
    study_id: str,
    n_trials: int | None,
    n_batch: int | None,
    batch_size: int,
    rec_mode: str,
    seed: int,
) -> None:
    if batch_size < 1:
        raise ValueError("--batch-size must be >= 1")
    if n_trials is None and n_batch is None:
        raise ValueError("Provide either --n-trials or --n-batch")
    if n_trials is None:
        n_trials = int(n_batch) * batch_size
    if n_batch is not None and n_trials != int(n_batch) * batch_size:
        raise ValueError("When both --n-trials and --n-batch are provided, require n_trials == n_batch * batch_size")
    if rec_mode != "auto-accept":
        raise ValueError("v1.1.0 only implements --rec-mode auto-accept")

    ensure_study_for_simulation(study_id, seed)

    remaining = n_trials
    while remaining > 0:
        n = min(batch_size, remaining)
        batch_id = recommend_configs(study_id=study_id, n_trials=n, policy="balanced", interactive=False)
        run_configs(study_id=study_id, batch_id=batch_id, config_ids=None, plate_id="P1", run_ids=None)
        simulate_completed_runs(study_id=study_id, batch_id=batch_id, seed=seed)
        sync_results(study_id=study_id)
        remaining -= n

    print("\nSimulation complete.")
    print_summary(study_id)


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=f"SteMy Ax optimizer CLI v{VERSION}")
    sub = parser.add_subparsers(dest="command", required=True)

    p_create = sub.add_parser("create-study", help="Create a persistent optimizer study.")
    p_create.add_argument("--study-id", required=True)
    p_create.add_argument("--params", default=None, help="Comma-separated optimizer parameters.")
    p_create.add_argument("--seed", type=int, default=123)
    p_create.add_argument("--overwrite", action="store_true")

    p_recommend = sub.add_parser("recommend", help="Generate proposed configs for a new batch.")
    p_recommend.add_argument("--study-id", required=True)
    p_recommend.add_argument("--n-trials", type=int, required=True)
    p_recommend.add_argument("--policy", default="balanced", help="Placeholder; v1.1.0 uses Ax default generation.")

    p_edit = sub.add_parser("edit-config", help="Create an edited config version.")
    p_edit.add_argument("--study-id", required=True)
    p_edit.add_argument("--batch-id", required=True)
    p_edit.add_argument("--config-id", required=True)
    p_edit.add_argument("--set", dest="sets", action="append", required=True, help="Parameter update name=value. Can be repeated.")
    p_edit.add_argument("--notes", default="")

    p_run = sub.add_parser("run", help="Create actual run-log rows from selected configs.")
    p_run.add_argument("--study-id", required=True)
    p_run.add_argument("--batch-id", required=True)
    p_run.add_argument("--config-ids", default=None, help="Comma-separated config IDs. Defaults to all latest proposed configs.")
    p_run.add_argument("--plate-id", default="P1")
    p_run.add_argument("--run-ids", default=None, help="Comma-separated explicit run IDs.")

    p_sim = sub.add_parser("simulate", help="Run automated simulator loop.")
    p_sim.add_argument("--study-id", required=True)
    p_sim.add_argument("--n-trials", type=int, default=None)
    p_sim.add_argument("--n-batch", type=int, default=None)
    p_sim.add_argument("--batch-size", type=int, default=6)
    p_sim.add_argument("--rec-mode", default="auto-accept")
    p_sim.add_argument("--seed", type=int, default=123)

    p_sync = sub.add_parser("sync-results", help="Sync completed results into Ax.")
    p_sync.add_argument("--study-id", required=True)
    p_sync.add_argument("--results-file", default=None)

    p_summary = sub.add_parser("summary", help="Print study summary.")
    p_summary.add_argument("--study-id", required=True)
    p_summary.add_argument("--json", action="store_true")

    return parser


def main() -> None:
    args = build_parser().parse_args()

    if args.command == "create-study":
        create_study(
            study_id=args.study_id,
            params=args.params,
            seed=args.seed,
            overwrite=args.overwrite,
        )
    elif args.command == "recommend":
        recommend_configs(
            study_id=args.study_id,
            n_trials=args.n_trials,
            policy=args.policy,
            interactive=True,
        )
    elif args.command == "edit-config":
        edit_config(
            study_id=args.study_id,
            batch_id=args.batch_id,
            config_id=args.config_id,
            updates=parse_set_args(args.sets),
            notes=args.notes,
            print_after=True,
        )
    elif args.command == "run":
        run_configs(
            study_id=args.study_id,
            batch_id=args.batch_id,
            config_ids=parse_csv_arg(args.config_ids) or None,
            plate_id=args.plate_id,
            run_ids=parse_csv_arg(args.run_ids) or None,
        )
    elif args.command == "simulate":
        simulate_loop(
            study_id=args.study_id,
            n_trials=args.n_trials,
            n_batch=args.n_batch,
            batch_size=args.batch_size,
            rec_mode=args.rec_mode,
            seed=args.seed,
        )
    elif args.command == "sync-results":
        sync_results(study_id=args.study_id, results_file=args.results_file)
    elif args.command == "summary":
        print_summary(args.study_id, as_json=args.json)
    else:
        raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
