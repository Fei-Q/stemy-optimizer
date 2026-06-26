"""Local Streamlit dashboard for the SteMy Ax Bayesian optimization workflow.

Run from the same folder as bo_ax_noisy.py and simulator_noise_updated.py:

    streamlit run stemy_gui.py

This dashboard is a thin GUI wrapper around the backend functions in
bo_ax_noisy.py. It preserves the CLI behavior and result-file structure.
"""

from __future__ import annotations

from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
import importlib
import json
import sys
import traceback
from typing import Any

import pandas as pd
import streamlit as st

import stemy_plots as plots


APP_DIR = Path(__file__).resolve().parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))


# -----------------------------------------------------------------------------
# Backend loading
# -----------------------------------------------------------------------------


@st.cache_resource(show_spinner=False)
def load_backend(module_name: str):
    """Import the selected BO backend module."""
    return importlib.import_module(module_name)


def try_load_backend(module_name: str):
    try:
        return load_backend(module_name), None
    except Exception as exc:  # noqa: BLE001 - show actionable error in GUI
        return None, exc


def capture_backend_call(fn, *args, **kwargs):
    """Run backend call and capture printed CLI output."""
    buffer = StringIO()
    with redirect_stdout(buffer):
        result = fn(*args, **kwargs)
    return result, buffer.getvalue()


def safe_read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def load_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, sep="\t")


def list_studies(results_dir: Path) -> list[str]:
    if not results_dir.exists():
        return []
    return sorted([p.name for p in results_dir.iterdir() if p.is_dir()])


def get_paths(backend, study_id: str) -> dict[str, Path]:
    try:
        return {k: Path(v) for k, v in backend.study_paths(study_id).items()}
    except Exception:
        root = Path(getattr(backend, "RESULTS_DIR", APP_DIR.parent / "results")) / study_id
        return {
            "root": root,
            "config": root / f"{study_id}_config.tsv",
            "log": root / f"{study_id}_log.tsv",
            "summary": root / f"{study_id}_summary.json",
            "metadata": root / f"{study_id}_metadata.json",
            "ax": root / f"{study_id}_ax_client_snapshot.json",
        }


def current_study_id(studies: list[str]) -> str | None:
    if "study_id" not in st.session_state and studies:
        st.session_state["study_id"] = studies[-1]
    return st.session_state.get("study_id")


def set_study(study_id: str):
    st.session_state["study_id"] = study_id


def display_cli_output(output: str):
    if output.strip():
        st.code(output.strip(), language="text")


def show_exception(exc: Exception):
    st.error(str(exc))
    with st.expander("Traceback"):
        st.code("".join(traceback.format_exception(exc)), language="text")


def numeric_columns(df: pd.DataFrame) -> list[str]:
    cols: list[str] = []
    for col in df.columns:
        s = pd.to_numeric(df[col], errors="coerce")
        if s.notna().any():
            cols.append(col)
    return cols


def status_table(df: pd.DataFrame, height: int = 360):
    if df.empty:
        st.info("No rows yet.")
    else:
        st.dataframe(df, use_container_width=True, height=height)


def available_batches(config_df: pd.DataFrame, log_df: pd.DataFrame) -> list[str]:
    batches = set()
    if not config_df.empty and "batch_id" in config_df.columns:
        batches.update(config_df["batch_id"].dropna().astype(str).tolist())
    if not log_df.empty and "batch_id" in log_df.columns:
        batches.update(log_df["batch_id"].dropna().astype(str).tolist())
    return sorted(batches)


def latest_proposed_configs(config_df: pd.DataFrame, batch_id: str) -> pd.DataFrame:
    if config_df.empty:
        return pd.DataFrame()
    df = config_df[config_df["batch_id"].astype(str) == str(batch_id)].copy()
    if df.empty:
        return df
    if "status" in df.columns:
        return df[df["status"].astype(str) == "proposed"].copy()
    return df


def comma_text_to_list(text: str | None) -> list[str] | None:
    if not text or not text.strip():
        return None
    return [part.strip() for part in text.split(",") if part.strip()]


def parameter_names_from_metadata(metadata: dict[str, Any]) -> list[str]:
    params = metadata.get("param_names", [])
    return [str(x) for x in params]


def metadata_or_backend_params(backend, metadata: dict[str, Any]) -> list[str]:
    params = parameter_names_from_metadata(metadata)
    if params:
        return params
    try:
        return list(backend.DEFAULT_OPTIMIZER_PARAMS)
    except Exception:
        return []


def backend_rows(backend, study_id: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Load backend config/log rows directly from the active study files."""
    paths = backend.study_paths(study_id)
    config_rows = backend.read_tsv(paths["config"])
    log_rows = backend.read_tsv(paths["log"])
    return config_rows, log_rows


def _batch_sort_key(batch_id: str) -> tuple[int, str]:
    text = str(batch_id)
    if len(text) > 1 and text[0].upper() == "B" and text[1:].isdigit():
        return int(text[1:]), text
    return 10**9, text


def running_batches_from_rows(log_rows: list[dict[str, Any]]) -> list[str]:
    batches = {
        str(row.get("batch_id", ""))
        for row in log_rows
        if row.get("batch_id") and str(row.get("run_status", "")) == "running"
    }
    return sorted(batches, key=_batch_sort_key)


def parse_bool(value: Any) -> bool:
    """Parse common string/bool values from TSV rows."""
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes", "y"}

def completed_unsynced_count(log_rows: list[dict[str, Any]]) -> int:
    count = 0
    for row in log_rows:
        if str(row.get("run_status", "")) != "completed":
            continue
        if parse_bool(row.get("synced", "false")):
            continue
        count += 1
    return count


def proposed_config_ids_by_batch(backend, config_rows: list[dict[str, Any]]) -> dict[str, list[str]]:
    """Return latest proposed config IDs grouped by batch."""
    batches = sorted(
        {str(row.get("batch_id", "")) for row in config_rows if row.get("batch_id")},
        key=_batch_sort_key,
    )
    grouped: dict[str, list[str]] = {}
    for batch_id in batches:
        try:
            latest = backend.latest_config_rows(config_rows, batch_id)
        except Exception:
            latest = [row for row in config_rows if str(row.get("batch_id", "")) == batch_id]
        ids = [
            str(row.get("config_id", ""))
            for row in latest
            if str(row.get("status", "")) == "proposed" and row.get("config_id")
        ]
        if ids:
            grouped[batch_id] = ids
    return grouped


def next_available_run_ids(log_rows: list[dict[str, Any]], batch_id: str, n: int, plate_id: str = "P1") -> list[str]:
    """Generate non-conflicting run IDs for a batch."""
    existing = {
        str(row.get("run_id", ""))
        for row in log_rows
        if str(row.get("batch_id", "")) == str(batch_id)
    }
    run_ids: list[str] = []
    i = 1
    while len(run_ids) < n:
        candidate = f"{plate_id}-W{i}"
        if candidate not in existing:
            run_ids.append(candidate)
            existing.add(candidate)
        i += 1
    return run_ids


def resume_automated_simulation_loop(
    backend,
    *,
    study_id: str,
    n_trials: int,
    batch_size: int,
    seed: int,
    variability_strength: float,
) -> None:
    """Resume-aware automated loop for the GUI.

    It clears already-pending workflow state before requesting new Ax trials:
    completed-unsynced -> sync; running -> simulate+sync; proposed -> run+simulate+sync;
    only then asks Ax for new recommendations.
    """
    if n_trials < 1:
        raise ValueError("n_trials must be >= 1")
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")

    backend.ensure_study_for_simulation(study_id, seed)

    remaining = int(n_trials)
    completed_now = 0
    safety_counter = 0
    max_steps = max(20, int(n_trials) * 10)

    while remaining > 0:
        safety_counter += 1
        if safety_counter > max_steps:
            raise RuntimeError("Automated loop stopped by safety guard; check study state manually.")

        config_rows, log_rows = backend_rows(backend, study_id)

        n_unsynced = completed_unsynced_count(log_rows)
        if n_unsynced:
            print(f"Syncing {n_unsynced} completed unsynced run(s).")
            backend.sync_results(study_id=study_id)
            continue

        running_batches = running_batches_from_rows(log_rows)
        if running_batches:
            for batch_id in running_batches:
                _, current_log_rows = backend_rows(backend, study_id)
                n_running = sum(
                    1
                    for row in current_log_rows
                    if str(row.get("batch_id", "")) == batch_id
                    and str(row.get("run_status", "")) == "running"
                )
                if not n_running:
                    continue
                print(f"Simulating {n_running} running row(s) in {batch_id}.")
                backend.simulate_completed_runs(
                    study_id=study_id,
                    batch_id=batch_id,
                    seed=seed,
                    variability_strength=variability_strength,
                )
                backend.sync_results(study_id=study_id)
                completed_now += n_running
                remaining -= n_running
                if remaining <= 0:
                    break
            continue

        proposed_by_batch = proposed_config_ids_by_batch(backend, config_rows)
        if proposed_by_batch:
            batch_id = sorted(proposed_by_batch, key=_batch_sort_key)[0]
            config_ids = proposed_by_batch[batch_id]
            run_ids = next_available_run_ids(log_rows, batch_id, len(config_ids), plate_id="P1")
            print(f"Creating run rows for {len(config_ids)} proposed config(s) in {batch_id}.")
            backend.run_configs(
                study_id=study_id,
                batch_id=batch_id,
                config_ids=config_ids,
                plate_id="P1",
                run_ids=run_ids,
            )
            continue

        n = min(int(batch_size), remaining)
        print(f"Generating new batch with {n} proposed config(s).")
        batch_id = backend.recommend_configs(
            study_id=study_id,
            n_trials=n,
            policy="balanced",
            interactive=False,
        )
        print(f"Creating run rows for {batch_id}.")
        backend.run_configs(
            study_id=study_id,
            batch_id=batch_id,
            config_ids=None,
            plate_id="P1",
            run_ids=None,
        )
        print(f"Simulating and syncing {batch_id}.")
        backend.simulate_completed_runs(
            study_id=study_id,
            batch_id=batch_id,
            seed=seed,
            variability_strength=variability_strength,
        )
        backend.sync_results(study_id=study_id)
        completed_now += n
        remaining -= n

    backend.save_summary(study_id)
    print(f"\nResume-aware automated simulation complete. Completed/synced approximately {completed_now} run(s) in this action.")


# -----------------------------------------------------------------------------
# App
# -----------------------------------------------------------------------------


st.set_page_config(page_title="SteMy BO Dashboard", layout="wide")
st.title("SteMy BO Dashboard")
st.caption("Local GUI wrapper for proposed configs, actual-as-treated run logs, noisy simulation, syncing, and diagnostics.")

with st.sidebar:
    st.header("Backend")
    backend_module = st.text_input("Backend module", value="bo_ax_noisy")
    st.caption("Use bo_ax_noisy as the canonical backend. Set variability strength to 0.0 for exact execution/no-noise runs.")

backend, backend_error = try_load_backend(backend_module)

if backend_error is not None:
    st.error("Could not import backend module.")
    st.write("Install backend dependencies and run from the folder containing `bo_ax_noisy.py` and `simulator_noise_updated.py`.")
    st.code("pip install streamlit pandas matplotlib numpy ax-platform", language="bash")
    show_exception(backend_error)
    st.stop()

results_dir = Path(getattr(backend, "RESULTS_DIR", APP_DIR.parent / "results"))
results_dir.mkdir(parents=True, exist_ok=True)

studies = list_studies(results_dir)
with st.sidebar:
    st.header("Study")
    st.write(f"Results directory: `{results_dir}`")
    if studies:
        selected = st.selectbox(
            "Load existing study",
            options=studies,
            index=studies.index(current_study_id(studies)) if current_study_id(studies) in studies else len(studies) - 1,
        )
        set_study(selected)
    else:
        st.info("No studies found yet. Create one in the Study tab.")

study_id = current_study_id(studies)
paths = get_paths(backend, study_id) if study_id else None
metadata = safe_read_json(paths["metadata"]) if paths else {}
summary = safe_read_json(paths["summary"]) if paths else {}
config_df = load_table(paths["config"]) if paths else pd.DataFrame()
log_df = load_table(paths["log"]) if paths else pd.DataFrame()
param_names = metadata_or_backend_params(backend, metadata)
batches = available_batches(config_df, log_df)
latest_batch = summary.get("latest_batch") or (batches[-1] if batches else None)

# The tabs intentionally mirror the CLI plus plotting/diagnostics.
tab_study, tab_recommend, tab_edit, tab_run, tab_simulate, tab_sync, tab_results, tab_diag, tab_files = st.tabs(
    [
        "Study",
        "Recommend",
        "Review & Edit",
        "Run",
        "Simulate",
        "Sync",
        "Results",
        "Variability Diagnostics",
        "Files",
    ]
)


# -----------------------------------------------------------------------------
# Study tab
# -----------------------------------------------------------------------------


with tab_study:
    st.subheader("Create or inspect a study")

    col_create, col_status = st.columns([1, 1])

    with col_create:
        st.markdown("#### Create study")
        default_study_id = "demo_noisy"
        with st.form("create_study_form"):
            new_study_id = st.text_input("Study ID", value=default_study_id)
            seed = st.number_input("Seed", min_value=0, max_value=2_147_483_647, value=123, step=1)

            all_opt_params = list(getattr(backend, "DEFAULT_OPTIMIZER_PARAMS", []))
            chosen_params = st.multiselect(
                "Optimizer parameters",
                options=all_opt_params,
                default=all_opt_params,
                help="Leave as default unless testing a smaller optimizer search space.",
            )
            overwrite = st.checkbox("Overwrite existing study", value=False)
            submitted = st.form_submit_button("Create study")

        if submitted:
            try:
                params_arg = ",".join(chosen_params) if chosen_params else None
                _, output = capture_backend_call(
                    backend.create_study,
                    study_id=new_study_id,
                    params=params_arg,
                    seed=int(seed),
                    overwrite=overwrite,
                )
                set_study(new_study_id)
                st.success(f"Created study `{new_study_id}`.")
                display_cli_output(output)
                st.rerun()
            except Exception as exc:  # noqa: BLE001
                show_exception(exc)

    with col_status:
        st.markdown("#### Current study")
        if study_id:
            st.write(f"Study: `{study_id}`")
            st.write(f"Backend: `{backend_module}`")
            st.write(f"Objective: `{summary.get('objective_name', getattr(backend, 'OBJECTIVE_NAME', 'ctnt_pct'))}`")
            st.write(f"Version: `{summary.get('version', metadata.get('version', getattr(backend, 'VERSION', 'unknown')) )}`")
            st.write(f"Latest batch: `{latest_batch}`")
            if st.button("Refresh summary", key="refresh_summary_main"):
                try:
                    result, output = capture_backend_call(backend.save_summary, study_id)
                    st.success("Summary refreshed.")
                    display_cli_output(output)
                    st.rerun()
                except Exception as exc:  # noqa: BLE001
                    show_exception(exc)
        else:
            st.info("Create or select a study.")

    if study_id:
        st.markdown("#### Counts")
        counts = summary.get("counts", {})
        metric_cols = st.columns(4)
        keys = [
            "n_batches",
            "n_configs",
            "n_runs",
            "n_completed_runs",
            "n_synced_runs",
            "n_running_runs",
            "n_unsynced_completed_runs",
            "n_failed_runs",
        ]
        for i, key in enumerate(keys):
            metric_cols[i % 4].metric(key, counts.get(key, 0))

        warnings = summary.get("warnings", [])
        if warnings:
            st.warning("\n".join(str(w) for w in warnings))

        st.markdown("#### Metadata")
        st.json(metadata or {})

        st.markdown("#### Summary JSON")
        st.json(summary or {})


# -----------------------------------------------------------------------------
# Recommend tab
# -----------------------------------------------------------------------------


with tab_recommend:
    st.subheader("Generate proposed configs")
    if not study_id:
        st.info("Create or select a study first.")
    else:
        with st.form("recommend_form"):
            n_trials = st.number_input("Number of proposed configs", min_value=1, max_value=96, value=6, step=1)
            policy = st.text_input("Policy", value="balanced")
            submitted = st.form_submit_button("Generate recommendations")

        if submitted:
            try:
                batch_id, output = capture_backend_call(
                    backend.recommend_configs,
                    study_id=study_id,
                    n_trials=int(n_trials),
                    policy=policy,
                    interactive=False,
                )
                st.success(f"Generated recommendations for `{batch_id}`.")
                display_cli_output(output)
                st.rerun()
            except Exception as exc:  # noqa: BLE001
                show_exception(exc)

        st.markdown("#### Config table")
        status_table(config_df, height=420)


# -----------------------------------------------------------------------------
# Review/edit tab
# -----------------------------------------------------------------------------


with tab_edit:
    st.subheader("Review and edit proposed configs")
    if not study_id:
        st.info("Create or select a study first.")
    elif config_df.empty:
        st.info("No configs yet. Generate recommendations first.")
    else:
        batch_options = batches or []
        selected_batch = st.selectbox(
            "Batch",
            options=batch_options,
            index=batch_options.index(latest_batch) if latest_batch in batch_options else max(len(batch_options) - 1, 0),
            key="edit_batch",
        )
        batch_configs = config_df[config_df["batch_id"].astype(str) == str(selected_batch)].copy()
        status_table(batch_configs, height=260)

        editable_configs = batch_configs[~batch_configs.get("status", pd.Series([], dtype=str)).astype(str).isin(["running", "completed"])]
        config_options = editable_configs["config_id"].dropna().astype(str).tolist() if not editable_configs.empty else []

        if not config_options:
            st.info("No editable configs in this batch.")
        else:
            selected_config = st.selectbox("Config to edit", options=config_options, key="edit_config_id")
            current_row = batch_configs[batch_configs["config_id"].astype(str) == selected_config].tail(1)
            current = current_row.iloc[0].to_dict() if not current_row.empty else {}

            st.markdown("#### Edit optimizer parameter values")
            with st.form("edit_config_form"):
                updates: dict[str, float] = {}
                cols = st.columns(2)
                for i, name in enumerate(param_names):
                    spec = backend.PARAM_SPECS.get(name, {})
                    low, high = spec.get("bounds", (None, None))
                    cur_value = pd.to_numeric(pd.Series([current.get(name)]), errors="coerce").iloc[0]
                    if pd.isna(cur_value):
                        cur_value = float(spec.get("default", 0.0))
                    with cols[i % 2]:
                        value = st.number_input(
                            name,
                            value=float(cur_value),
                            min_value=float(low) if low is not None else None,
                            max_value=float(high) if high is not None else None,
                            step=0.1,
                            format="%.6g",
                            key=f"edit_{selected_batch}_{selected_config}_{name}",
                        )
                    if float(value) != float(cur_value):
                        updates[name] = float(value)
                notes = st.text_area("Notes", value="edited in GUI")
                submitted = st.form_submit_button("Save edited config")

            if submitted:
                if not updates:
                    st.info("No parameter value changed.")
                else:
                    try:
                        new_config_id, output = capture_backend_call(
                            backend.edit_config,
                            study_id=study_id,
                            batch_id=selected_batch,
                            config_id=selected_config,
                            updates=updates,
                            notes=notes,
                            print_after=True,
                        )
                        st.success(f"Saved edited config `{new_config_id}`.")
                        display_cli_output(output)
                        st.rerun()
                    except Exception as exc:  # noqa: BLE001
                        show_exception(exc)


# -----------------------------------------------------------------------------
# Run tab
# -----------------------------------------------------------------------------


with tab_run:
    st.subheader("Create run rows")
    st.write("Convert proposed optimizer configs into run-log rows. This tab does not simulate outcomes.")

    if not study_id:
        st.info("Create or select a study first.")
    else:
        batch_options = batches or []
        if not batch_options:
            st.info("No proposed batches yet. Generate recommendations first, or use the Simulate tab's automated dry run to create and process batches automatically.")
        else:
            selected_batch = st.selectbox(
                "Batch",
                options=batch_options,
                index=batch_options.index(latest_batch) if latest_batch in batch_options else max(len(batch_options) - 1, 0),
                key="run_batch_isolated",
            )

            proposed = latest_proposed_configs(config_df, selected_batch)
            proposed_ids = proposed["config_id"].dropna().astype(str).tolist() if not proposed.empty else []

            col_form, col_table = st.columns([1, 2])
            with col_form:
                st.markdown("#### Run proposed configs")
                if not proposed_ids:
                    st.info("No proposed configs available in this batch. They may already be running/completed/rejected.")
                with st.form("run_configs_form_isolated"):
                    config_ids = st.multiselect("Config IDs", options=proposed_ids, default=proposed_ids)
                    plate_id = st.text_input("Plate ID", value="P1")
                    custom_run_ids = st.text_input("Run IDs, optional comma-separated", value="")
                    submitted = st.form_submit_button("Create run rows")

                if submitted:
                    try:
                        run_ids_arg = comma_text_to_list(custom_run_ids)
                        result, output = capture_backend_call(
                            backend.run_configs,
                            study_id=study_id,
                            batch_id=selected_batch,
                            config_ids=config_ids or None,
                            plate_id=plate_id,
                            run_ids=run_ids_arg,
                        )
                        st.success(f"Created {len(result)} run row(s).")
                        display_cli_output(output)
                        st.rerun()
                    except Exception as exc:  # noqa: BLE001
                        show_exception(exc)

            with col_table:
                st.markdown("#### Proposed configs in selected batch")
                if proposed.empty:
                    st.info("No proposed configs in this batch.")
                else:
                    status_table(proposed, height=360)

            st.markdown("#### Run log")
            status_table(log_df, height=360)


# -----------------------------------------------------------------------------
# Simulate tab
# -----------------------------------------------------------------------------


with tab_simulate:
    st.subheader("Simulate noisy execution")
    st.write("Fill actual values, deviations, and cTnT% for running rows. This tab does not create proposed configs unless you use the automated dry run.")

    if not study_id:
        st.info("Create or select a study first.")
    else:
        col_manual, col_loop = st.columns([1, 1])

        with col_manual:
            st.markdown("#### Manual batch simulation")
            st.caption("Requires existing running rows. No QC labels or QC scores are assigned here.")

            batch_options = batches or []
            if not batch_options:
                st.info("No batches exist yet. Use automated dry run below, or generate recommendations first.")
            else:
                selected_batch = st.selectbox(
                    "Batch to simulate",
                    options=batch_options,
                    index=batch_options.index(latest_batch) if latest_batch in batch_options else max(len(batch_options) - 1, 0),
                    key="simulate_batch_isolated",
                )

                running_for_batch = pd.DataFrame()
                if not log_df.empty and "batch_id" in log_df.columns and "run_status" in log_df.columns:
                    running_for_batch = log_df[
                        (log_df["batch_id"].astype(str) == str(selected_batch))
                        & (log_df["run_status"].astype(str) == "running")
                    ].copy()

                st.markdown("Running rows in selected batch")
                status_table(running_for_batch, height=220)

                with st.form("simulate_batch_form_isolated"):
                    sim_seed = st.number_input("Simulation seed", min_value=0, max_value=2_147_483_647, value=123, step=1)
                    mode = st.selectbox("Execution mode", ["Exact execution", "Low variability", "Moderate variability", "Custom"], key="manual_sim_mode")
                    default_strength = {"Exact execution": 0.0, "Low variability": 0.05, "Moderate variability": 0.10, "Custom": 0.05}[mode]
                    variability_strength = st.number_input(
                        "Variability strength",
                        min_value=0.0,
                        max_value=2.0,
                        value=float(default_strength),
                        step=0.01,
                        format="%.4f",
                        key="manual_variability_strength",
                    )
                    submitted = st.form_submit_button("Simulate selected batch")

                if submitted:
                    try:
                        _, output = capture_backend_call(
                            backend.simulate_completed_runs,
                            study_id=study_id,
                            batch_id=selected_batch,
                            seed=int(sim_seed),
                            variability_strength=float(variability_strength),
                        )
                        st.success("Simulation completed.")
                        display_cli_output(output)
                        st.rerun()
                    except Exception as exc:  # noqa: BLE001
                        show_exception(exc)

        with col_loop:
            st.markdown("#### Automated dry run / resume loop")
            st.caption("Resume-aware workflow: sync completed rows → simulate running rows → run proposed configs → generate new batches as needed.")
            with st.form("simulate_loop_form_isolated"):
                loop_trials = st.number_input(
                    "Target completed/synced runs for this action",
                    min_value=1,
                    max_value=500,
                    value=24,
                    step=1,
                    help="If pending proposed/running rows already exist, the loop processes those first and counts them toward this target.",
                )
                batch_size = st.number_input("New-batch size", min_value=1, max_value=96, value=6, step=1)
                loop_seed = st.number_input("Loop seed", min_value=0, max_value=2_147_483_647, value=123, step=1)
                loop_mode = st.selectbox("Loop execution mode", ["Exact execution", "Low variability", "Moderate variability", "Custom"], key="loop_sim_mode")
                loop_default = {"Exact execution": 0.0, "Low variability": 0.05, "Moderate variability": 0.10, "Custom": 0.05}[loop_mode]
                loop_variability = st.number_input(
                    "Loop variability strength",
                    min_value=0.0,
                    max_value=2.0,
                    value=float(loop_default),
                    step=0.01,
                    format="%.4f",
                    key="loop_variability_strength",
                )
                submitted = st.form_submit_button("Run automated dry run / resume")

            if submitted:
                try:
                    _, output = capture_backend_call(
                        resume_automated_simulation_loop,
                        backend,
                        study_id=study_id,
                        n_trials=int(loop_trials),
                        batch_size=int(batch_size),
                        seed=int(loop_seed),
                        variability_strength=float(loop_variability),
                    )
                    st.success("Automated dry run / resume completed.")
                    display_cli_output(output)
                    st.rerun()
                except Exception as exc:  # noqa: BLE001
                    show_exception(exc)

        st.markdown("#### Current run log")
        status_table(log_df, height=420)


# -----------------------------------------------------------------------------
# Sync tab
# -----------------------------------------------------------------------------


with tab_sync:
    st.subheader("Actual-as-treated Ax sync")
    if not study_id:
        st.info("Create or select a study first.")
    else:
        st.write(
            "Sync attaches a new Ax trial using actual executed optimizer parameters, "
            "then completes that actual-as-treated trial with cTnT%. It does not treat the original planned trial as exact execution."
        )
        if not log_df.empty:
            unsynced = log_df[(log_df.get("run_status", "") == "completed") & (~log_df.get("synced", "false").astype(str).str.lower().isin(["true", "1", "yes", "y"]))].copy()
            st.markdown("#### Completed unsynced runs")
            status_table(unsynced, height=240)
        with st.form("sync_results_form"):
            alt_file = st.text_input("Alternate results file, optional", value="")
            submitted = st.form_submit_button("Sync completed results into Ax")

        if submitted:
            try:
                count, output = capture_backend_call(
                    backend.sync_results,
                    study_id=study_id,
                    results_file=alt_file or None,
                )
                st.success(f"Synced {count} run(s).")
                display_cli_output(output)
                st.rerun()
            except Exception as exc:  # noqa: BLE001
                show_exception(exc)


# -----------------------------------------------------------------------------
# Results tab
# -----------------------------------------------------------------------------


with tab_results:
    st.subheader("Outcome plots")
    if not study_id:
        st.info("Create or select a study first.")
    else:
        col_a, col_b = st.columns(2)
        with col_a:
            st.pyplot(plots.plot_ctnt_over_time(log_df), clear_figure=True)
        with col_b:
            st.pyplot(plots.plot_batch_performance(log_df), clear_figure=True)

        st.markdown("#### Best observed")
        st.json(summary.get("best_observed", {}))

        st.markdown("#### Current Ax recommendation")
        st.json(summary.get("current_recommendation", {}))


# -----------------------------------------------------------------------------
# Diagnostics tab
# -----------------------------------------------------------------------------


with tab_diag:
    st.subheader("Planned/target vs actual variability diagnostics")
    if not study_id:
        st.info("Create or select a study first.")
    elif log_df.empty:
        st.info("No log data yet.")
    else:
        params_for_plot = plots.available_parameters_for_planned_actual(log_df)
        delta_cols = plots.delta_columns(log_df)
        if not params_for_plot:
            st.info("No planned/target and actual columns available yet.")
        else:
            selected_param = st.selectbox("Parameter", options=params_for_plot)
            col_a, col_b = st.columns(2)
            with col_a:
                st.pyplot(plots.plot_planned_vs_actual(log_df, selected_param), clear_figure=True)
            with col_b:
                st.pyplot(plots.plot_delta_vs_ctnt(log_df, selected_param), clear_figure=True)

        group = st.radio("Deviation group", options=["all", "dynamic", "control"], horizontal=True)
        col_c, col_d = st.columns(2)
        with col_c:
            st.pyplot(plots.plot_delta_by_parameter(log_df, group=group), clear_figure=True)
        with col_d:
            st.pyplot(plots.plot_delta_heatmap(log_df, group=group), clear_figure=True)

        if delta_cols:
            st.markdown("#### Deviation columns")
            status_table(log_df[[c for c in ["batch_id", "run_id", "ctnt_pct", *delta_cols] if c in log_df.columns]], height=320)


# -----------------------------------------------------------------------------
# Files tab
# -----------------------------------------------------------------------------


with tab_files:
    st.subheader("Raw result files")
    if not study_id or not paths:
        st.info("Create or select a study first.")
    else:
        st.write(f"Study directory: `{paths['root']}`")
        col_config, col_log = st.columns(2)
        with col_config:
            st.markdown("#### Config TSV")
            status_table(config_df, height=300)
            if paths["config"].exists():
                st.download_button(
                    "Download config TSV",
                    data=paths["config"].read_bytes(),
                    file_name=paths["config"].name,
                    mime="text/tab-separated-values",
                )
        with col_log:
            st.markdown("#### Log TSV")
            status_table(log_df, height=300)
            if paths["log"].exists():
                st.download_button(
                    "Download log TSV",
                    data=paths["log"].read_bytes(),
                    file_name=paths["log"].name,
                    mime="text/tab-separated-values",
                )

        st.markdown("#### JSON files")
        for key in ["metadata", "summary", "ax"]:
            path = paths.get(key)
            if path and path.exists():
                st.download_button(
                    f"Download {path.name}",
                    data=path.read_bytes(),
                    file_name=path.name,
                    mime="application/json" if path.suffix == ".json" else "application/octet-stream",
                )
