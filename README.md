# SteMy Optimizer

Bayesian optimization prototype for SteMy's iPSC-to-cardiomyocyte differentiation workflow.

The project provides both:

- a CLI backend for Ax-based Bayesian optimization; and
- a local Streamlit GUI dashboard for proposed configs, actual-as-treated run logs, noisy simulation, syncing, results, and variability diagnostics.

Generated study outputs are written to `results/<study_id>/` and are ignored by Git.

## Repository layout

```text
stemy-optimizer/
├── scripts/
│   ├── bo_ax.py              # original Ax optimizer CLI
│   ├── bo_ax_noisy.py        # canonical noisy/actual-as-treated optimizer backend
│   ├── simulator.py          # original or exact simulator backend, if used
│   ├── simulator_noisy.py    # noisy planned-to-actual simulator backend
│   ├── stemy_gui.py          # Streamlit GUI dashboard
│   └── stemy_plots.py        # plotting utilities used by GUI/notebooks
├── results/                  # generated study outputs
├── requirements.txt
├── README.md
└── .gitignore
```

If your updated GUI file is named `stemy_gui_updated.py`, either run it directly:

```bash
streamlit run scripts/stemy_gui_updated.py
```

or replace the older GUI file:

```bash
mv scripts/stemy_gui_updated.py scripts/stemy_gui.py
```

## Setup

Create and activate the Conda environment:

```bash
conda create -n stemy-opt python=3.11 -y
conda activate stemy-opt
```

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

If pip installation is killed on your machine, install the large packages through conda-forge first:

```bash
conda activate stemy-opt
conda install -c conda-forge ax-platform numpy pandas matplotlib streamlit
```

## Recommended backend

Use `bo_ax_noisy.py` as the canonical backend. It supports both exact and noisy execution:

```text
variability_strength = 0.0  -> exact/no-noise execution; planned = actual
variability_strength > 0.0  -> noisy planned-vs-actual execution
```

This avoids maintaining separate GUI paths for noisy and no-noise experiments.

## GUI dashboard

Run from the repository root or from the `scripts/` folder containing the GUI/backend files:

```bash
streamlit run scripts/stemy_gui.py
```

or, if already inside `scripts/`:

```bash
streamlit run stemy_gui.py
```

Then open the localhost URL printed by Streamlit, usually:

```text
http://localhost:8501
```

The GUI dashboard wraps the CLI behavior:

1. Create/load a study.
2. Generate proposed configs.
3. Review or edit proposed configs.
4. Create run-log rows from proposed configs.
5. Simulate running rows with exact or noisy execution.
6. Sync completed results into Ax using actual-as-treated parameter values.
7. Review outcome plots and variability diagnostics.

The GUI does **not** implement QC status, QC score, eligibility, filtering, or observation-noise logic. Those remain reserved for a separate SteMy QC/reproducibility module.

## Core CLI commands

Create a persistent study:

```bash
python scripts/bo_ax_noisy.py create-study --study-id test1 --seed 123
```

Create a study with a selected optimizer parameter subset:

```bash
python scripts/bo_ax_noisy.py create-study \
  --study-id test1 \
  --params chir_conc_uM,chir_dur_hr,iwp2_start_day,iwp2_conc_uM,iwp2_dur_hr \
  --seed 123
```

Generate recommended configs for the next auto-incremented batch:

```bash
python scripts/bo_ax_noisy.py recommend --study-id test1 --n-trials 6
```

Edit one proposed config while preserving config history:

```bash
python scripts/bo_ax_noisy.py edit-config \
  --study-id test1 \
  --batch-id B01 \
  --config-id C03 \
  --set chir_conc_uM=11.5 \
  --set iwp2_conc_uM=5.0 \
  --notes "more conservative CHIR"
```

Create run-log rows from the latest configs in a batch:

```bash
python scripts/bo_ax_noisy.py run --study-id test1 --batch-id B01 --plate-id P1
```

Run a fully automated simulator loop:

```bash
python scripts/bo_ax_noisy.py simulate --study-id sim1 --n-trials 30 --batch-size 6 --seed 123
```

Sync completed run results into Ax:

```bash
python scripts/bo_ax_noisy.py sync-results --study-id test1
```

Print and save a study summary:

```bash
python scripts/bo_ax_noisy.py summary --study-id test1
```

Machine-readable summary:

```bash
python scripts/bo_ax_noisy.py summary --study-id test1 --json
```

## Output files

Each study writes outputs under `results/<study_id>/`:

```text
results/<study_id>/
├── <study_id>_ax_client_snapshot.json
├── <study_id>_metadata.json
├── <study_id>_summary.json
├── <study_id>_config.tsv
└── <study_id>_log.tsv
```

`<study_id>_config.tsv` stores proposed experimental configurations and edit history.

Core config columns:

```text
study_id
batch_id
config_id
status
trial_index
created_at
started_at
finished_at
created_by
notes
<optimizer parameter columns...>
```

Allowed config statuses:

```text
proposed
rejected
running
completed
failed
```

`<study_id>_log.tsv` stores actual executed runs and measured or simulated outcomes.

Core log columns include:

```text
study_id
batch_id
run_id
config_id
trial_index
run_status
ctnt_pct
synced
started_at
finished_at
notes
<planned/target parameter columns...>
<actual parameter columns...>
<delta parameter columns...>
```

`ctnt_pct` is the optimizer objective: measured or simulated cTnT-positive percentage. `synced=true` means the row has been reported back to Ax.

In the noisy/actual-as-treated workflow, the log table distinguishes:

```text
x_planned / x_actual / delta_x  -> dynamic optimizer variables
z_target / z_actual / delta_z   -> static/control variables
```

## ID conventions

```text
study_id:    test1
batch_id:    B01
config_id:   C01 or C01-1
run_id:      P1-W3
trial_index: Ax integer, e.g. 1
```

`batch_id` is one optimizer recommendation/update cycle. `config_id` identifies one proposed experimental setup within a batch. Edited configs use version suffixes, such as `C03-1`. `run_id` identifies the physical execution location, typically plate and well.

## Plotting and diagnostics

`stemy_plots.py` provides reusable plotting utilities for:

- cTnT over completed run order
- best-so-far cTnT
- batch performance
- planned/target vs actual values
- parameter deviations
- deviation heatmaps
- deviation vs outcome

The GUI imports these functions directly, but they can also be used from notebooks/scripts.

## Versioning

Keep the main executable backend named `scripts/bo_ax_noisy.py` for the current actual-as-treated workflow. Use Git tags for stable versions, for example:

```bash
git tag -a v1.1.0 -m "SteMy optimizer CLI v1.1.0"
git push origin v1.1.0
```
