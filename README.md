# SteMy Optimizer

Bayesian optimization prototype for SteMy's iPSC-to-cardiomyocyte differentiation workflow.

The current CLI uses Ax to recommend candidate experimental configurations, preserve user edits, log executed runs, sync completed outcomes back into the optimizer, and run fully simulated optimization loops.

## Repository layout

```text
stemy-optimizer/
├── scripts/
│   ├── bo_ax.py          # main Ax optimizer CLI
│   └── simulator_v1.py   # lightweight simulator used for development/testing
├── requirements.txt
├── README.md
└── .gitignore
```

Generated study outputs are written to `results/<study_id>/` and are ignored by Git.

## Setup

Create and activate a Python environment:

```bash
conda create -n stemy-opt python=3.11 -y
conda activate stemy-opt
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Core commands

Create a persistent study:

```bash
python scripts/bo_ax.py create-study --study-id test1 --seed 123
```

Create a study with a selected optimizer parameter subset:

```bash
python scripts/bo_ax.py create-study \
  --study-id test1 \
  --params chir_conc_uM,chir_dur_hr,iwp2_start_day,iwp2_conc_uM,iwp2_dur_hr \
  --seed 123
```

Generate recommended configs for the next auto-incremented batch:

```bash
python scripts/bo_ax.py recommend --study-id test1 --n-trials 6
```

Edit one proposed config while preserving config history:

```bash
python scripts/bo_ax.py edit-config \
  --study-id test1 \
  --batch-id B01 \
  --config-id C03 \
  --set chir_conc_uM=11.5 \
  --set iwp2_conc_uM=5.0 \
  --notes "more conservative CHIR"
```

Create run-log rows from the latest configs in a batch:

```bash
python scripts/bo_ax.py run --study-id test1 --batch-id B01 --plate-id P1
```

Run a fully automated simulator loop:

```bash
python scripts/bo_ax.py simulate --study-id sim1 --n-trials 30 --batch-size 6 --seed 123
```

Sync completed run results into Ax. This is a fallback/debug command; simulator mode and future backend integration should call sync automatically.

```bash
python scripts/bo_ax.py sync-results --study-id test1
```

Print and save a study summary:

```bash
python scripts/bo_ax.py summary --study-id test1
```

Machine-readable summary:

```bash
python scripts/bo_ax.py summary --study-id test1 --json
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

Core columns:

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

Allowed config statuses in this version:

```text
proposed
rejected
running
completed
failed
```

`<study_id>_log.tsv` stores actual executed runs and measured outcomes.

Core columns:

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
<actual parameter / run data columns...>
```

`ctnt_pct` is the optimizer objective: measured or simulated cTnT-positive percentage. `synced=true` means the row has been reported back to Ax.

## ID conventions

```text
study_id:    test1
batch_id:    B01
config_id:   C01 or C01-1
run_id:      P1-W3
trial_index: Ax integer, e.g. 1
```

`batch_id` is one optimizer recommendation/update cycle. `config_id` identifies one proposed experimental setup within a batch. Edited configs use version suffixes, such as `C03-1`. `run_id` identifies the physical execution location, typically plate and well.

## Versioning

Keep the executable script named `scripts/bo_ax.py`. Track versions with Git tags, for example:

```bash
git tag -a v1.1.0 -m "SteMy optimizer CLI v1.1.0"
git push origin v1.1.0
```
