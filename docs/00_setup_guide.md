# 00 — Setup Guide: Wiring the Full Agent Workspace

This is the first file the agent reads when the project is brand new.
After completing every step in this guide, the agent is fully operational
and can begin executing `/scaffold` commands.

---

## Step 0 — Understand the toolchain

Three tools are working together in this project. Each has a distinct role.

| Tool | Role | Primary files |
|---|---|---|
| **OpenCode** | The IDE agent that writes code and runs commands | All source files |
| **Graphify** | Builds a queryable knowledge graph of the codebase | `graphify-out/` — never edited manually |
| **docs/ + AGENTS.md** | Prescriptive specs — what to build and how | `docs/01–04`, `AGENTS.md` |

The relationship is: docs tell the agent **what to build**, Graphify tells the agent **what already exists**. They are never in conflict — they answer different questions.

---

## Step 1 — Confirm directory layout

The following files must already exist before proceeding. The agent verifies each one:

```
seismic_ml/
├── AGENTS.md                            ✓ must exist
├── .gitignore                           ✓ must exist
├── .opencode/
│   ├── instructions.md                  ✓ must exist
│   ├── commands.md                      ✓ must exist
│   └── plugins/graphify.js              ✓ written by graphify install --platform opencode
├── docs/
│   ├── 01_project_overview.md           ✓ must exist
│   ├── 02_data_pipeline.md              ✓ must exist
│   ├── 03_model_architecture.md         ✓ must exist
│   └── 04_training_and_export.md        ✓ must exist
```

If any of these are missing, stop and restore them before proceeding.
`graphify.js` is written by running `graphify install --platform opencode` from the repo root.

---

## Step 2 — Verify Graphify is installed and wired

Run from the repo root:

```bash
graphify --help                           # confirms graphify CLI is available
```

If this fails, install via:
```bash
uv tool install graphifyy
graphify install --platform opencode
```

Confirm the plugin hook is active:
```bash
cat .opencode/plugins/graphify.js         # must exist and be non-empty
```

Confirm the skill is installed globally:
```bash
ls ~/.config/opencode/skills/graphify/SKILL.md    # must exist
```

If either is missing, re-run `graphify install --platform opencode` from the repo root.

---

## Step 3 — Check AGENTS.md integrity

The `graphify install` command may have appended content to `AGENTS.md`.
The agent must verify the file structure is exactly:

```
1. Project description (top)
2. docs/ index table
3. The 5 Invariants (must be present verbatim)
4. Preprocessing pipeline order (7 steps, must be present)
5. Key architectural facts
6. Session start / end procedures
7. Hard Rules for the Agent
8. Graphify Knowledge Graph section   ← appended by setup, must be at the bottom
```

If the Graphify section is missing, append the content from `AGENTS.md` lines starting with `## Graphify Knowledge Graph`.
If the Invariants or preprocessing order are missing (overwritten), restore them from `docs/01_project_overview.md`.

---

## Step 4 — Generate the three config JSON files

Before any code is scaffolded, all three config files must exist.
The agent generates them from the schemas in the docs.

**configs/preprocessing_config.json** — schema in `docs/01_project_overview.md`
**configs/model_config.json** — schema in `docs/03_model_architecture.md`
**configs/training_config.json** — schema in `docs/04_training_and_export.md`

After generating, run:
```
/validate-config
```
All checks must pass before proceeding.

---

## Step 5 — Scaffold the full project skeleton

Run in sequence — do not implement anything yet, only create file skeletons:

```
/scaffold preprocessing
/scaffold dataset
/scaffold augmentation
/scaffold encoder
/scaffold heads
/scaffold seismic_net
/scaffold loss
/scaffold metrics
/scaffold utils
/scaffold train
/scaffold evaluate
/scaffold export
```

Each `/scaffold` command creates the file with:
- All function/class signatures from the governing doc
- Docstrings referencing the exact doc section
- `TODO: implement` stubs — no logic yet
- Type annotations matching the tensor shapes in the doc

After all scaffolds are complete, verify structure:
```bash
find seismic/ training/ tests/ -name "*.py" | sort
```

---

## Step 6 — First Graphify build

Now that skeleton files exist, build the first graph:

```
/regraph
```

This runs `/graphify .` on the entire project. After the rebuild:

1. Open `graphify-out/graph.html` in a browser — explore the module graph visually
2. Read `graphify-out/GRAPH_REPORT.md` — check "surprising connections" section
3. Run a few orientation queries:
   ```
   graphify query "show me all modules in seismic/"
   graphify query "what does seismic_net.py import"
   graphify query "which files are in the model/ subdirectory"
   ```
4. Log the rebuild in `.opencode/instructions.md` Graphify State block

At this stage the graph will be sparse (only signatures). That is expected — it becomes useful after Step 7.

---

## Step 7 — Begin implementation in order

Follow the 19-step implementation order in `.opencode/instructions.md`.

The agent must:
- Run `/regraph check` at the start of any session after code has been written
- Run `/regraph` at each implementation milestone (after each major module group)
- Run `/test [module]` after implementing each module — never move forward with failing tests
- Update `.opencode/instructions.md` at the end of every session with `/sync-state`

Implementation milestones where `/regraph` is mandatory:

| Milestone | What triggers the rebuild |
|---|---|
| After Step 4 (config files generated) | First structural snapshot before code |
| After Step 11 (all model files done) | Graph now shows full call chain: seismic_net → encoder → heads |
| After Step 14 (loss + metrics + utils done) | Graph shows loss routing and checkpoint flow |
| After Step 17 (training pipeline done) | Graph shows full end-to-end train → checkpoint → export path |

---

## Step 8 — GPU server setup checklist

Before the first training run, complete these steps on the SSH GPU server.
These are not automated — the human must verify each one manually.

```
[ ] STEAD HDF5 files downloaded to data/raw/ on GPU server
    Source: https://github.com/smousavi05/STEAD
    Files: chunk1.hdf5, chunk2.hdf5, metadata.csv

[ ] data/splits/ generated by scripts/generate_splits.py
    Verify: no event ID appears in more than one split file
    Verify: split is by event ID, not by sample index

[ ] Conda environment created from environment.yml on GPU server
    Command: conda env create -f environment.yml
    Verify: python -c "import torch; print(torch.cuda.is_available())" returns True

[ ] MLflow server running in tmux on GPU server
    Command: mlflow server --backend-store-uri sqlite:///mlflow/mlflow.db \
                           --default-artifact-root ./mlflow/artifacts \
                           --host 127.0.0.1 --port 5000

[ ] SSH tunnel active from local machine
    Command: ssh -N -L 5000:localhost:5000 user@gpu-server
    Verify: curl http://localhost:5000 returns MLflow HTML

[ ] GPU visible
    Command (on GPU server): nvidia-smi

[ ] /checklist phase1 passes all 10 checks
```

Do not launch training until every item above is checked.

---

## Step 9 — .gitignore confirmation

Confirm these paths are ignored before the first commit:

```bash
git check-ignore -v graphify-out/         # must be ignored
git check-ignore -v data/raw/             # must be ignored
git check-ignore -v checkpoints/          # must be ignored
git check-ignore -v exported_models/      # must be ignored
git check-ignore -v mlflow/mlflow.db      # must be ignored
```

These must NOT be ignored (they should be committed):
```bash
git check-ignore -v data/splits/train.txt    # should NOT be ignored
git check-ignore -v configs/                 # should NOT be ignored
git check-ignore -v docs/                    # should NOT be ignored
git check-ignore -v AGENTS.md               # should NOT be ignored
git check-ignore -v .opencode/instructions.md  # should NOT be ignored
```

---

## Summary — Full file inventory after setup

```
Committed to git:
  AGENTS.md, .gitignore, README.md
  .opencode/instructions.md, commands.md, plugins/graphify.js
  docs/00–04 .md files
  configs/*.json (all three)
  seismic/**/*.py (all skeletons and implementations)
  training/*.py
  tests/*.py
  data/splits/train.txt, val.txt, test.txt
  environment.yml, requirements_inference.txt

NOT committed (in .gitignore):
  graphify-out/          (rebuilt on demand)
  data/raw/              (large HDF5, transfer via scp)
  data/processed/        (derived, rebuildable)
  checkpoints/           (large .pt files, stay on GPU server)
  exported_models/       (large .pt files, transfer separately)
  mlflow/mlflow.db       (lives on GPU server)
  mlflow/artifacts/      (lives on GPU server)
```
