# .opencode/commands.md — Custom Slash Commands

These commands are available in every session. The agent executes them exactly as described.

---

## /scaffold [module]

Creates a file skeleton for a module using the spec in the relevant `docs/` file.
Produces: function signatures, class definitions, docstrings referencing the doc section, and `TODO: implement` stubs.
Does NOT write implementation logic — that is done with `/implement`.

Examples:
```
/scaffold preprocessing     → creates seismic/preprocessing.py skeleton
/scaffold encoder           → creates seismic/model/encoder.py skeleton
/scaffold train             → creates training/train.py skeleton
```

Before scaffolding, the agent must:
1. Read the governing doc for that module (see AGENTS.md docs/ index)
2. Confirm the file does not already exist (ask before overwriting)

---

## /implement [function] in [file]

Implements a single function or class following the exact spec in the governing doc.
The agent reads the relevant doc section before writing any code.
After implementing, the agent runs the relevant test if one exists.

Examples:
```
/implement preprocess_waveform in seismic/preprocessing.py
/implement ResBlock in seismic/model/encoder.py
/implement SeismicLoss in seismic/loss.py
/implement save_checkpoint in seismic/utils.py
```

The agent must not implement beyond the specified function in one command.
If the implementation requires a helper not in the spec, ask before adding it.

---

## /test [module]

Runs pytest for a specific test file and reports results clearly.

Examples:
```
/test preprocessing     → runs tests/test_preprocessing.py, shows pass/fail per test
/test model             → runs tests/test_model_shapes.py
/test dataset           → runs tests/test_dataset.py
/test all               → runs full test suite, reports summary
```

On failure, the agent:
1. Shows the exact assertion error
2. Identifies which doc section the failing test enforces
3. Proposes a fix — does not auto-apply without confirmation

---

## /validate-config

Checks all three config JSON files for correctness.

Performs:
- Schema version field present in all three files
- All required keys present (agent derives required keys from doc schemas)
- preprocessing_config.json and model_config.json schema_version strings match
- No float values where int is required and vice versa
- bandpass_low_hz < bandpass_high_hz
- p_arrival_sample_index < window_length_samples
- All phase loss weights are non-negative floats

Reports: PASS or list of violations with the config key and expected value.

---

## /checklist [phase]

Runs the pre-training checklist from `docs/04_training_and_export.md` before a GPU training run.

```
/checklist phase1
/checklist phase2
/checklist phase3
```

Checks (in order):
1. tests/test_preprocessing.py — all pass
2. tests/test_model_shapes.py — all pass
3. tests/test_dataset.py — no split leakage
4. MLflow server reachable at localhost:5000 (curl check)
5. data/raw/ HDF5 files present on GPU server (SSH ls check)
6. data/splits/train.txt, val.txt, test.txt present
7. GPU visible (nvidia-smi via SSH)
8. preprocessing_config.json schema_version matches train.py expectation
9. run-name argument is set and descriptive (not 'test' or 'run1')
10. For phase2/phase3: phase gate metric from previous phase is met

Reports: numbered list, PASS/FAIL per item. Blocks training suggestion if any FAIL.

---

## /phase-gate [phase_number]

Checks if the metric threshold for transitioning to the next phase is satisfied.
Reads metrics from the latest MLflow val log for the current phase run.

```
/phase-gate 1   → checks: val/detection_f1 > 0.90
/phase-gate 2   → checks: val/magnitude_rmse < 0.50
```

If gate is NOT met:
- Shows current metric value vs required threshold
- Suggests: extend current phase by N epochs OR investigate specific failure mode
- Does NOT proceed to next phase automatically

If gate IS MET:
- Confirms best checkpoint path to use as Phase N+1 starting point
- Generates the Phase N+1 train.py command with correct --resume and --phase arguments
- Reminds agent to create a new MLflow run and tag with parent_run_id

---

## /resume-training [checkpoint_path]

Generates the correct `train.py` CLI command to resume training from a checkpoint.

Before generating the command:
1. Loads the checkpoint and reads `preprocessing_config_version`
2. Compares against current `configs/preprocessing_config.json:schema_version`
3. If mismatch: raises error, does not generate resume command
4. If match: generates the full CLI command with all required arguments

Output: the exact command to run on the GPU server, ready to paste into the terminal.

---

## /export [checkpoint_path]

Runs `training/export.py` against a checkpoint and reports the result.

Steps executed:
1. Validates preprocessing config version match
2. Tunes detection threshold from val predictions
3. Applies selective quantization (ResNet only)
4. Applies TorchScript
5. Runs CPU benchmark (100 forward passes, batch=1)
6. Writes artifact bundle to exported_models/

Reports:
- Mean / P95 latency (ms) vs targets (<150ms mean, <250ms P95)
- Model file size (MB) vs target (<50MB)
- Smoke test result (pass/fail per assertion)
- Full artifact bundle path

---

## /status

Reads `.opencode/instructions.md` and prints a structured summary:

```
=== SeismicNet Project Status ===
Phase:          [current phase]
Active tasks:   [task list]
Best val metrics:
  detection_f1:       X.XXX  [GATE: need >0.90 for phase2]
  magnitude_rmse:     X.XXX  [GATE: need <0.50 for phase3]
Open issues:    N items (list them)
Next action:    [top item from implementation progress]
=================================
```

---

## /regraph [scope]

Rebuilds the Graphify knowledge graph. Must be run at the milestones defined in `AGENTS.md`.

```
/regraph                # full project rebuild — runs /graphify . on entire repo
/regraph seismic        # scope to seismic/ + docs/ only (faster, mid-milestone)
/regraph docs           # rebuild from docs/ only (after updating a guideline .md)
/regraph check          # staleness check only — no rebuild
                        # compares mtime of graph.json vs all .py and .md files
                        # reports: FRESH or STALE with list of newer files
```

### What the agent does after every rebuild

1. Reads `graphify-out/GRAPH_REPORT.md` — specifically the "surprising connections" and "suggested questions" sections
2. Checks for any connection that contradicts the docs:
   - Signal processing logic outside `seismic/preprocessing.py` → violation of Invariant 1
   - Checkpoint save without config copy → violation of Invariant 2
   - Phase picking head reading from pooled features → architecture violation
3. Reports any violations to the human before continuing
4. Logs the rebuild in `.opencode/instructions.md` with timestamp and scope

### When NOT to rebuild

Do not rebuild after every single function implementation — it is slow and wasteful.
Rebuild at milestones only (see AGENTS.md Graphify section for the full list).
Use `/regraph check` to decide if a rebuild is needed before running the full rebuild.

### Querying the graph without rebuilding

```
graphify query "what does preprocessing.py export"
graphify query "which modules call SeismicNet.forward"
graphify query "what is the call path from train.py to loss.py"
graphify query "which files define ResBlock"
graphify query "what imports augmentation"
graphify query "show me all modules in the seismic/ package"
```

Use queries for targeted navigation. Use `GRAPH_REPORT.md` only for broad orientation.
Never grep raw source when the graph is fresh and the question is structural.

---

## /sync-state

Updates `.opencode/instructions.md` with the current session's changes.

The agent must call this at the end of every session. It:
1. Updates the implementation progress checklist (marks completed items)
2. Appends a new line to the Session Log with date and summary
3. Updates "Current Status" block
4. Updates "Known Issues" (marks resolved items, adds new ones discovered)
5. Does NOT change the Architectural Decisions section unless a deviation occurred

After updating, the agent reminds the human to review and commit the file.