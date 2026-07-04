# GPU Server Runbook — SeismicNet Training

Reproducible SSH commands for every training run. All paths assume the GPU server
home directory is `~/seismic_ml/` and the conda env is named `seismic_risk`.

---

## 1. rsync Project to GPU Server

Run from the **local** machine. Excludes everything in `.gitignore` (raw data,
checkpoints, exported models, Python artifacts).

```bash
# Set these variables
GPU_HOST="user@gpu-server-address"
REMOTE_DIR="~/seismic_ml"

# Dry run first — check what will transfer
rsync -avzn --delete \
  --exclude='.git/' \
  --exclude='graphify-out/' \
  --exclude='data/raw/' \
  --exclude='data/processed/' \
  --exclude='checkpoints/' \
  --exclude='exported_models/' \
  --exclude='mlflow/mlflow.db' \
  --exclude='mlflow/artifacts/' \
  --exclude='__pycache__/' \
  --exclude='*.pyc' \
  --exclude='.pytest_cache/' \
  --exclude='.env' \
  --exclude='.venv' \
  --exclude='*.egg-info/' \
  --exclude='dist/' \
  --exclude='build/' \
  --exclude='.idea/' \
  --exclude='.vscode/' \
  --exclude='*.swp' \
  --exclude='*.swo' \
  --exclude='scratch/' \
  --exclude='*.log' \
  --exclude='*.tmp' \
  --exclude='seismic_risk.egg-info/' \
  ./ "${GPU_HOST}:${REMOTE_DIR}"

# Actual transfer (remove -n flag)
rsync -avz --delete \
  --exclude='.git/' \
  --exclude='graphify-out/' \
  --exclude='data/raw/' \
  --exclude='data/processed/' \
  --exclude='checkpoints/' \
  --exclude='exported_models/' \
  --exclude='mlflow/mlflow.db' \
  --exclude='mlflow/artifacts/' \
  --exclude='__pycache__/' \
  --exclude='*.pyc' \
  --exclude='.pytest_cache/' \
  --exclude='.env' \
  --exclude='.venv' \
  --exclude='*.egg-info/' \
  --exclude='dist/' \
  --exclude='build/' \
  --exclude='.idea/' \
  --exclude='.vscode/' \
  --exclude='*.swp' \
  --exclude='*.swo' \
  --exclude='scratch/' \
  --exclude='*.log' \
  --exclude='*.tmp' \
  --exclude='seismic_risk.egg-info/' \
  ./ "${GPU_HOST}:${REMOTE_DIR}"
```

**Note**: The STEAD HDF5 files (`data/raw/*.hdf5` and `data/raw/metadata.csv`)
must already exist on the GPU server. Do NOT rsync them — they are too large.
Download them separately per the STEAD instructions.

---

## 2. First-Time Setup on GPU Server

SSH into the GPU server and run these once.

### 2a. Verify GPU

```bash
nvidia-smi
# Should show at least one GPU with CUDA 12.x driver
```

### 2b. Create Conda Environment

```bash
cd ~/seismic_ml

# Create env from environment.yml
conda env create -f environment.yml

# Activate
conda activate seismic_risk

# Verify PyTorch sees the GPU
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"
```

### 2c. Generate Data Splits

Splits must be generated on the GPU server where the STEAD HDF5 files live.

```bash
cd ~/seismic_ml
conda activate seismic_risk

python scripts/generate_splits.py \
  --metadata-csv data/raw/metadata.csv \
  --output-dir data/splits \
  --seed 42

# Verify splits exist
wc -l data/splits/*.txt
# Expected: ~840k train, ~180k val, ~180k test (for full STEAD)
```

### 2d. Run Local Tests (on GPU server)

```bash
cd ~/seismic_ml
conda activate seismic_risk

python -m pytest tests/test_preprocessing.py tests/test_model_shapes.py tests/test_dataset.py -v
```

### 2e. Create MLflow Directory

```bash
mkdir -p ~/seismic_ml/mlflow
```

---

## 3. Launch MLflow Server (tmux)

Start MLflow in a persistent tmux session. This stays alive across SSH disconnects.

```bash
# Create a new tmux session
tmux new-session -d -s mlflow

# Send commands to the tmux session
tmux send-keys -t mlflow "cd ~/seismic_ml && conda activate seismic_risk && mlflow server --backend-store-uri sqlite:///mlflow/mlflow.db --default-artifact-root ./mlflow/artifacts --host 127.0.0.1 --port 5000" Enter

# Verify it's running
sleep 3
curl -s http://localhost:5000 | head -c 100
# Should return HTML content

# Reattach to check logs if needed
tmux attach -t mlflow

# Detach from tmux: Ctrl+B then D
```

### SSH Tunnel (from local machine)

```bash
# In a separate local terminal
ssh -N -L 5000:localhost:5000 user@gpu-server-address

# Access MLflow UI at http://localhost:5000
# Use autossh for persistent tunnels:
autossh -M 0 -f -N -L 5000:localhost:5000 user@gpu-server-address
```

---

## 4. Run Phase 1 Training

### Pre-Flight Checklist (run every time before training)

```bash
cd ~/seismic_ml
conda activate seismic_risk

# 1. GPU visible
nvidia-smi

# 2. MLflow running
curl -s http://localhost:5000 > /dev/null && echo "MLflow OK" || echo "MLflow DOWN"

# 3. Split files exist
ls -la data/splits/train.txt data/splits/val.txt data/splits/test.txt

# 4. STEAD data present
ls data/raw/*.hdf5 | wc -l
# Should be >0

# 5. Configs valid
python -c "
import json
for name in ['preprocessing_config', 'model_config', 'training_config']:
    c = json.load(open(f'configs/{name}.json'))
    assert 'schema_version' in c, f'{name} missing schema_version'
    print(f'{name}: {c[\"schema_version\"]}')
print('All configs OK')
"
```

### Launch Phase 1

```bash
cd ~/seismic_ml
conda activate seismic_risk

# Create output directory
mkdir -p checkpoints

# Phase 1: Tasks A+B (detection + phase picking)
python training/train.py \
  --model-config configs/model_config.json \
  --training-config configs/training_config.json \
  --preprocessing-config configs/preprocessing_config.json \
  --data-dir data/ \
  --output-dir checkpoints/ \
  --mlflow-uri http://localhost:5000 \
  --experiment-name seismic_risk \
  --run-name phase1_resnet_transformer_baseline \
  --phase 1
```

### Monitor Training

```bash
# Check training progress in MLflow UI: http://localhost:5000
# Or watch the tmux output:
tmux attach -t mlflow

# Check GPU utilization
watch -n 2 nvidia-smi
```

---

## 5. Phase Transition Protocol

After Phase 1 completes, check the best `val/detection_f1` before proceeding.

### Phase 1 → Phase 2

Gate: `detection_f1 > 0.90` on val set.

```bash
# Find the best Phase 1 checkpoint
ls -lt checkpoints/run_*/epoch_*.pt | head -5

# Launch Phase 2 with --resume pointing to the best checkpoint
python training/train.py \
  --model-config configs/model_config.json \
  --training-config configs/training_config.json \
  --preprocessing-config configs/preprocessing_config.json \
  --data-dir data/ \
  --output-dir checkpoints/ \
  --mlflow-uri http://localhost:5000 \
  --experiment-name seismic_risk \
  --run-name phase2_add_magnitude \
  --phase 2 \
  --resume checkpoints/run_<PHASE1_RUN_ID>/epoch_<BEST>.pt
```

### Phase 2 → Phase 3

Gate: `magnitude_rmse < 0.5` on val set.

```bash
python training/train.py \
  --model-config configs/model_config.json \
  --training-config configs/training_config.json \
  --preprocessing-config configs/preprocessing_config.json \
  --data-dir data/ \
  --output-dir checkpoints/ \
  --mlflow-uri http://localhost:5000 \
  --experiment-name seismic_risk \
  --run-name phase3_add_risk \
  --phase 3 \
  --resume checkpoints/run_<PHASE2_RUN_ID>/epoch_<BEST>.pt
```

---

## 6. Evaluate on Test Split

```bash
python training/evaluate.py \
  --checkpoint checkpoints/run_<RUN_ID>/epoch_<BEST>.pt \
  --preprocessing-config configs/preprocessing_config.json \
  --data-dir data/ \
  --output-dir eval_results/ \
  --mlflow-uri http://localhost:5000
```

---

## 7. Export for Inference

```bash
python training/export.py \
  --checkpoint checkpoints/run_<RUN_ID>/epoch_<BEST>.pt \
  --preprocessing-config configs/preprocessing_config.json \
  --output-dir exported_models/seismic_v1/ \
  --benchmark
```

Produces:
```
exported_models/seismic_v1/
├── model_scripted.pt
├── preprocessing_config.json
├── model_config.json
├── label_schema.json
└── benchmark_results.json
```

---

## 8. Retrieve Artifacts (from local machine)

```bash
# Download checkpoints
rsync -avz user@gpu-server-address:~/seismic_ml/checkpoints/ ./checkpoints/

# Download exported model
rsync -avz user@gpu-server-address:~/seismic_ml/exported_models/ ./exported_models/

# Download eval results
rsync -avz user@gpu-server-address:~/seismic_ml/eval_results/ ./eval_results/
```

---

## Troubleshooting

### Out of Memory (OOM)

Reduce `batch_size` in `configs/training_config.json`. Current: 256. Try 128 or 64.

### CUDA Not Available

```bash
conda activate seismic_risk
python -c "import torch; print(torch.version.cuda)"
# Should print "12.1" or similar
# If None, reinstall: conda install pytorch pytorch-cuda=12.1 -c pytorch -c nvidia
```

### MLflow Connection Refused

```bash
# Check if MLflow is running
tmux list-sessions
# If not, relaunch per Section 3

# Check port
ss -tlnp | grep 5000
```

### Preprocessing Config Version Mismatch

This error means the checkpoint was trained with a different `preprocessing_config.json`.
**Do not bypass this error.** Either:
1. Use the correct config version, or
2. Retrain from scratch with the current config

### Import Errors on GPU Server

```bash
cd ~/seismic_ml
conda activate seismic_risk
pip install -e .
# Or: PYTHONPATH=~/seismic_ml python training/train.py ...
```
