# FedRAPT: Federated Representation-Aligned Prototypical Contrastive Learning

Personalized federated learning for wearable sensor-based Human Activity Recognition (HAR).
FedRAPT addresses statistical heterogeneity (non-IID) across clients through **Cross-Client Representation Alignment (CCRA)**: the server maintains global class prototypes aggregated from all participating clients, and each client aligns its local feature representations toward these prototypes via InfoNCE contrastive loss — without sharing raw data.

---

## Method Overview

```
Client k                                     Server
--------                                     ------
 Local data (non-IID)                    Global prototypes {mu_c}
      |                                           ^
      v                                           |  EMA update:
 LSTM Encoder ──> Projection Head ──> z_{k,c}    |  mu_c <- beta*mu_c + (1-beta)*mean(z_{k,c})
      |                                           |
      |  L_total = L_CE + lambda * L_InfoNCE      |
      |  (align z toward mu_c from server)        |
      v
 Personalized FC Classifier (local only, NOT aggregated)
```

**Prototype EMA update:**
```
mu_c^{t+1} = beta * mu_c^t + (1 - beta) * (1/K_c * sum_k z_{k,c})
```
where beta=0.9 (default), z_{k,c} = mean embedding for class c on client k.

---

## Project Structure

```
FedRAPT/
├── main.py                     # Entry point: federated training loop
│                               # Note: configs/*.yaml are reference docs, not loaded at runtime.
├── args.py                     # Argument parser and defaults
├── server.py                   # Federated server: FedAvg + EMA prototype update
├── client.py                   # Client: local training (CE + InfoNCE) + evaluation
├── model.py                    # LSTM_FedRAPT: encoder + projection head + FC classifier
├── contrastive.py              # InfoNCE loss, collect_class_embeddings, update_prototypes
├── data_process.py             # .npz -> DataLoader (70/15/15 train/val/test split)
├── preprocess.py               # Raw datasets -> per-client .npz files
├── utils.py                    # Seed fixing, arg logging, checkpoint save/load
├── run.sh                      # Single-dataset runner (recommended entry point)
├── run_prototype_experiments.sh # EMA strategy ablation (direct/simple_avg/cumulative)
├── configs/                    # Per-dataset default hyperparameters
│   ├── wisdm.yaml
│   ├── ucihar.yaml
│   ├── ucihar_alpha01.yaml     # UCI-HAR Dirichlet alpha=0.1
│   ├── ucihar_alpha05.yaml     # UCI-HAR Dirichlet alpha=0.5
│   └── motionsense.yaml
├── scripts/
│   ├── run_all.sh              # Reproduce all 5 datasets x 5 runs
│   └── summarize_results.py   # Print results table from saved CSVs
├── result/                     # Created at runtime — metrics CSVs per dataset
├── checkpoints/                # Created at runtime — saved model weights
└── requirements.txt
```

---

## Requirements

```bash
# PyTorch with CUDA 11.8
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 \
    --index-url https://download.pytorch.org/whl/cu118

pip install -r requirements.txt
```

Tested on: Python 3.10, PyTorch 2.4.0, CUDA 11.8.

---

## Datasets

### Download

| Dataset | Clients | Classes | Input | Download |
|---------|---------|---------|-------|----------|
| WISDM v1.1 | 36 | 6 | Accelerometer (x,y,z) | https://www.cis.fordham.edu/wisdm/dataset.php |
| UCI-HAR | 30 | 6 | Total acceleration (x,y,z) | https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones |
| MotionSense | 24 | 6 | Total acceleration (x,y,z) | https://github.com/mmalekzadeh/motion-sense |

| WISDM | raw 20 Hz upsampled to 50 Hz via linear interpolation; sliding window (128 steps, stride 64) |
| MotionSense | sliding window (128 steps, stride 64) |
| UCI-HAR | pre-segmented 128-step windows provided by dataset; no sliding window applied |

### Data Split Strategy

| Dataset Setting | Split Type | Description |
|---|---|---|
| WISDM natural | Natural | Each of 36 users is one client |
| UCI-HAR natural | Natural | Each of 30 subjects is one client |
| UCI-HAR α=0.1 | Dirichlet | Data pooled and re-split with Dirichlet(α=0.1); extreme non-IID (Max/Min sample ratio ~234×, avg 3.4 classes/client) |
| UCI-HAR α=0.5 | Dirichlet | Dirichlet(α=0.5); moderate non-IID (Max/Min ~6.5×, avg 5.6 classes/client) |
| MotionSense natural | Natural | Each of 24 subjects is one client |

**What does α mean?** The Dirichlet concentration parameter α controls heterogeneity. Lower α → each client holds fewer classes → more non-IID. α→∞ approaches IID.

**Train/Val/Test split per client:** 70% / 15% / 15%, randomly shuffled within each client using a deterministic SHA-256 seed derived from the client name (reproducible across runs).

### Preprocess

```bash
# Step 1 — WISDM (provide path to WISDM_ar_v1.1_raw.txt)
python preprocess.py --dataset wisdm \
    --data_dir /path/to/WISDM_ar_v1.1_raw.txt \
    --output_dir ./data/wisdm_npz

# Step 2 — UCI-HAR natural split
python preprocess.py --dataset ucihar \
    --data_dir /path/to/UCI_HAR_Dataset \
    --output_dir ./data/ucihar_npz

# Step 3 — UCI-HAR Dirichlet alpha=0.1 (non-IID, extreme)
python preprocess.py --dataset ucihar \
    --data_dir /path/to/UCI_HAR_Dataset \
    --output_dir ./data/ucihar_npz_alpha01 \
    --dirichlet --alpha 0.1 --seed 42

# Step 4 — UCI-HAR Dirichlet alpha=0.5 (non-IID, moderate)
python preprocess.py --dataset ucihar \
    --data_dir /path/to/UCI_HAR_Dataset \
    --output_dir ./data/ucihar_npz_alpha05 \
    --dirichlet --alpha 0.5 --seed 42

# Step 5 — MotionSense (provide path to A_DeviceMotion_data/)
python preprocess.py --dataset motionsense \
    --data_dir /path/to/A_DeviceMotion_data \
    --output_dir ./data/motionsense_npz
```

Each output file is a `.npz` with keys `X` (N, 128, 3) and `y` (N,) int64.

---

## Training

Use `run.sh` to select a dataset and run:

```bash
# Single run
bash run.sh wisdm
bash run.sh ucihar
bash run.sh ucihar_01        # UCI-HAR Dirichlet alpha=0.1
bash run.sh ucihar_05        # UCI-HAR Dirichlet alpha=0.5
bash run.sh motionsense

# 5 independent runs (for mean ± std reporting)
bash run.sh ucihar_01 --runs 5

# Custom settings
bash run.sh ucihar_01 --runs 5 --global_rounds 100 --local_epochs 3 --seed 42

# Custom data directory
bash run.sh wisdm --data_dir /my/custom/path/wisdm_npz
```

**Option names:**
- `--runs N` : number of independent repetitions (each saves a separate CSV)
- `--global_rounds N` : number of federated communication rounds (default: 100)
- `--local_epochs N` : local training epochs per client per round (default: 3)
- `--seed N` : random seed; if omitted, each run uses a different random seed

Or run directly:

```bash
python main.py \
    --dataset ucihar \
    --data_dir ./data/ucihar_npz_alpha01 \
    --r 100 --E 3 --B 50 \
    --clients_per_round 10 \
    --seed 42 \
    --metrics_csv result/ucihar_alpha01/fedrapt_metrics_iter1.csv
```

### Key Hyperparameters

| Argument | Default | Description |
|---|---|---|
| `--r` | 100 | Communication rounds (global_rounds) |
| `--E` | 3 | Local training epochs per round (local_epochs) |
| `--B` | 50 | Batch size |
| `--clients_per_round` | 12/10/8 | Clients sampled each round (WISDM 12, UCI-HAR 10, MotionSense 8) |
| `--lr` | 0.01 | Learning rate (SGD) |
| `--contrastive_weight` | 0.5 | λ: InfoNCE loss weight |
| `--temperature` | 0.07 | τ: InfoNCE temperature |
| `--prototype_beta` | 0.9 | β: EMA momentum for prototype update |
| `--projection_dim` | 64 | Projection head output dimension |
| `--hidden_size` | 64 | LSTM hidden dimension |
| `--proto_update_strategy` | `ema` | Prototype update: `ema`, `direct`, `simple_avg`, `cumulative` |
| `--seed` | None | Random seed (unset = random) |

---

## Reproducing Main Results

```bash
# Set data paths (or edit scripts/run_all.sh)
export DATA_WISDM=./data/wisdm_npz
export DATA_UCIHAR=./data/ucihar_npz
export DATA_UCIHAR_01=./data/ucihar_npz_alpha01
export DATA_UCIHAR_05=./data/ucihar_npz_alpha05
export DATA_MOTIONSENSE=./data/motionsense_npz

bash scripts/run_all.sh
```

Print results table:

```bash
python scripts/summarize_results.py
```

Expected results (mean ± std over 5 independent runs, personalized accuracy at round 100):

| Dataset | Accuracy (%) | F1 Score (%) |
|---|---|---|
| WISDM | 96.30 ± 0.47 | 96.23 ± 0.50 |
| UCI-HAR (natural) | 96.13 ± 0.37 | 96.06 ± 0.38 |
| UCI-HAR (α=0.1) | 96.09 ± 0.55 | 95.99 ± 0.61 |
| UCI-HAR (α=0.5) | 94.42 ± 4.78 | — |
| MotionSense | 94.00 ± 0.71 | 93.74 ± 0.80 |

---

## Output Files

```
result/
└── ucihar_alpha01/
    ├── fedrapt_metrics_iter1.csv       # Per-client, per-round metrics
    ├── fedrapt_metrics_iter1_comm.csv  # Communication cost per round
    └── fedrapt_metrics_iter1_config.json  # Full args + git hash + date

checkpoints/
└── ucihar_alpha01/
    └── run1/
        ├── global_final.pth              # Global LSTM encoder weights
        └── user_1_windows_final.pth      # Per-client personalized model
```

### Metrics CSV columns

| Column | Description |
|---|---|
| `round` | Communication round number |
| `client` | Client identifier (e.g. `user_1_windows`) |
| `accuracy` | Personalized test accuracy |
| `f1` | Weighted F1 score |
| `loss` | Test cross-entropy loss |
| `forgetting_rate` | Catastrophic forgetting metric |
| `scope` | `personal` (tracked each round) or `personal_final` (last round only) |

---

## Prototype Update Strategy Ablation

To compare EMA vs. alternative prototype update strategies:

```bash
bash run_prototype_experiments.sh
```

Runs `direct`, `simple_avg`, `cumulative` strategies (5 runs each) on UCI-HAR α=0.1.
EMA results come from `result/ucihar_alpha01/`.

Results from the paper (UCI-HAR α=0.1):

| Strategy | Accuracy (%) | F1 (%) |
|---|---|---|
| Direct | 87.02 ± 1.13 | 84.13 ± 1.22 |
| Simple Average | 86.89 ± 2.65 | 83.99 ± 2.97 |
| Cumulative | 86.71 ± 2.66 | 83.88 ± 3.45 |
| **EMA β=0.9 (Proposed)** | **96.09 ± 0.55** | **95.99 ± 0.61** |
