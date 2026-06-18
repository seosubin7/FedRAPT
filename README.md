# FedRAPT: Federated Representation-Aligned Prototypical Contrastive Learning

FedRAPT is a personalized federated learning framework for Human Activity Recognition under heterogeneous client data distributions. It jointly performs cross-client class-level alignment, local sample discrimination, and client-specific personalization without sharing raw data.

<p align="center">
  <img src="figures/framework.png" width="850" alt="FedRAPT Framework Overview"/>
</p>

---

## Motivation

In non-IID federated learning, samples belonging to the same class may be represented differently across clients. This inconsistency can cause representation drift, unstable aggregation, and performance degradation.

FedRAPT addresses this problem by using global class prototypes as common alignment anchors while preserving sample-level discriminability through local contrastive learning.

---

## Method

FedRAPT separates each client model into shared and personalized components.

| Component       | Role                                                               | Aggregation |
| --------------- | ------------------------------------------------------------------ | ----------- |
| LSTM encoder    | Extracts a 64-dimensional representation from each sensor window   | FedAvg      |
| Projection head | Maps representations into a normalized contrastive embedding space | FedAvg      |
| FC classifier   | Learns a client-specific decision boundary                         | Local only  |

The encoder and projection head are shared across clients. The classifier remains local and is never transmitted to the server.

### Cross-Client Representation Alignment

<p align="center">
  <img src="figures/ccra_module.png" width="750" alt="CCRA Module"/>
</p>

For each anchor embedding, CCRA constructs:

| Set          | Elements                                                                        |
| ------------ | ------------------------------------------------------------------------------- |
| Positive set | Global prototype of the same class and local samples from the same class        |
| Negative set | Global prototypes of different classes and local samples from different classes |

This design combines class-level alignment across clients with sample-level discrimination within each client.

### Training Objective

```math
\mathcal{L}_{\mathrm{total}}
=
\mathcal{L}_{\mathrm{CE}}
+
\lambda \mathcal{L}_{\mathrm{CL}}
```

Here, `L_CE` is the local classification loss and `L_CL` is the CCRA-based contrastive loss.

### Global Prototype Update

Each selected client computes a class-wise mean embedding and sends it to the server. The server updates each global class prototype using exponential moving average:

```math
\mu_c^{t+1}
=
\beta \mu_c^t
+
(1-\beta)
\left(
\frac{1}{K_c}
\sum_k z_{k,c}
\right)
```

Here, `β = 0.9`, `z_{k,c}` is the mean embedding of class `c` on client `k`, and `K_c` is the number of participating clients containing class `c`.

---

## Training Workflow

1. The server selects participating clients.
2. The server sends the shared model parameters and global class prototypes.
3. Each client trains its encoder, projection head, and personalized classifier.
4. Each client uploads the updated shared parameters and class-wise mean embeddings.
5. The server aggregates shared parameters using FedAvg.
6. The server updates global prototypes using EMA.
7. Local classifiers and raw data remain on each client.

---

## Key Contributions

* **Cross-client class-level alignment:** global prototypes provide consistent class-wise reference points across clients.
* **Sample-level discrimination:** local positive and negative samples preserve fine-grained discriminative relationships.
* **Personalized classification:** local classifiers adapt to client-specific data distributions.
* **Stable prototype management:** EMA reduces abrupt prototype fluctuations under partial participation and non-IID data.

---

## Project Structure

```
FedRAPT/
├── main.py                          # Entry point: federated training loop
├── args.py                          # Argument parser and defaults
├── utils.py                         # Seed fixing, arg logging, checkpoint save/load
├── run.sh                           # Single-dataset runner (recommended entry point)
├── run_prototype_experiments.sh     # Prototype strategy ablation
│
├── federation/                      # Federated learning core
│   ├── server.py                    # FedAvg aggregation + EMA prototype update
│   └── client.py                    # Local training (CE + InfoNCE) + evaluation
│
├── models/                          # Model definitions
│   ├── lstm.py                      # LSTM encoder + projection head + FC classifier
│   └── contrastive.py               # InfoNCE loss, collect_class_embeddings, update_prototypes
│
├── data/                            # Data pipeline
│   ├── loader.py                    # .npz -> DataLoader (70/15/15 train/val/test split)
│   └── preprocess.py                # Raw datasets -> per-client .npz files
│
├── configs/                         # Per-dataset reference hyperparameters (not loaded at runtime)
│   ├── wisdm.yaml
│   ├── ucihar.yaml
│   ├── ucihar_alpha01.yaml
│   ├── ucihar_alpha05.yaml
│   └── motionsense.yaml
├── scripts/
│   ├── run_all.sh                   # Reproduce all 5 datasets x 5 runs
│   └── summarize_results.py         # Print results table from saved CSVs
├── figures/                         # Paper figures
├── result/                          # Created at runtime — metrics CSVs
├── checkpoints/                     # Created at runtime — saved model weights
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

| Dataset | Preprocessing |
|---|---|
| WISDM | Raw 20 Hz signals are upsampled to 50 Hz via linear interpolation, followed by sliding-window segmentation with 128 steps and stride 64 |
| MotionSense | Sliding-window segmentation with 128 steps and stride 64 |
| UCI-HAR | Pre-segmented 128-step windows provided by the dataset are used directly |

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
python data/preprocess.py --dataset wisdm \
    --data_dir /path/to/WISDM_ar_v1.1_raw.txt \
    --output_dir ./data/wisdm_npz

# Step 2 — UCI-HAR natural split
python data/preprocess.py --dataset ucihar \
    --data_dir /path/to/UCI_HAR_Dataset \
    --output_dir ./data/ucihar_npz

# Step 3 — UCI-HAR Dirichlet alpha=0.1 (non-IID, extreme)
python data/preprocess.py --dataset ucihar \
    --data_dir /path/to/UCI_HAR_Dataset \
    --output_dir ./data/ucihar_npz_alpha01 \
    --dirichlet --alpha 0.1 --seed 42

# Step 4 — UCI-HAR Dirichlet alpha=0.5 (non-IID, moderate)
python data/preprocess.py --dataset ucihar \
    --data_dir /path/to/UCI_HAR_Dataset \
    --output_dir ./data/ucihar_npz_alpha05 \
    --dirichlet --alpha 0.5 --seed 42

# Step 5 — MotionSense (provide path to A_DeviceMotion_data/)
python data/preprocess.py --dataset motionsense \
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
- `--seed N` : random seed; if omitted, the random seed is not fixed, resulting in different stochastic runs

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
| `--seed` | None | Random seed; if omitted, the random seed is not fixed, resulting in different stochastic runs |

---

## Results

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

| Dataset | Accuracy (%) | F1 Score (%) | Loss | Forgetting Rate |
|---|---|---|---|---|
| WISDM | 96.30 ± 0.54 | 96.21 ± 0.59 | 0.118 ± 0.020 | 0.011 ± 0.003 |
| UCI-HAR (natural) | 96.13 ± 0.24 | 96.04 ± 0.28 | 0.117 ± 0.008 | 0.011 ± 0.002 |
| UCI-HAR (α=0.1) | 96.09 ± 0.61 | 95.99 ± 0.68 | 0.120 ± 0.019 | 0.012 ± 0.004 |
| UCI-HAR (α=0.5) | 94.42 ± 4.78 | 93.84 ± 5.90 | 0.168 ± 0.121 | 0.024 ± 0.036 |
| MotionSense | 94.00 ± 1.92 | 93.65 ± 2.29 | 0.193 ± 0.057 | 0.016 ± 0.005 |

### Output Files

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

## Ablation

### Prototype Update Strategy

To compare EMA vs. alternative prototype update strategies:

```bash
bash run_prototype_experiments.sh
```

Runs `direct`, `simple_avg`, `cumulative` strategies (5 runs each) on UCI-HAR α=0.1.
EMA results come from `result/ucihar_alpha01/`.

Results from the paper (UCI-HAR α=0.1):

| Strategy | Accuracy (%) | F1 (%) | Loss | Forgetting Rate |
|---|---|---|---|---|
| Direct | 87.02 ± 1.13 | 84.13 ± 1.22 | 0.355 ± 0.030 | 0.019 ± 0.006 |
| Simple Average | 86.89 ± 2.65 | 83.99 ± 2.97 | 0.365 ± 0.071 | 0.031 ± 0.018 |
| Cumulative | 86.71 ± 2.66 | 83.88 ± 3.45 | 0.394 ± 0.060 | 0.031 ± 0.021 |
| **EMA β=0.9 (Proposed)** | **96.09 ± 0.61** | **95.99 ± 0.68** | **0.120 ± 0.019** | **0.012 ± 0.004** |

### Component Ablation

Results from the paper showing the contribution of each component (UCI-HAR α=0.1 and WISDM):

<table>
  <thead>
    <tr>
      <th>Dataset</th>
      <th>Method</th>
      <th>Accuracy (↑)</th>
      <th>F1 Score (↑)</th>
      <th>Loss (↓)</th>
      <th>Forgetting Rate (↓)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="5"><strong>WISDM</strong></td>
      <td>w/o All</td>
      <td>93.33 ± 1.82</td>
      <td>92.99 ± 2.06</td>
      <td>0.179 ± 0.039</td>
      <td>0.013 ± 0.001</td>
    </tr>
    <tr>
      <td>w/o Per</td>
      <td>96.05 ± 1.56</td>
      <td>95.88 ± 1.73</td>
      <td>0.125 ± 0.048</td>
      <td>0.020 ± 0.013</td>
    </tr>
    <tr>
      <td>w/o P</td>
      <td>89.99 ± 15.16</td>
      <td>88.54 ± 18.22</td>
      <td>0.274 ± 0.371</td>
      <td>0.032 ± 0.055</td>
    </tr>
    <tr>
      <td>w/o CCRA</td>
      <td>87.13 ± 14.91</td>
      <td>85.31 ± 18.04</td>
      <td>0.337 ± 0.341</td>
      <td>0.045 ± 0.071</td>
    </tr>
    <tr>
      <td><strong>Proposed</strong></td>
      <td><strong>96.30 ± 0.54</strong></td>
      <td><strong>96.21 ± 0.59</strong></td>
      <td><strong>0.118 ± 0.020</strong></td>
      <td><strong>0.011 ± 0.003</strong></td>
    </tr>
    <tr>
      <td rowspan="5"><strong>UCI HAR α=0.1</strong></td>
      <td>w/o All</td>
      <td>76.17 ± 7.84</td>
      <td>71.84 ± 8.97</td>
      <td>0.673 ± 0.227</td>
      <td>0.073 ± 0.073</td>
    </tr>
    <tr>
      <td>w/o Per</td>
      <td>86.78 ± 2.34</td>
      <td>85.10 ± 2.55</td>
      <td>0.375 ± 0.053</td>
      <td>0.040 ± 0.011</td>
    </tr>
    <tr>
      <td>w/o P</td>
      <td>81.78 ± 1.17</td>
      <td>78.10 ± 1.42</td>
      <td>0.506 ± 0.019</td>
      <td>0.026 ± 0.008</td>
    </tr>
    <tr>
      <td>w/o CCRA</td>
      <td>75.18 ± 2.98</td>
      <td>69.69 ± 3.82</td>
      <td>0.639 ± 0.076</td>
      <td>0.067 ± 0.019</td>
    </tr>
    <tr>
      <td><strong>Proposed</strong></td>
      <td><strong>96.09 ± 0.61</strong></td>
      <td><strong>95.99 ± 0.68</strong></td>
      <td><strong>0.120 ± 0.019</strong></td>
      <td><strong>0.012 ± 0.004</strong></td>
    </tr>
  </tbody>
</table>

---

## Citation

If you use this code, please cite:

```bibtex
@article{fedrapt2025,
  title   = {FedRAPT: Federated Representation-Aligned Prototypical Contrastive Learning},
  author  = {},
  year    = {2025}
}
```

## License

This project is licensed under the MIT License.
