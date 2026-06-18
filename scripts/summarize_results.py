#!/usr/bin/env python3
"""
Summarize results from run_all.sh into a single table.
Reads result/{dataset}/fedrapt_metrics_iter*.csv and prints mean +- std.
"""
import os
import glob
import numpy as np
import pandas as pd

CONFIGS = [
    ('WISDM',            'wisdm'),
    ('UCI-HAR (natural)','ucihar'),
    ('UCI-HAR (a=0.1)',  'ucihar_alpha01'),
    ('UCI-HAR (a=0.5)',  'ucihar_alpha05'),
    ('MotionSense',      'motionsense'),
]

BASE = os.path.join(os.path.dirname(__file__), '..', 'result')

print(f"\n{'Dataset':<22} {'Accuracy':>16} {'F1 Score':>16} {'Loss':>14} {'Forgetting Rate':>18}")
print('-' * 90)

for label, tag in CONFIGS:
    files = sorted(f for f in glob.glob(os.path.join(BASE, tag, 'fedrapt_metrics_iter*.csv'))
                   if '_comm' not in f and '_config' not in f)
    if not files:
        print(f"{label:<22}  (no results found in result/{tag}/)")
        continue

    accs, f1s, losses, frs = [], [], [], []
    for f in files:
        df = pd.read_csv(f)
        final = df[(df['scope'] == 'personal_final') & (df['round'] == df['round'].max())]
        accs.append(final['accuracy'].mean() * 100)
        f1s.append(final['f1'].mean() * 100)
        losses.append(final['loss'].mean())
        frs.append(final['forgetting_rate'].mean())

    acc_str  = f"{np.mean(accs):.2f} +- {np.std(accs):.2f}"
    f1_str   = f"{np.mean(f1s):.2f} +- {np.std(f1s):.2f}"
    loss_str = f"{np.mean(losses):.4f} +- {np.std(losses):.4f}"
    fr_str   = f"{np.mean(frs):.4f} +- {np.std(frs):.4f}"

    print(f"{label:<22} {acc_str:>16} {f1_str:>16} {loss_str:>14} {fr_str:>18}  (n={len(files)})")

print()
