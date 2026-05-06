"""
MCDA v2 Sensitivity Analysis — Layers 2 and 3
==============================================
Layer 2: 2D weight-plane robustness maps (5 variants)
Layer 3: Score-perturbation Monte Carlo (3 envelopes × 4 profiles × 2 horizons)

Updated scores from final Monte Carlo:
  T=5: C1(C)=47, C2(C)=59  (was 70, 65)
  T=7: C1(C)=100, C2(C)=71 (C overtakes B on EMV at 6.1yr crossover)

All numerical results printed to console for cross-checking.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch
import os

OUT = os.path.dirname(os.path.abspath(__file__))
os.makedirs(OUT, exist_ok=True)

# ── Locked inputs ──
scores_t5 = np.array([
    [  0, 100,  47],   # C1
    [100,   0,  59],   # C2
    [ 85,  20,  45],   # C3
    [ 95,  20,  55],   # C4
    [ 95,  10,  75],   # C5
    [ 15,  55,  90],   # C6
], dtype=float)

scores_t7 = np.array([
    [  0,  76, 100],   # C1
    [100,   0,  71],   # C2
    [ 85,  20,  45],   # C3
    [ 95,  20,  55],   # C4
    [ 95,  10,  75],   # C5
    [ 15,  55,  90],   # C6
], dtype=float)

swing_raw = np.array([
    [100,  30,  60, 100],
    [ 50,  70,  30, 100],
    [ 15,  90,  35, 100],
    [ 10, 100,  40, 100],
    [ 25,  50,  80, 100],
    [ 35,  25, 100, 100],
], dtype=float)
weights = swing_raw / swing_raw.sum(axis=0)
profiles = ['CFO', 'CRO', 'Strategy', 'Balanced']
options = ['A', 'B', 'C']

def profile_points():
    return {p: (weights[0,j]+weights[1,j], weights[2,j]+weights[3,j])
            for j,p in enumerate(profiles)}

# ════════════════════════════════════════════════════════════
# LAYER 2: 2D WEIGHT-PLANE
# ════════════════════════════════════════════════════════════
def compute_plane(scores, fin_split, risk_split, step=0.01):
    grid = np.arange(0, 1.0001, step)
    Z = np.full((len(grid), len(grid)), -1, dtype=int)
    for i, r in enumerate(grid):
        for j, f in enumerate(grid):
            if f + r > 1.0 + 1e-9:
                continue
            res = max(0.0, 1.0 - f - r)
            w = np.array([
                f * fin_split[0], f * fin_split[1],
                r * risk_split[0], r * risk_split[1],
                res * 0.5, res * 0.5
            ])
            s = w @ scores
            Z[i, j] = int(np.argmax(s))
    return grid, Z

def plot_plane(grid, Z, title, outfile, pts):
    fig, ax = plt.subplots(figsize=(8, 6.5))
    cmap = ListedColormap(['#f0f0f0', '#5B9BD5', '#ED7D31', '#70AD47'])
    norm = BoundaryNorm([-1.5, -0.5, 0.5, 1.5, 2.5], cmap.N)
    ax.imshow(Z, origin='lower', extent=[0,1,0,1], cmap=cmap, norm=norm,
              aspect='auto', interpolation='nearest')
    markers = {'CFO':'o','CRO':'s','Strategy':'^','Balanced':'D'}
    for name,(f,r) in pts.items():
        ax.scatter(f, r, s=180, marker=markers[name],
                   facecolor='white', edgecolor='black', linewidth=2, zorder=5)
        ax.annotate(name, (f,r), textcoords='offset points',
                    xytext=(8,8), fontsize=10, fontweight='bold')
    ax.set_xlabel('Combined financial weight (w(C1)+w(C2))', fontsize=11)
    ax.set_ylabel('Combined risk weight (w(C3)+w(C4))', fontsize=11)
    ax.set_title(title, fontsize=11)
    ax.set_xlim(0,1); ax.set_ylim(0,1)
    ax.legend(handles=[
        Patch(facecolor='#5B9BD5', edgecolor='black', label='Option A wins'),
        Patch(facecolor='#ED7D31', edgecolor='black', label='Option B wins'),
        Patch(facecolor='#70AD47', edgecolor='black', label='Option C wins'),
        Patch(facecolor='#f0f0f0', edgecolor='black', label='Infeasible'),
    ], loc='upper right', framealpha=0.95)
    ax.plot([0,1],[1,0],'k--',alpha=0.4,linewidth=1)
    plt.tight_layout()
    plt.savefig(outfile, dpi=150, bbox_inches='tight')
    plt.close()

def region_summary(Z, label):
    feasible = (Z >= 0).sum()
    counts = {o: (Z == i).sum() for i, o in enumerate(options)}
    pcts = {o: 100*counts[o]/feasible for o in options}
    print(f'  {label:<55} A={pcts["A"]:>5.1f}%  B={pcts["B"]:>5.1f}%  C={pcts["C"]:>5.1f}%')
    return pcts

pts = profile_points()

print('=' * 80)
print('LAYER 2: 2D WEIGHT-PLANE ROBUSTNESS MAPS')
print('=' * 80)
print(f'\nProfile coordinates: {pts}')

# Variant 1: Main (T=5, 1:1/1:1)
grid, Z1 = compute_plane(scores_t5, (0.5,0.5), (0.5,0.5))
plot_plane(grid, Z1, 'T=5 main: 1:1 financial, 1:1 risk', f'{OUT}/layer2_t5_main.png', pts)
pct1 = region_summary(Z1, 'T=5 Main (1:1 fin, 1:1 risk)')

# Variant 2: Robustness 1 (T=5, 2:1 financial)
grid, Z2 = compute_plane(scores_t5, (2/3,1/3), (0.5,0.5))
plot_plane(grid, Z2, 'T=5 robustness: 2:1 financial (C1 heavier), 1:1 risk', f'{OUT}/layer2_t5_rob1_fin21.png', pts)
pct2 = region_summary(Z2, 'T=5 Rob1 (2:1 fin, 1:1 risk)')

# Variant 3: Robustness 2 (T=5, 1:2 financial)
grid, Z3 = compute_plane(scores_t5, (1/3,2/3), (0.5,0.5))
plot_plane(grid, Z3, 'T=5 robustness: 1:2 financial (C2 heavier), 1:1 risk', f'{OUT}/layer2_t5_rob2_fin12.png', pts)
pct3 = region_summary(Z3, 'T=5 Rob2 (1:2 fin, 1:1 risk)')

# Variant 4: T=7 main (1:1/1:1)
grid, Z4 = compute_plane(scores_t7, (0.5,0.5), (0.5,0.5))
plot_plane(grid, Z4, 'T=7 main: 1:1 financial, 1:1 risk (post-crossover)', f'{OUT}/layer2_t7_main.png', pts)
pct4 = region_summary(Z4, 'T=7 Main (1:1 fin, 1:1 risk)')

# Variant 5: Appendix single-criterion plane (C1 vs C4)
def compute_c1_c4_plane(scores, step=0.01):
    grid = np.arange(0, 1.0001, step)
    Z = np.full((len(grid), len(grid)), -1, dtype=int)
    for i, w4 in enumerate(grid):
        for j, w1 in enumerate(grid):
            if w1 + w4 > 1.0 + 1e-9:
                continue
            res = max(0.0, 1.0 - w1 - w4)
            w = np.array([w1, res/4, res/4, w4, res/4, res/4])
            s = w @ scores
            Z[i,j] = int(np.argmax(s))
    return grid, Z

grid, Z5 = compute_c1_c4_plane(scores_t5)
fig, ax = plt.subplots(figsize=(8, 6.5))
cmap = ListedColormap(['#f0f0f0', '#5B9BD5', '#ED7D31', '#70AD47'])
norm = BoundaryNorm([-1.5, -0.5, 0.5, 1.5, 2.5], cmap.N)
ax.imshow(Z5, origin='lower', extent=[0,1,0,1], cmap=cmap, norm=norm,
          aspect='auto', interpolation='nearest')
# Plot profile points at (C1, C4) coordinates
for j, p in enumerate(profiles):
    ax.scatter(weights[0,j], weights[3,j], s=180,
               marker={'CFO':'o','CRO':'s','Strategy':'^','Balanced':'D'}[p],
               facecolor='white', edgecolor='black', linewidth=2, zorder=5)
    ax.annotate(p, (weights[0,j], weights[3,j]), textcoords='offset points',
                xytext=(8,8), fontsize=10, fontweight='bold')
ax.set_xlabel('w(C1) — EMV weight', fontsize=11)
ax.set_ylabel('w(C4) — Run-risk weight', fontsize=11)
ax.set_title('T=5 appendix: C1 vs C4 (other 4 criteria share residual equally)', fontsize=11)
ax.legend(handles=[
    Patch(facecolor='#5B9BD5', edgecolor='black', label='A wins'),
    Patch(facecolor='#ED7D31', edgecolor='black', label='B wins'),
    Patch(facecolor='#70AD47', edgecolor='black', label='C wins'),
    Patch(facecolor='#f0f0f0', edgecolor='black', label='Infeasible'),
], loc='upper right', framealpha=0.95)
ax.plot([0,1],[1,0],'k--',alpha=0.4)
plt.tight_layout()
plt.savefig(f'{OUT}/layer2_t5_appendix_c1_c4.png', dpi=150, bbox_inches='tight')
plt.close()
pct5 = region_summary(Z5, 'T=5 Appendix (C1 vs C4)')

# Summary table
print(f'\n  {"Variant":<55} {"A":>6} {"B":>6} {"C":>6}')
for label, pct in [('T=5 Main (1:1/1:1)', pct1), ('T=5 Rob1 (2:1 fin)', pct2),
                    ('T=5 Rob2 (1:2 fin)', pct3), ('T=7 Main (1:1/1:1)', pct4),
                    ('T=5 Appendix (C1 vs C4)', pct5)]:
    print(f'  {label:<55} {pct["A"]:>5.1f}% {pct["B"]:>5.1f}% {pct["C"]:>5.1f}%')

# ════════════════════════════════════════════════════════════
# LAYER 3: SCORE-PERTURBATION MONTE CARLO
# ════════════════════════════════════════════════════════════
print(f'\n{"=" * 80}')
print('LAYER 3: SCORE-PERTURBATION MONTE CARLO')
print('=' * 80)

rng = np.random.default_rng(seed=42)
N_ITER = 1000
QUAL_ROWS = [2, 3, 4, 5]  # C3-C6 (0-indexed)

def run_perturbation(scores, delta, label):
    results = {p: {'A':0,'B':0,'C':0} for p in profiles}
    gaps = {p: [] for p in profiles}

    for _ in range(N_ITER):
        pert = rng.uniform(-delta, delta, size=(len(QUAL_ROWS), 3))
        ps = scores.copy()
        for k, ri in enumerate(QUAL_ROWS):
            ps[ri] = np.clip(scores[ri] + pert[k], 0, 100)
        for j, p in enumerate(profiles):
            s = weights[:, j] @ ps
            w_idx = int(np.argmax(s))
            results[p][options[w_idx]] += 1
            # Track gap between winner and runner-up for the contested profiles
            sorted_s = np.sort(s)[::-1]
            gaps[p].append((s[2] - s[0], s))  # (C-A gap, full scores)

    print(f'\n  {label} (±{delta:.0f}, N={N_ITER}):')
    print(f'  {"Profile":<12} {"A wins":>8} {"B wins":>8} {"C wins":>8}')
    for p in profiles:
        r = results[p]
        print(f'  {p:<12} {100*r["A"]/N_ITER:>7.1f}% {100*r["B"]/N_ITER:>7.1f}% {100*r["C"]/N_ITER:>7.1f}%')

    return results, gaps

# T=5 at three envelopes
results_t5 = {}
gaps_t5 = {}
for delta in [5, 10, 15]:
    r, g = run_perturbation(scores_t5, delta, f'T=5')
    results_t5[delta] = r
    gaps_t5[delta] = g

# T=7 at ±10
results_t7_10, gaps_t7_10 = run_perturbation(scores_t7, 10, 'T=7')

# ── Summary table ──
print(f'\n  COMBINED PERTURBATION SUMMARY:')
print(f'  {"Horizon":<8} {"Envelope":<10} {"CFO":>12} {"CRO":>12} {"Strategy":>12} {"Balanced":>12}')
for delta in [5, 10, 15]:
    r = results_t5[delta]
    row = f'  {"T=5":<8} {"±"+str(delta):<10}'
    for p in profiles:
        winner = max(r[p], key=r[p].get)
        pct = 100 * r[p][winner] / N_ITER
        row += f' {winner}={pct:>5.1f}%   '
    print(row)
r = results_t7_10
row = f'  {"T=7":<8} {"±10":<10}'
for p in profiles:
    winner = max(r[p], key=r[p].get)
    pct = 100 * r[p][winner] / N_ITER
    row += f' {winner}={pct:>5.1f}%   '
print(row)

# ── Histograms ──
# Balanced C-A gap at T=5 for all three envelopes
fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
for idx, delta in enumerate([5, 10, 15]):
    ax = axes[idx]
    gap_arr = np.array([g[0] for g in gaps_t5[delta]['Balanced']])
    ax.hist(gap_arr, bins=40, color='#70AD47', edgecolor='black', alpha=0.85)
    ax.axvline(0, color='red', linestyle='--', linewidth=2, label='Tie (C=A)')
    ax.axvline(gap_arr.mean(), color='black', linestyle='-', linewidth=1.5,
               label=f'Mean={gap_arr.mean():.2f}')
    pct_c = 100 * (gap_arr > 0).sum() / len(gap_arr)
    ax.set_title(f'±{delta}: C wins {pct_c:.1f}%', fontsize=11)
    ax.set_xlabel('C score − A score')
    if idx == 0:
        ax.set_ylabel('Frequency')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
fig.suptitle('T=5 Balanced: C−A gap distribution under score perturbation (seed=42)', fontsize=12)
plt.tight_layout()
plt.savefig(f'{OUT}/layer3_balanced_gaps_t5.png', dpi=150, bbox_inches='tight')
plt.close()

# Balanced C-A gap at T=7 ±10
gap_t7 = np.array([g[0] for g in gaps_t7_10['Balanced']])
fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(gap_t7, bins=40, color='#70AD47', edgecolor='black', alpha=0.85)
ax.axvline(0, color='red', linestyle='--', linewidth=2, label='Tie (C=A)')
ax.axvline(gap_t7.mean(), color='black', linestyle='-', linewidth=1.5,
           label=f'Mean={gap_t7.mean():.2f}')
pct_c_t7 = 100 * (gap_t7 > 0).sum() / len(gap_t7)
ax.set_title(f'T=7 Balanced: C−A gap (±10, C wins {pct_c_t7:.1f}%)', fontsize=11)
ax.set_xlabel('C score − A score')
ax.set_ylabel('Frequency')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f'{OUT}/layer3_balanced_gap_t7.png', dpi=150, bbox_inches='tight')
plt.close()

# CFO C-B gap at T=5 ±10 (thin margin profile)
gap_cfo = np.array([g[1][2] - g[1][1] for g in gaps_t5[10]['CFO']])  # C score - B score
fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(gap_cfo, bins=40, color='#70AD47', edgecolor='black', alpha=0.85)
ax.axvline(0, color='red', linestyle='--', linewidth=2, label='Tie (C=B)')
ax.axvline(gap_cfo.mean(), color='black', linestyle='-', linewidth=1.5,
           label=f'Mean={gap_cfo.mean():.2f}')
pct_c_cfo = 100 * (gap_cfo > 0).sum() / len(gap_cfo)
ax.set_title(f'T=5 CFO: C−B gap (±10, C wins {pct_c_cfo:.1f}%)', fontsize=11)
ax.set_xlabel('C score − B score')
ax.set_ylabel('Frequency')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f'{OUT}/layer3_cfo_cb_gap_t5.png', dpi=150, bbox_inches='tight')
plt.close()

# ── Gap statistics ──
print(f'\n  GAP DISTRIBUTION STATISTICS:')
for label, gap_arr in [
    ('T=5 Balanced ±5',  np.array([g[0] for g in gaps_t5[5]['Balanced']])),
    ('T=5 Balanced ±10', np.array([g[0] for g in gaps_t5[10]['Balanced']])),
    ('T=5 Balanced ±15', np.array([g[0] for g in gaps_t5[15]['Balanced']])),
    ('T=7 Balanced ±10', gap_t7),
    ('T=5 CFO C-B ±10',  gap_cfo),
]:
    pct_pos = 100 * (gap_arr > 0).sum() / len(gap_arr)
    print(f'  {label:<22} mean={gap_arr.mean():>+7.3f}  sd={gap_arr.std():>6.3f}  '
          f'min={gap_arr.min():>+7.3f}  max={gap_arr.max():>+7.3f}  '
          f'%positive={pct_pos:>5.1f}%')

print(f'\n  All figures saved to {OUT}/')
print(f'  Files: {sorted(os.listdir(OUT))}')
