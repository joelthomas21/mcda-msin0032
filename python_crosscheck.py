"""
MCDA v2 — Independent Python cross-check
==========================================
Verifies all Excel-computed values against NumPy computation.
"""
import numpy as np

# ── Locked scores ──
# T=5 (rows: C1..C6, cols: A, B, C)
scores_t5 = np.array([
    [  0, 100,  47],   # C1 EMV: A≈€0, B=€12.602m→100, C=€5.893m→47
    [100,   0,  59],   # C2 downside: A≈€0→100, B=-€69.785m→0, C=-€28.452m→59
    [ 85,  20,  45],   # C3 UK licence
    [ 95,  20,  55],   # C4 run-risk
    [ 95,  10,  75],   # C5 reversibility
    [ 15,  55,  90],   # C6 governance + GBP
], dtype=float)

# T=7 (C overtakes B on EMV at 6.1yr crossover)
scores_t7 = np.array([
    [  0,  76, 100],   # C1: C=€23.065m→100, B=€17.643m→76
    [100,   0,  71],   # C2: B=-€97.699m→0, C=-€28.452m→71
    [ 85,  20,  45],   # C3 unchanged
    [ 95,  20,  55],   # C4 unchanged
    [ 95,  10,  75],   # C5 unchanged
    [ 15,  55,  90],   # C6 unchanged
], dtype=float)

# ── SWING raw ranks and normalisation ──
swing_raw = np.array([
    [100,  30,  60, 100],   # C1
    [ 50,  70,  30, 100],   # C2
    [ 15,  90,  35, 100],   # C3
    [ 10, 100,  40, 100],   # C4
    [ 25,  50,  80, 100],   # C5
    [ 35,  25, 100, 100],   # C6
], dtype=float)

weights = swing_raw / swing_raw.sum(axis=0)
profiles = ['CFO', 'CRO', 'Strategy', 'Balanced']
options = ['A', 'B', 'C']

print("=" * 80)
print("PYTHON CROSS-CHECK — MCDA v2")
print("=" * 80)

# ── Normalised weights ──
print("\nNormalised weights:")
crit_names = ['C1 EMV', 'C2 Downside', 'C3 Licence', 'C4 Run-risk', 'C5 Reversibility', 'C6 Governance']
print(f"{'Criterion':<20} " + " ".join(f"{p:>10}" for p in profiles))
for i, name in enumerate(crit_names):
    print(f"{name:<20} " + " ".join(f"{weights[i,j]:>10.4f}" for j in range(4)))
print(f"{'Sum':<20} " + " ".join(f"{weights[:,j].sum():>10.4f}" for j in range(4)))

# ── Weighted scores ──
def compute_results(scores, label):
    ws = weights.T @ scores  # (4 profiles) × (3 options)
    print(f"\n{'─' * 80}")
    print(f"Weighted scores — {label}")
    print(f"{'─' * 80}")
    print(f"{'Profile':<12} " + " ".join(f"{o:>10}" for o in options) + f"  {'Winner':>8}  {'Margin':>8}  {'2nd':>6}")
    for i, p in enumerate(profiles):
        row = ws[i]
        winner_idx = int(np.argmax(row))
        sorted_desc = np.sort(row)[::-1]
        margin = sorted_desc[0] - sorted_desc[1]
        second_idx = int(np.where(row == sorted_desc[1])[0][0])
        print(f"{p:<12} " + " ".join(f"{row[j]:>10.2f}" for j in range(3))
              + f"  {options[winner_idx]:>8}  {margin:>8.2f}  {options[second_idx]:>6}")
    return ws

ws_t5 = compute_results(scores_t5, "T=5")
ws_t7 = compute_results(scores_t7, "T=7")

# ── One-way switchover thresholds (T=5) ──
print(f"\n{'=' * 80}")
print("ONE-WAY SWITCHOVER THRESHOLDS — T=5")
print(f"{'=' * 80}")
print(f"{'Criterion':<18} {'B=C':>12} {'A=C':>12}  Interpretation")

for i in range(6):
    # B=C: w where B score = C score
    # B(w) = w*B[i] + (1-w)/5 * sum(B[rest])
    # C(w) = w*C[i] + (1-w)/5 * sum(C[rest])
    # B=C when: w*(B[i]-C[i]) + (1-w)/5 * (sum_B_rest - sum_C_rest) = 0
    sum_B_rest = scores_t5[:,1].sum() - scores_t5[i,1]
    sum_C_rest = scores_t5[:,2].sum() - scores_t5[i,2]
    sum_A_rest = scores_t5[:,0].sum() - scores_t5[i,0]

    # B=C
    gap_bc = sum_C_rest - sum_B_rest
    den_bc = 5 * (scores_t5[i,1] - scores_t5[i,2]) + gap_bc
    w_bc = gap_bc / den_bc if den_bc != 0 else None

    # A=C
    gap_ac = sum_C_rest - sum_A_rest
    den_ac = 5 * (scores_t5[i,0] - scores_t5[i,2]) + gap_ac
    w_ac = gap_ac / den_ac if den_ac != 0 else None

    def fmt(v):
        if v is None: return 'undefined'
        return f'{v:.4f}'

    def interp(v_bc, v_ac):
        parts = []
        if isinstance(v_bc, (int, float)) and 0 < v_bc < 1:
            parts.append(f'B overtakes C at w>{v_bc:.3f}')
        if isinstance(v_ac, (int, float)) and 0 < v_ac < 1:
            parts.append(f'A overtakes C at w>{v_ac:.3f}')
        if not parts:
            return 'No feasible flip for B; ' + (f'A overtakes C at w>{v_ac:.3f}' if isinstance(v_ac, (int, float)) and 0 < v_ac < 1 else 'A already leads or no flip')
        return '; '.join(parts)

    print(f"C{i+1:<17} {fmt(w_bc):>12} {fmt(w_ac):>12}  {interp(w_bc, w_ac)}")

# ── Profile condition check against thresholds ──
print(f"\n{'=' * 80}")
print("PROFILE CONDITION CHECK AGAINST T=5 THRESHOLDS")
print(f"{'=' * 80}")

# For C to win, these conditions must hold:
# C1: 0.2193 < w < 0.4525
# C2: w < 0.0969
# C3: w < 0.0950
# C4: w < 0.0950
# C5: w < 0.0099
# C6: w > 0.2004

thresholds = {
    'C1 lower': (0, 0.2193, 'above'),  # w must be ABOVE 0.2193
    'C1 upper': (0, 0.4525, 'below'),  # w must be BELOW 0.4525
    'C2': (1, 0.0969, 'below'),
    'C3': (2, 0.0950, 'below'),
    'C4': (3, 0.0950, 'below'),
    'C5': (4, 0.0099, 'below'),
    'C6': (5, 0.2004, 'above'),
}

print(f"\n{'Condition':<14} {'Threshold':>10} {'Dir':>8} {'CFO':>10} {'CRO':>10} {'Strategy':>10} {'Balanced':>10}")
for label, (ci, thresh, direction) in thresholds.items():
    row = f"{label:<14} {thresh:>10.4f} {'w>t' if direction=='above' else 'w<t':>8}"
    for j in range(4):
        w = weights[ci, j]
        if direction == 'above':
            ok = w > thresh
        else:
            ok = w < thresh
        status = f"{w:.4f} {'✓' if ok else '✗'}"
        row += f" {status:>10}"
    print(row)

# ── Verification summary ──
print(f"\n{'=' * 80}")
print("SUMMARY COMPARISON: Excel vs Python")
print(f"{'=' * 80}")

# Compare T=5 results
print("\nT=5 results match check:")
excel_t5 = {
    'CFO':     (43.09, 53.94, 59.15),
    'CRO':     (80.21, 23.77, 57.78),
    'Strategy':(54.71, 40.00, 67.72),
    'Balanced':(65.00, 34.17, 61.83),
}
for i, p in enumerate(profiles):
    py = tuple(round(ws_t5[i, j], 2) for j in range(3))
    ex = excel_t5[p]
    match = all(abs(py[j] - ex[j]) < 0.02 for j in range(3))
    print(f"  {p:<12} Excel={ex}  Python={py}  {'✓ MATCH' if match else '✗ MISMATCH'}")

print("\nT=7 results match check:")
excel_t7 = {
    'CFO':     (43.09, 43.72, 84.26),
    'CRO':     (80.21, 21.79, 64.44),
    'Strategy':(54.71, 35.83, 77.99),
    'Balanced':(65.00, 30.17, 72.67),
}
for i, p in enumerate(profiles):
    py = tuple(round(ws_t7[i, j], 2) for j in range(3))
    ex = excel_t7[p]
    match = all(abs(py[j] - ex[j]) < 0.02 for j in range(3))
    print(f"  {p:<12} Excel={ex}  Python={py}  {'✓ MATCH' if match else '✗ MISMATCH'}")

print("\nSwitchover threshold match check (T=5):")
excel_thresholds = {
    'C1': (0.4525, 0.2193),
    'C2': (-0.5691, 0.0969),
    'C3': (8.8125, 0.0950),
    'C4': (-2.9773, 0.0950),
    'C5': (-0.4509, 0.0099),
    'C6': (-2.9773, 0.2004),
}
for i in range(6):
    sum_B_rest = scores_t5[:,1].sum() - scores_t5[i,1]
    sum_C_rest = scores_t5[:,2].sum() - scores_t5[i,2]
    sum_A_rest = scores_t5[:,0].sum() - scores_t5[i,0]
    gap_bc = sum_C_rest - sum_B_rest
    den_bc = 5 * (scores_t5[i,1] - scores_t5[i,2]) + gap_bc
    w_bc = gap_bc / den_bc if den_bc != 0 else None
    gap_ac = sum_C_rest - sum_A_rest
    den_ac = 5 * (scores_t5[i,0] - scores_t5[i,2]) + gap_ac
    w_ac = gap_ac / den_ac if den_ac != 0 else None

    ex = excel_thresholds[f'C{i+1}']
    match_bc = abs(w_bc - ex[0]) < 0.001 if w_bc is not None else False
    match_ac = abs(w_ac - ex[1]) < 0.001 if w_ac is not None else False
    print(f"  C{i+1}: B=C  Excel={ex[0]:.4f}  Python={w_bc:.4f}  {'✓' if match_bc else '✗'}  |  "
          f"A=C  Excel={ex[1]:.4f}  Python={w_ac:.4f}  {'✓' if match_ac else '✗'}")
