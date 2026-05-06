# MCDA Sensitivity Analysis Code

Supporting Python scripts for the Multi-Criteria Decision Analysis in Part II of the dissertation. These scripts generate the sensitivity figures and independently verify the weighted-score calculations in the accompanying Excel workbook in the 'Data' folder.

## Scripts

### `sensitivity.py`

Generates the Layer 2 and Layer 3 sensitivity outputs reported in the MCDA section and appendix.

**Layer 2 (2D weight-plane):** Computes the winning option at every feasible point on the (combined financial weight, combined risk weight) plane. Produces five PNG figures:

- `layer2_t5_main.png` — T=5 main variant (1:1 financial, 1:1 risk)
- `layer2_t5_rob1_fin21.png` — T=5 robustness variant (2:1 financial split)
- `layer2_t5_rob2_fin12.png` — T=5 robustness variant (1:2 financial split)
- `layer2_t7_main.png` — T=7 main variant (post-crossover)
- `layer2_t5_appendix_c1_c4.png` — T=5 single-criterion plane (C1 vs C4)

**Layer 3 (score-perturbation Monte Carlo):** Perturbs each qualitative score (C3–C6) by a uniform random draw of ±5, ±10, or ±15, recomputes the weighted totals, and records how often each option wins across 1,000 iterations. Produces three gap-distribution histograms:

- `layer3_balanced_gaps_t5.png` — Balanced C−A gap at ±5/±10/±15
- `layer3_balanced_gap_t7.png` — Balanced C−A gap at T=7 ±10
- `layer3_cfo_cb_gap_t5.png` — CFO C−B gap at ±10

All numerical results (region shares, win percentages, z-scores, gap statistics) are printed to the console.

### `python_crosscheck.py`

Independent verification of the MCDA weighted scores computed in the Excel workbook (MCDA_workbook.xlsx). Reads the locked scores and weights, computes SUMPRODUCT totals for all four profiles at both T=5 and T=7, and confirms agreement with the Excel results. Also computes:

- One-way switchover thresholds for all six criteria
- Layer 1 condition-check (which profiles violate which one-way conditions)
- T=7 rescored results

All outputs are printed to the console.

## Requirements

- Python 3.10 or later
- numpy
- matplotlib

## Setup and installation

1. Clone this repository:
   ```
   git clone https://github.com/joelthomas21/mcda-msin0032.git
   cd mcda-msin0032
   ```

2. Create and activate a virtual environment:
   ```
   python -m venv test_env
   source test_env/bin/activate      # Mac/Linux
   test_env\Scripts\activate         # Windows
   ```

3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Running the scripts

Generate the sensitivity figures and perturbation results:
```
python sensitivity.py
```
Output PNG files are saved to the `outputs/` folder (created automatically if it does not exist). All numerical results are printed to the console.

Run the independent cross-check:
```
python python_crosscheck.py
```
This prints the weighted scores for all four profiles at both horizons and confirms they match the Excel workbook values to four decimal places.

## Reproducibility

- All random operations use `numpy.random.seed(42)` so that the perturbation results are identical on every run.
- The scripts are self-contained. All scores, weights, and parameters are defined within the scripts rather than read from external files, so no additional data files are needed to run them.
- The accompanying Excel workbook (MCDA_workbook.xlsx) contains the same scores and weights and can be used to manually verify any individual calculation.

Excel files and code setup and formatted using Claude Sonnet 4.6 by Anthropic.