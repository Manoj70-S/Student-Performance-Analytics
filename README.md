# Student-Performance-Analytics

Hands-on Python NumPy project focused on numerical computing, array manipulation, data analysis, statistical operations, and real-world problem-solving.

## Overview

This project analyzes exam performance for **1,000 engineering students** segregated across **5 streams** (`CS`, `IS`, `EC`, `ME`, `AI` — 200 students each) using **pure NumPy** (no pandas dependency).

### Scoring Scheme (VTU-Style)
- **IA 1 Marks:** `0 – 50`
- **IA 2 Marks:** `0 – 50`
- **Total IA:** `(IA 1 + IA 2) / 2` — reduces combined `100` to `50`
- **SEE Marks:** `0 – 100` (reduced to `50` via `SEE / 2`)
- **Final Score:** `Total IA (50) + SEE Reduced (50) = 100`
- **Pass Criteria:** `Total IA >= 20`, `SEE >= 35`, and `Final Score >= 40`

## Features
- **Dataset Generator (`generate_dataset.py`):** Generates a reproducible 1,000-student master CSV plus 5 stream-segregated CSVs (`1DT24<STREAM><XXX>` USN format) with realistic correlated scores.
- **Descriptive Statistics:** Mean, median, standard deviation, min, max, quartiles (`Q25`, `Q75`), and `P90` across all exam components.
- **Stream-Wise Comparison:** Compares averages, spread, pass rates, and stream toppers across `AI`, `CS`, `IS`, `EC`, and `ME`.
- **Rankings & Toppers:** Multi-key tie-broken rankings (`np.lexsort`) overall and per stream, plus Top Most Improved students (`IA 1 -> IA 2`).
- **Failed, At-Risk & Outlier Detection:** Failure reason breakdown, borderline passing students (within 5 marks of failing), Z-score outliers (`|Z| >= 2.5`), and IA-vs-SEE divergence detection.
- **Correlation Matrix & Grade Distribution:** 5x5 Pearson correlation matrix (`np.corrcoef`) and vectorized letter grades (`O`, `A+`, `A`, `B+`, `B`, `C`, `P`, `F`).
- **Interactive CLI (`main.py`):** Student lookup (by USN or partial name), side-by-side student comparison, "What-If" SEE target calculator, stream filtering, and enriched CSV report export.

## NumPy Concepts Demonstrated
- Array creation & stacking: `np.array`, `np.empty`, `np.arange`, `np.vstack`
- Random simulation & rounding: `np.random.default_rng`, `rng.normal`, `rng.choice`, `np.clip`, `np.rint`, `np.round`
- Statistics & aggregations: `np.mean`, `np.median`, `np.std`, `np.min`, `np.max`, `np.percentile`, `np.sum`, `np.argmax`
- Boolean masking & vectorized logic: `&`, `|`, `~`, `np.where`, `np.select`, `np.any`
- Sorting, ranking & percentiles: `np.sort`, `np.argsort`, `np.lexsort`, `np.searchsorted`
- Correlation, counts & string ops: `np.corrcoef`, `np.unique(..., return_counts=True)`, `np.char.upper`, `np.char.find`

## Project Structure

```text
.
├── analytics.py                     # Pure NumPy analytics engine
├── main.py                          # Interactive CLI & report runner
├── generate_dataset.py              # Reproducible 1,000-student CSV generator
├── requirements.txt
├── data/
│   ├── students_all.csv             # All 1,000 students (segregated by stream)
│   └── streams/
│       ├── AI_students.csv
│       ├── CS_students.csv
│       ├── EC_students.csv
│       ├── IS_students.csv
│       └── ME_students.csv
└── output/
    └── student_results_report.csv   # Exported enriched analytics report
```

## Quick Start

```bash
# Install dependency
pip install -r requirements.txt

# (Optional) Regenerate the 1,000-student dataset
python generate_dataset.py

# Print full analytics report & export output/student_results_report.csv
python main.py --report

# Report for a single stream (CS, IS, EC, ME, or AI)
python main.py --report --stream CS

# Launch interactive CLI menu
python main.py
```
