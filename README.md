# DrishtiSingh-19951212
# Etraveli Group — Data Insight Analyst Assessment

## Setup Instructions

### Requirements
- Python 3.10+
- Node.js 18+ (for presentation generation only)

### Install Python dependencies
```bash
pip install pandas==2.2.2 pyarrow==16.0.0 numpy==1.26.4 matplotlib==3.8.4 \
            seaborn==0.13.2 scipy==1.13.0 scikit-learn==1.4.2 xgboost==2.0.3
```

### Install Node dependencies
```bash
npm install pptxgenjs
```

### Run the analysis
```bash
# Place orders.parquet and errands.parquet in the project root
# All paths are resolved relative to the script location

python src/parts1_eda.py  # Parts 1
python src/parts2_3_4.py        # Parts 2 & 3 — statistical + ML analysis
node presentation/build_pptx.js           # Part 4 — generates executive_presentation.pptx
```

### Output
- `charts/` — all publication-quality visualisations
- `executive_presentation.pptx` — 5-slide executive deck

---

## Approach Summary (150 words)

The analysis begins with a critical data quality step: the join key between the two datasets is non-trivial. `order_number` in the errands table is a base-36 encoding of the integer `order_id` in the orders table. Resolving this unlocks a 100% match rate on real errands. A further 204,404 test records (`is_test_errand=1`) were filtered before any analysis.

The statistical work tests four order-level factors against contact propensity using chi-square and Mann-Whitney U tests, with honest reporting of both statistical and practical significance. Survival analysis characterises time-to-first-contact without an external library by constructing the empirical survival function directly.

The machine learning section builds two models — Logistic Regression and Random Forest — using only booking-time features to prevent leakage. AUC is moderate (0.61) and the analysis is transparent about why. Segmentation uses KMeans with silhouette selection. The executive deck translates all findings into business language with no statistical jargon.

---


