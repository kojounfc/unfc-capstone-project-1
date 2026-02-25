# Analyzing Profit Erosion from Product Returns in E-Commerce
### A Multi-Method Analytics Framework

**Course**: DAMO-699-4 Capstone Project — Winter 2026
**Institution**: University of Niagara Falls Canada

---

## Problem Statement

Product returns are **economic reversal events** that directly erode realized revenue and margin. This project reframes returns beyond operational metrics to quantify:

- **Margin reversal** — the margin lost on returned items (sale price − cost)
- **Incremental processing costs** — customer service, inspection, restocking, logistics ($12 base per return, category-tier adjusted)

> **Core formula:** `Profit Erosion = Margin Reversal + Processing Cost`

---

## Research Questions

| RQ | Focus | Method | Key Result |
|----|-------|--------|------------|
| **RQ1** | Profit erosion differences across product categories & brands | Descriptive Analysis + Kruskal-Wallis | Significant cross-category differences (H₀ rejected) |
| **RQ2** | Customer behavioral segments with differential profit erosion | K-Means Clustering + Gini/Lorenz/Pareto | Behaviorally distinct segments confirmed |
| **RQ3** | Predict high-erosion customers (target AUC > 0.70) | ML Classification (RF, GB, LR) | RF champion AUC = 0.9798 |
| **RQ4** | Marginal associations between behaviors and profit erosion | Log-Linear OLS Regression | Significant behavioral predictors identified |

---

## Data Sources

**Primary**: `bigquery-public-data.thelook_ecommerce` (Google BigQuery)

| Table | Description |
|-------|-------------|
| `order_items` | Item-level transactions (grain level) |
| `orders` | Order-level information |
| `products` | Product catalog with cost and pricing |
| `users` | Customer demographics and acquisition |

**External Validation**: School Specialty LLC (SSL) — U.S. educational supplies B2B retailer
- `data/raw/SSL_Returns_df_yoy.csv` — ~234K return order lines, ~16.7K accounts, 2024–2025
- Used for directional validation of RQ3 and RQ4 models

---

## Repository Structure

```
unfc-capstone-project/
├── app/                        # Streamlit dashboard
│   ├── Home.py                 # Landing page with project KPIs
│   ├── pages/
│   │   ├── 1_RQ1_Category_Analysis.py
│   │   ├── 2_RQ2_Customer_Segments.py
│   │   ├── 3_RQ3_Predictive_Model.py
│   │   └── 4_RQ4_Behavioral_Associations.py
│   └── requirements.txt        # Dashboard dependencies
├── data/
│   ├── raw/                    # Source CSV files (not tracked)
│   └── processed/              # Parquet/CSV outputs (tracked)
│       ├── rq1/                # RQ1 statistical summaries
│       └── rq2/                # RQ2 cluster artifacts
├── figures/                    # Generated visualizations
│   ├── rq1/                    # 7 RQ1 category/brand figures
│   ├── rq2/                    # RQ2 clustering & concentration figures
│   └── rq4/                    # RQ4 regression diagnostic figures
├── reports/
│   ├── rq3/                    # RQ3 model & validation artifacts
│   └── rq4/                    # RQ4 regression & validation artifacts
├── notebooks/                  # Jupyter analysis notebooks
│   ├── profit_erosion_analysis.ipynb   # MASTER pipeline (all RQs)
│   ├── rq3_predictive_modeling.ipynb
│   ├── rq3_ssl_validation.ipynb
│   ├── rq4_behavioral_associations.ipynb
│   └── rq4_ssl_validation.ipynb
├── src/                        # Python source modules (flat, no sub-packages)
│   ├── config.py               # Path constants and thresholds
│   ├── data_processing.py      # ETL pipeline and data loading
│   ├── data_cleaning.py        # Data quality validation
│   ├── feature_engineering.py  # Profit erosion feature creation
│   ├── analytics.py            # Analysis, segmentation, validation
│   ├── visualization.py        # Shared plotting functions
│   ├── descriptive_transformations.py
│   ├── rq1_stats.py            # Kruskal-Wallis + Dunn post-hoc
│   ├── rq3_modeling.py         # ML pipeline: screening, training, evaluation
│   ├── rq3_visuals.py          # ROC curves, feature importance, confusion matrices
│   ├── rq3_validation.py       # SSL external validation pipeline
│   ├── rq4_econometrics.py     # Log-linear OLS regression pipeline
│   ├── rq4_validation.py       # RQ4 SSL validation
│   └── rq4_visuals.py          # Coefficient forest plot, residual diagnostics
├── tests/                      # pytest unit tests
│   ├── test_rq3_modeling.py
│   ├── test_rq3_validation.py
│   ├── test_rq4_econometrics.py
│   ├── test_rq4_validation.py
│   └── test_rq4_visuals.py
├── docs/                       # Technical documentation
│   ├── DATA_DICTIONARY.md
│   ├── PROCESSING_COST_METHODOLOGY.md
│   ├── rq1_technical_documentation.md
│   ├── rq3_technical_documentation.md
│   └── validation_module_reference.md
├── .github/workflows/ci.yml    # GitHub Actions CI
├── CLAUDE.md                   # Claude Code development instructions
├── pytest.ini
└── requirements.txt
```

---

## Cost Model

**Base cost: $12.00 per return** (conservative mid-range of $10–$25 literature range)

| Component | Amount |
|-----------|--------|
| Customer Care | $4.00 |
| Inspection | $2.50 |
| Restocking | $3.00 |
| Logistics | $2.50 |

**Category tier multipliers** (justified by margin CV = 59.4% across categories):

| Tier | Multiplier | Effective Cost | Categories |
|------|------------|----------------|------------|
| Premium | 1.3× | $15.60 | Outerwear, Jeans, Suits, Dresses, Sweaters |
| Moderate | 1.15× | $13.80 | Active, Swim, Accessories, Sleep & Lounge |
| Standard | 1.0× | $12.00 | Tops & Tees, Intimates, Socks, Underwear |

---

## Getting Started

### Prerequisites

- Python 3.11+
- Git

### Setup

```bash
git clone https://github.com/your-org/unfc-capstone-project.git
cd unfc-capstone-project

python -m venv venv
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

pip install -r requirements.txt
```

### Run the master notebook

```bash
jupyter notebook notebooks/profit_erosion_analysis.ipynb
```

Run cells top-to-bottom. Each section is guarded with `try/except` so partial runs are safe.

### Run tests

```bash
pytest tests/ -v
```

### Launch the dashboard

```bash
cd app
pip install -r requirements.txt
streamlit run Home.py
```

---

## Analysis Notebooks

| Notebook | Purpose |
|----------|---------|
| `profit_erosion_analysis.ipynb` | **Master pipeline** — validates all RQs end-to-end |
| `rq3_predictive_modeling.ipynb` | RQ3 standalone: feature screening → train → evaluate |
| `rq3_ssl_validation.ipynb` | RQ3 external validation against SSL data |
| `rq4_behavioral_associations.ipynb` | RQ4 OLS regression with diagnostics |
| `rq4_ssl_validation.ipynb` | RQ4 external validation against SSL data |

---

## RQ3 — Known Results

| Model | CV AUC | Test AUC | F1 | Precision | Recall |
|-------|--------|----------|----|-----------|--------|
| **Random Forest ★** | 0.9792 | **0.9798** | 0.8419 | 0.7822 | 0.9115 |
| Gradient Boosting | 0.9797 | 0.9795 | 0.8484 | 0.7801 | 0.9299 |
| Logistic Regression | 0.9646 | 0.9687 | 0.8256 | 0.7591 | 0.9048 |

Surviving features (7/12): `return_frequency`, `avg_order_value`, `avg_basket_size`, `total_margin`, `avg_item_margin`, `total_items`, `customer_return_rate`

**SSL External Validation**: Directional accuracy = 76.4%, Spearman ρ = 0.75 (p ≈ 0.00)

---

## Continuous Integration

GitHub Actions runs the full pytest suite on every PR to `main` and `dev`. PRs require passing checks before merge.

---

## Team

| Name | Student ID | Primary RQ |
|------|------------|-----------|
| Mario Zamudio | NF1002499 | RQ1 & RQ2 |
| Joseph Kojo Foli | NF1007842 | RQ3 & RQ4 |
| Avinash Brandon Maharaj | NF1002706 | RQ2 |
| Roberto San Miguel | NF1001332 | RQ1 |

---

*Data: `bigquery-public-data.thelook_ecommerce` | External validation: School Specialty LLC (SSL) 2024–2025*
*Academic use only — University of Niagara Falls Canada*
