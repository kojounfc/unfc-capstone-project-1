# Analyzing Profit Erosion from Product Returns in E-Commerce
### A Multi-Method Analytics Framework

**Course**: DAMO-699-4 Capstone Project - Winter 2026
**Institution**: University of Niagara Falls Canada

---

## Problem Statement

Product returns are economic reversal events that directly erode realized revenue and margin. This project reframes returns beyond operational metrics to quantify:

- Margin reversal: the margin lost on returned items (`sale_price - cost`)
- Incremental processing costs: customer service, inspection, restocking, and logistics (`$12` base per return, category-tier adjusted)

> **Core formula:** `Profit Erosion = Margin Reversal + Processing Cost`

---

## Research Questions

| RQ | Focus | Method | Key Result |
|----|-------|--------|------------|
| **RQ1** | Profit erosion differences across product categories and brands | Descriptive analysis + Kruskal-Wallis | Significant cross-category differences |
| **RQ2** | Customer behavioral segments with differential profit erosion | K-Means clustering + Gini/Lorenz/Pareto | Behaviorally distinct segments confirmed |
| **RQ3** | Predict high-erosion customers (target AUC > 0.70) | ML classification (RF, GB, LR) | RF champion AUC = 0.9798 |
| **RQ4** | Marginal associations between behaviors and profit erosion | Log-linear OLS regression | Significant behavioral predictors identified |

---

## Data Sources

**Primary**: `bigquery-public-data.thelook_ecommerce` (Google BigQuery)

| Table | Description |
|-------|-------------|
| `order_items` | Item-level transactions |
| `orders` | Order-level information |
| `products` | Product catalog with cost and pricing |
| `users` | Customer demographics and acquisition |

**External Validation**: School Specialty LLC (SSL), a U.S. educational supplies B2B retailer

- `data/raw/SSL_Returns_df_yoy.csv`
- Used for directional and structural validation of RQ1, RQ3, and RQ4 outputs

---

## Repository Structure

```text
unfc-capstone-project/
|-- app/                        # Streamlit dashboard
|   |-- Home.py
|   `-- pages/
|       |-- 1_RQ1_Category_Analysis.py
|       |-- 2_RQ2_Customer_Segments.py
|       |-- 3_RQ3_Predictive_Model.py
|       `-- 4_RQ4_Behavioral_Associations.py
|-- data/
|   |-- raw/
|   `-- processed/
|       |-- rq1/
|       |-- rq1_ssl/
|       |-- rq2/
|       |-- rq3/
|       `-- rq4/
|-- figures/
|   |-- rq1/
|   |-- rq1_ssl/
|   |-- rq2/
|   |-- rq3/
|   `-- rq4/
|-- reports/
|   |-- rq1/
|   |-- rq1_ssl/
|   |-- rq2/
|   |-- rq3/
|   `-- rq4/
|-- notebooks/
|   |-- profit_erosion_analysis.ipynb   # Master notebook: sections 1-10
|   |-- rq3_predictive_modeling.ipynb
|   |-- rq3_ssl_validation.ipynb
|   `-- rq4_ssl_validation.ipynb
|-- src/
|   |-- config.py
|   |-- data_processing.py
|   |-- data_cleaning.py
|   |-- feature_engineering.py
|   |-- analytics.py
|   |-- visualization.py
|   |-- descriptive_transformations.py
|   |-- rq1_stats.py
|   |-- rq1_ssl_preprocessing.py
|   |-- rq1_ssl_validation.py
|   |-- rq2_run.py
|   |-- rq3_modeling.py
|   |-- rq3_visuals.py
|   |-- rq3_validation.py
|   |-- rq4_econometrics.py
|   |-- rq4_validation.py
|   `-- rq4_visuals.py
|-- tests/
|-- docs/
|-- pytest.ini
`-- requirements.txt
```

---

## Notebook Structure

The main notebook is `notebooks/profit_erosion_analysis.ipynb` and is organized into the following top-level sections:

1. Setup
2. Data Loading
3. Data Cleaning and Baseline / Descriptive EDA
4. Feature Engineering
5. Descriptive and Group-Level Transformations
6. RQ1: Descriptive Analysis
7. RQ2: Customer Segmentation
8. RQ3: Predictive Modeling
9. RQ4: Econometric Analysis
10. Summary and Conclusions

Artifact discipline in the notebook:

- Parquet and model-ready datasets are written to `data/processed/<rq>`
- CSV report artifacts are written to `reports/<rq>`
- Figures are written to `figures/<rq>`
- SSL validation outputs for RQ1 are split into `data/processed/rq1_ssl`, `reports/rq1_ssl`, and `figures/rq1_ssl`

---

## Cost Model

**Base cost:** `$12.00` per return

| Component | Amount |
|-----------|--------|
| Customer Care | $4.00 |
| Inspection | $2.50 |
| Restocking | $3.00 |
| Logistics | $2.50 |

| Tier | Multiplier | Effective Cost | Categories |
|------|------------|----------------|------------|
| Premium | 1.3x | $15.60 | Outerwear, Jeans, Suits, Dresses, Sweaters |
| Moderate | 1.15x | $13.80 | Active, Swim, Accessories, Sleep and Lounge |
| Standard | 1.0x | $12.00 | Tops and Tees, Intimates, Socks, Underwear |

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

### Run the main notebook

```bash
jupyter notebook notebooks/profit_erosion_analysis.ipynb
```

Run cells top-to-bottom. The notebook writes section outputs to the `data/processed`, `reports`, and `figures` folders listed above.

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
| `profit_erosion_analysis.ipynb` | Master pipeline covering Sections 1-10 and all final RQ outputs |
| `rq3_predictive_modeling.ipynb` | RQ3 standalone modeling workflow |
| `rq3_ssl_validation.ipynb` | RQ3 external validation against SSL data |
| `rq4_ssl_validation.ipynb` | RQ4 external validation against SSL data |

---

## RQ3 Known Results

| Model | CV AUC | Test AUC | F1 | Precision | Recall |
|-------|--------|----------|----|-----------|--------|
| **Random Forest** | 0.9792 | **0.9798** | 0.8419 | 0.7822 | 0.9115 |
| Gradient Boosting | 0.9797 | 0.9795 | 0.8484 | 0.7801 | 0.9299 |
| Logistic Regression | 0.9646 | 0.9687 | 0.8256 | 0.7591 | 0.9048 |

Surviving features (7/12): `return_frequency`, `avg_order_value`, `avg_basket_size`, `total_margin`, `avg_item_margin`, `total_items`, `customer_return_rate`

**SSL external validation:** Directional accuracy = 76.4%, Spearman rho = 0.75

---

## Continuous Integration

GitHub Actions runs the pytest suite on pull requests to `main` and `dev`.

---

## Team

| Name | Student ID | Primary RQ |
|------|------------|------------|
| Mario Zamudio | NF1002499 | RQ1 and RQ2 |
| Joseph Kojo Foli | NF1007842 | RQ3 and RQ4 |
| Avinash Brandon Maharaj | NF1002706 | RQ2 |
| Roberto San Miguel | NF1001332 | RQ1 |

---

*Data: `bigquery-public-data.thelook_ecommerce` | External validation: School Specialty LLC (SSL) 2024-2025*  
*Academic use only - University of Niagara Falls Canada*
