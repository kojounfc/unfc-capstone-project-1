# Capstone Project – Profit Erosion in E-commerce

## Overview

This project investigates **profit erosion driven by product returns** in e-commerce. While much of the existing analytics literature emphasizes return rates or customer satisfaction, our work reframes returns as an economic problem by quantifying:

- **Margin reversal** on returned items using observed sale price and product cost
- **Incremental profit erosion** after incorporating return process costs (customer care, inspection, restocking, logistics)

## Problem Statement

Our capstone focuses on profit erosion driven by product returns and post-transaction credits. We aim to:
1. Quantify margin reversal on returned items
2. Model incremental profit erosion from return processing costs
3. Identify customer and product segments with high return exposure
4. Provide actionable recommendations using analytics

## Data Source

**BigQuery Public Dataset**: `bigquery-public-data.thelook_ecommerce`

### Why This Dataset
- Publicly available via Google BigQuery (not Kaggle-hosted)
- Explicit product cost and sale price fields enable direct margin computation
- Clear item-level return indicators allow identification of economic reversals
- Customer-level attributes enable behavioral analysis

### Tables Used
| Table | Description |
|-------|-------------|
| `order_items` | Individual items within orders (grain level) |
| `orders` | Order-level information |
| `products` | Product catalog with cost and pricing |
| `users` | Customer demographics and acquisition |

## Repository Structure

```
unfc-capstone-project/
├── data/
│   ├── raw/              # Source CSV files from BigQuery (not tracked)
│   └── processed/        # Merged and engineered parquet files (tracked)
├── figures/              # Generated charts and visualizations
├── notebooks/            # Jupyter notebooks for EDA and analysis
├── reports/              # Project proposal and final report
├── src/                  # Python modules
│   ├── __init__.py
│   ├── config.py         # Configuration constants and paths
│   ├── data_cleaning.py  # Data cleaning functions
│   ├── data_processing.py # Data loading, cleaning, merging
│   ├── visualization.py  # Plotting functions
│   └── modeling.py       # Profit erosion analysis functions
├── tests/                # Unit tests (pytest)
├── docs/                 # Technical documentation
├── .github/
│   └── workflows/
│       └── ci.yml        # GitHub Actions CI workflow
├── .gitignore
├── Branching_Strategy_Capstone.md
├── CONTRIBUTING.md
├── README.md
├── pytest.ini            # Pytest configuration
└── requirements.txt
```

## Tools & Technologies

- **Python 3.11** (Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn)
- **Jupyter Notebook** for exploratory analysis
- **BigQuery** for data extraction
- **PyArrow** for parquet file handling
- **pytest** for test-driven development
- **GitHub Actions** for continuous integration (automated testing on PRs)
- **GitHub** for version control
- **Power BI / Tableau** for dashboards (where applicable)

## Getting Started

### Prerequisites
- Python 3.11 or higher
- Git

### Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-org/unfc-capstone-project.git
   cd unfc-capstone-project
   ```

2. **Create virtual environment**:
   ```bash
   python -m venv venv
   # Windows:
   venv\Scripts\activate
   # Linux/Mac:
   source venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the analysis**:
   ```bash
   jupyter notebook notebooks/the_look_ecom_EDA.ipynb
   ```

5. **Run tests**:
   ```bash
   pytest tests/ -v
   ```

## Usage

### Using the src modules in notebooks:

```python
# Data processing
from src.data_processing import load_processed_data, build_analysis_dataset

# Load existing processed data
df = load_processed_data()

# Or rebuild from raw files
df = build_analysis_dataset()
```

## RQ1 — Return Drivers & Profit Erosion

Run RQ1 artifact generation (writes to `data/processed/rq1/`):

```bash
python -m src.rq1_run
```

Run RQ1 statistical tests (reads `rq1_returned_items.parquet` and writes summaries):

```bash
python -m src.rq1_stats
```

Generate RQ1 figures (writes to `figures/rq1/`):

```bash
python -m src.rq1_visuals
```

## RQ3 — Predictive Modeling (High Profit Erosion Customers)

Run the full RQ3 modeling pipeline (feature screening, train/evaluate 3 models, export results to `reports/rq3/`):

```bash
jupyter notebook notebooks/rq3_predictive_modeling.ipynb
```

Run external validation against School Specialty LLC data (requires `data/raw/SSL_Returns_df_yoy.csv`):

```bash
jupyter notebook notebooks/rq3_ssl_validation.ipynb
```

## RQ4 — Behavioral Associations with Profit Erosion

Interactive econometric analysis with data exploration and model diagnostics:

```bash
jupyter notebook notebooks/rq4_behavioral_associations.ipynb
```

Run external validation against School Specialty LLC data (requires `data/raw/SSL_Returns_df_yoy.csv`):

```bash
jupyter notebook notebooks/rq4_ssl_validation.ipynb
``

```python
# Visualization
from src.visualization import plot_margin_distribution, plot_return_rate_by_category

plot_margin_distribution(df, returned_only=True)
plot_return_rate_by_category(df, top_n=15)

# Profit erosion metrics (returned items only)
from src.feature_engineering import summarize_profit_erosion, engineer_customer_behavioral_features

returned_df = df[df["is_returned_item"] == 1].copy()
summary = summarize_profit_erosion(returned_df)
customer_features = engineer_customer_behavioral_features(df)

# Customer segmentation
from src.analytics import segment_customers_by_return_behavior

customer_segments = segment_customers_by_return_behavior(df)
```

## Continuous Integration

This project uses **GitHub Actions** for automated testing. The CI workflow:

- Runs automatically on all pull requests to `main` and `dev` branches
- Executes the full test suite using pytest
- Ensures code quality before merging

Pull requests require passing status checks before the merge button is enabled.

## Status

**In Progress** – Winter 2026 Capstone Project
University of Niagara Falls Canada

## Team

| Name | Student ID | Email |
|------|------------|-------|
| Mario Zamudio | NF1002499 | mario.zamudio2499@myunfc.ca |
| Joseph Kojo Foli | NF1007842 | joseph.foli7842@myunfc.ca |
| Avinash Brandon Maharaj | NF1002706 | avinash.maharaj2706@myunfc.ca |
| Roberto San Miguel | NF1001332 | roberto.san1332@myunfc.ca |

## License

This project is for academic purposes as part of the UNFC Data Analytics Capstone.
