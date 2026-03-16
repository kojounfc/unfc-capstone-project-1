# Contributing to Profit Erosion E-commerce Capstone

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Workflow](#development-workflow)
- [Commit Message Conventions](#commit-message-conventions)
- [Code Style Guidelines](#code-style-guidelines)
- [Testing Requirements](#testing-requirements)
- [Pull Request Process](#pull-request-process)
- [Project Structure](#project-structure)

## Code of Conduct

This project follows a code of conduct that emphasizes:
- **Respect**: Treat all contributors with respect and kindness
- **Collaboration**: Work together to build better solutions
- **Agile Practices**: Follow Agile principles and maintain traceability
- **Quality**: Write clean, modular, and documented code

## Getting Started

### Prerequisites

- Python 3.11 or higher
- Git
- GitHub account

### Setup Development Environment

1. **Clone the repository**:
   ```bash
   git clone https://github.com/kojounfc/unfc-capstone-project-1.git
   cd unfc-capstone-project-1
   ```

2. **Create a virtual environment**:
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On Linux/Mac:
   source venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. **Install development tools**:
   ```bash
   pip install black isort pytest pytest-cov
   ```

5. **Verify setup**:
   ```bash
   pytest tests/ -v
   ```

## Development Workflow

### 1. Create a Feature Branch

**IMPORTANT**: Always create a new branch for your work.

```bash
git checkout dev
git pull origin dev
git checkout -b feat/your-feature-name
```

**Branch naming conventions**:
- `feat/` - New features (e.g., `feat/margin-analysis`, `feat/customer-segmentation`)
- `fix/` - Bug fixes (e.g., `fix/data-merge-issue`, `fix/margin-calculation`)
- `docs/` - Documentation updates
- `test/` - Adding or updating tests
- `refactor/` - Code refactoring
- `chore/` - Maintenance tasks

### 2. Make Your Changes

- Write clean, modular code following Python best practices
- Add docstrings to all functions and classes (Google style)
- Follow the project structure (see [Project Structure](#project-structure))
- Functions should return ready-to-use data objects or plot objects
- Never hardcode file paths - use `src/config.py` constants

### 3. Write Tests

**Every new feature must include tests**:

- Create test files in `tests/` matching the source structure
- Test file naming: `test_<module_name>.py`
- Use pytest conventions
- Aim for good test coverage

Example:
```python
# src/data_processing.py -> tests/test_data_processing.py
# src/visualization.py -> tests/test_visualization.py
```

### 4. Format Your Code

Before committing, format your code:

```bash
# Format with Black
black src/ tests/

# Sort imports with isort
isort src/ tests/

# Verify formatting
black --check src/ tests/
isort --check src/ tests/
```

### 5. Run Tests

**Always run tests before committing**:

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_data_processing.py -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

### 6. Commit Your Changes

Use conventional commit messages (see [Commit Message Conventions](#commit-message-conventions)):

```bash
git add .
git commit -m "feat(data-processing): add margin calculation function"
```

### 7. Keep Your Branch Updated

Regularly sync with dev branch:

```bash
git fetch origin
git rebase origin/dev
# Resolve any conflicts if needed
```

### 8. Push and Create Pull Request

```bash
git push origin feat/your-feature-name
```

Then create a Pull Request on GitHub targeting the `dev` branch.

## Commit Message Conventions

We follow [Conventional Commits](https://www.conventionalcommits.org/) specification.

### Format

```
<type>(<scope>): <description>

[optional body]

[optional footer]
```

### Types

- **`feat`**: New feature
- **`fix`**: Bug fix
- **`docs`**: Documentation changes
- **`test`**: Adding or updating tests
- **`refactor`**: Code refactoring (no feature change or bug fix)
- **`chore`**: Maintenance tasks, dependency updates
- **`style`**: Code style changes (formatting, missing semicolons, etc.)
- **`perf`**: Performance improvements

### Scope (Optional)

The scope should be the name of the module affected:
- `data-processing` - Data loading and transformation
- `visualization` - Plotting and charts
- `modeling` - Analysis and modeling functions
- `config` - Configuration changes
- `notebook` - Jupyter notebook updates
- `docs` - Documentation

### Examples

```bash
# Feature with scope
feat(data-processing): implement margin calculation function

# Bug fix
fix(visualization): resolve axis label overlap in heatmap

# Documentation
docs(readme): update installation instructions

# Test
test(modeling): add unit tests for customer segmentation

# Refactoring
refactor(data-processing): optimize data merge performance
```

## Code Style Guidelines

### Python Style

- Follow PEP 8 style guide
- Use Black for code formatting (line length: 88)
- Use isort for import sorting
- Maximum line length: 88 characters (Black default)

### Import Organization

```python
# Standard library imports
import os
from pathlib import Path
from typing import Dict, List, Optional

# Third-party imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Local imports
from src.config import RAW_DATA_DIR, PROCESSED_DATA_DIR
from src.data_processing import load_processed_data
```

### Docstrings

Use Google style docstrings:

```python
def calculate_margins(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate margin and discount metrics for profit erosion analysis.

    Args:
        df: Merged DataFrame with price and cost columns.

    Returns:
        DataFrame with margin and discount columns added.

    Raises:
        ValueError: If required columns are missing.
    """
    pass
```

### Type Hints

Always use type hints for function signatures:

```python
from typing import Dict, List, Optional, Tuple
import pandas as pd

def load_raw_data(
    raw_dir: Optional[Path] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    pass
```

## Testing Requirements

### Test Structure

- Tests mirror source code structure
- Each module has a corresponding test file in `tests/`
- Test classes group related tests
- Use descriptive test names

### Test Naming

```python
class TestDataProcessing:
    """Test cases for data processing functionality."""

    def test_clean_columns_removes_whitespace(self):
        """Test that clean_columns strips whitespace from column names."""
        pass

    def test_merge_datasets_validates_grain(self):
        """Test that merge validates many-to-one relationships."""
        pass
```

### Running Tests

```bash
# All tests
pytest

# Specific test file
pytest tests/test_data_processing.py

# Specific test
pytest tests/test_data_processing.py::TestDataProcessing::test_clean_columns

# With coverage
pytest --cov=src --cov-report=term-missing

# Verbose output
pytest -v

# Stop on first failure
pytest -x
```

## Pull Request Process

### Before Submitting

1. All tests pass locally
2. Code is formatted (Black + isort)
3. Documentation updated (if needed)
4. Commit messages follow conventions
5. Branch is up to date with `dev`

### Continuous Integration

When you create a pull request, **GitHub Actions** automatically:

1. Sets up Python 3.11 environment
2. Installs all dependencies from `requirements.txt`
3. Runs the full test suite using pytest

**Important**: The merge button will only be enabled after:
- All CI status checks pass (tests must pass)
- At least one peer review approval

### Review Process

1. **Create PR**: Push your branch and create a pull request to `dev`
2. **CI Runs**: Automated tests run via GitHub Actions
3. **Peer Review**: At least one team member must review your PR
4. **Address Feedback**: Make requested changes (CI re-runs on new commits)
5. **Status Check**: Ensure all tests pass before merging
6. **Merge**: Once approved and CI passes, the PR can be merged

### Merge Policy

- No direct commits to **main** or **dev**
- All feature branches are merged into **dev** via Pull Requests
- At least one peer review is required before merging
- The Scrum Lead merges **dev** → **main** after final review

## Project Structure

```
unfc-capstone-project/
├── app/                        # Streamlit dashboard
│   ├── Home.py
│   └── pages/
│       ├── 0_EDA.py
│       ├── 1_RQ1_Category_Analysis.py
│       ├── 2_RQ2_Customer_Segments.py
│       ├── 3_RQ3_Predictive_Model.py
│       └── 4_RQ4_Behavioral_Associations.py
├── data/
│   ├── raw/                    # Source CSV files (not tracked)
│   └── processed/              # Processed parquet/CSV files (tracked)
│       ├── rq1/
│       ├── rq1_ssl/
│       ├── rq2/
│       ├── rq3/
│       └── rq4/
├── figures/                    # Generated visualizations
│   ├── eda/
│   ├── rq1/  rq1_ssl/  rq2/  rq2_ssl/  rq3/  rq4/
├── reports/                    # CSV and JSON report artifacts
│   ├── rq1/  rq1_ssl/  rq2/  rq3/  rq4/
├── notebooks/
│   └── profit_erosion_analysis.ipynb   # Master notebook (sections 1–10)
├── src/                        # Python modules (flat — no sub-packages)
│   ├── config.py               # Path constants and thresholds
│   ├── data_processing.py      # ETL pipeline
│   ├── data_cleaning.py        # Data quality validation
│   ├── feature_engineering.py  # Profit erosion metrics
│   ├── model_ready_views.py    # Feature matrix construction
│   ├── analytics.py            # Segmentation and validation helpers
│   ├── visualization.py        # General plotting functions
│   ├── descriptive_transformations.py
│   ├── rq1_run.py  rq1_stats.py  rq1_ssl_preprocessing.py  rq1_ssl_validation.py
│   ├── rq2_run.py  rq2_segmentation.py  rq2_concentration.py
│   ├── rq3_modeling.py  rq3_sensitivity.py  rq3_visuals.py  rq3_validation.py
│   └── rq4_econometrics.py  rq4_validation.py  rq4_ssl_validation.py  rq4_visuals.py
├── tests/                      # pytest unit tests (512 tests)
├── docs/                       # Technical documentation per RQ
├── .github/
│   └── workflows/
│       └── ci.yml              # GitHub Actions CI workflow
├── .gitignore
├── CONTRIBUTING.md
├── README.md
├── pytest.ini
└── requirements.txt
```

## Getting Help

- **Documentation**: Check `README.md` and docstrings
- **Code**: Review existing code for patterns
- **Team**: Reach out to team members

---

By contributing, you agree that your contributions will be part of this academic capstone project.
