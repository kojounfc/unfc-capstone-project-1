# Data Cleaning Technical Report

## Executive Summary

This document provides a comprehensive technical overview of the data cleaning procedures implemented in the Profit Erosion E-commerce Capstone Project. The cleaning process applies a multi-layered validation approach using both removal and flagging strategies to maintain data integrity while preserving complete datasets for analysis.

---

## 1. Data Cleaning Steps

The data cleaning process follows a sequential, modular approach with nine major steps, executed in a specific order to ensure logical consistency and optimal data quality:

### 1.1 Step 1: Duplicate Detection and Removal
**Purpose:** Eliminate exact duplicate records that represent the same transaction recorded multiple times.

**Method:**
- Identifies rows that are completely identical across all specified columns
- Removes duplicates by keeping only the first occurrence
- Provides detailed statistics on the number of duplicate records found

**Outcome:** Ensures each transaction appears only once in the dataset, preventing artificial inflation of record counts.

---

### 1.2 Step 2: Missing Values Analysis and Handling
**Purpose:** Identify and document missing values while preserving data integrity.

**Method:**
- Analyzes all columns to identify missing data
- Generates comprehensive statistics including:
  - Count of missing values per column
  - Percentage of missing values relative to total rows
  - List of columns with missing data

**Handling Strategy:** Uses the **"report"** strategy (see Section 2 for detailed rationale)
- Missing values are documented but not removed
- Allows data science team to analyze missing patterns
- Preserves complete records for context-aware analysis

**Outcome:** Complete visibility into data quality issues without data loss.

---

### 1.3 Step 3: Outlier Detection
**Purpose:** Identify statistical anomalies that may represent data errors or edge cases.

**Method:** Uses the **Interquartile Range (IQR) Method** with a 1.5 multiplier (see Section 2 for detailed rationale)
- Calculates Q1 (25th percentile) and Q3 (75th percentile) for each numeric column
- Computes IQR = Q3 - Q1
- Defines bounds as:
  - Lower bound: Q1 - (1.5 × IQR)
  - Upper bound: Q3 + (1.5 × IQR)
- Flags values outside these bounds as outliers

**Handling Action:** Outliers are flagged but not automatically removed
- Adds `is_outlier` flag column for manual review
- Allows business context assessment before removal
- Preserves potentially valid extreme values

**Outcome:** Data scientists can evaluate outliers contextually before deciding to exclude them.

---

### 1.4 Step 4: Price Consistency Validation
**Purpose:** Ensure logical relationships between different price points.

**Validation Rules:**
- All prices must be non-negative (≥ 0)
- Sale price must not exceed retail price
- Cost must not exceed sale price
- Logical hierarchy: Retail price ≥ Sale price ≥ Cost ≥ 0

**Issues Detected:**
- Negative prices
- Sale price exceeding retail price
- Cost exceeding sale price

**Handling Action:** Inconsistencies are flagged with `has_price_inconsistency` column
- Allows review of business rules violations
- Enables investigation of data entry errors
- Preserves records for validation team assessment

**Outcome:** Identifies economically illogical transactions without automatic removal.

---

### 1.5 Step 5: Status Consistency Validation
**Purpose:** Verify logical consistency in order and item status flags.

**Validation Rules:**
- Returned items must have a "returned" status
- Delivered items must have been shipped
- Item return status must match order return status
- Status progression must follow business rules

**Issues Detected:**
- Items flagged as returned but lacking "returned" status
- Items marked delivered without shipment records
- Item and order status mismatches

**Handling Action:** Inconsistencies are flagged with `has_status_inconsistency` column
- Enables workflow validation
- Identifies status synchronization problems
- Helps resolve cross-table logical inconsistencies

**Outcome:** Ensures order workflow integrity while preserving records for investigation.

---

### 1.6 Step 6: Temporal Consistency Validation
**Purpose:** Verify that timestamps follow logical temporal sequences.

**Validation Rules:**
- Item delivery timestamp must be after shipment timestamp
- Item return timestamp must be after delivery timestamp
- Temporal order: Shipped → Delivered → Returned

**Issues Detected:**
- Items delivered before shipment
- Items returned before delivery
- Impossible temporal sequences

**Handling Action:** Inconsistencies are flagged with `has_temporal_inconsistency` column
- Enables timestamp quality assessment
- Identifies timezone or data entry issues
- Preserves records for correction

**Outcome:** Validates event sequence logic without data loss.

---

### 1.7 Step 7: Categorical Values Cleaning
**Purpose:** Standardize categorical data values for consistency and eliminate redundant variations.

**Cleaning Operations:**
- Converts all categorical values to lowercase (optional, configurable)
- Strips leading and trailing whitespace (optional, configurable)
- Eliminates duplicate values created by case or spacing variations
- Applies to all object/string/category type columns

**Outcome:** Improved categorical data consistency and reduced unique value counts through value normalization.

---

### 1.8 Step 8: Low Variance Column Removal
**Purpose:** Remove columns with insufficient variability for analysis.

**Method:**
- Calculates standard deviation for numeric columns
- Removes columns with variance below the threshold (default: 0.01)
- Eliminates near-constant columns that provide minimal analytical value

**Handling:** Configurable via `remove_low_variance` parameter in `perform_deep_clean()`

**Outcome:** Reduces dimensionality by removing non-informative columns (optional cleanup step).

---

## 2. Treatment of Missing Values and Outliers

### 2.1 Missing Values Handling

**Strategy: Report-Only Approach**

The data cleaning module uses a **"report"** strategy for missing values, which means:
- Missing values are identified and documented
- No rows are removed based on missing data
- Comprehensive statistics are generated for each column with missing values

**Rationale:**

1. **Preserve Data Context:** E-commerce data often contains meaningful patterns in missingness. For example:
   - Returned items may have differently populated fields than shipped items
   - Optional fields may be legitimately empty
   - Missing values can indicate specific business conditions

2. **Enables Domain-Specific Decision Making:** 
   - Data scientists can analyze missing patterns to determine if they're random or systematic
   - Allows targeted imputation strategies based on business logic rather than generic approaches
   - Enables recovery of useful information from incomplete records

3. **Prevents Information Loss:**
   - Rows with missing values in one column may have complete information in other critical columns
   - Removes rows only removes entire observations when partial information remains valuable
   - Example: A product return record may lack `created_at` but have complete return timing information

4. **Supports Exploratory Data Analysis:**
   - Missing data patterns themselves can be analytically valuable
   - Enables investigation of whether missingness is informative for the profit erosion analysis
   - Facilitates root cause analysis of data quality issues

**Implementation:**
- Function: `handle_missing_values(df, strategy="report")`
- Report includes:
  - Count and percentage of missing values per column
  - List of all columns with missing data
  - Total missing cells in the dataset

---

### 2.2 Outlier Detection and Handling

**Method: Interquartile Range (IQR) with 1.5 Multiplier**

Outliers are detected using the IQR method with a multiplier of 1.5, and are **flagged rather than removed**.

**Technical Specification:**
```
For each numeric column (excluding flag columns and derived metrics):
  Q1 = 25th percentile
  Q3 = 75th percentile
  IQR = Q3 - Q1
  
  Lower Bound = Q1 - (1.5 × IQR)
  Upper Bound = Q3 + (1.5 × IQR)
  
  Outliers: values < Lower Bound OR > Upper Bound
```

**Columns Excluded from Outlier Detection:**
- Flag columns (is_*, has_*)
- Derived metric columns (contains "margin" or "discount")
- ID columns (already converted to string type)

**Outlier Documentation:**
When `action="flag"`, the function creates three columns:
1. **`is_outlier`** - Binary flag (1 = outlier, 0 = normal)
2. **`outlier_columns`** - Semicolon-separated list of columns flagged as outliers (e.g., "cost;sale_price")
3. **`outlier_values`** - Semicolon-separated column=value pairs (e.g., "cost=5.50;sale_price=1999.99")

This detailed tracking enables:
- Identification of exactly which columns triggered the outlier flag per row
- Review of the actual outlier values
- Targeted investigation and validation
- Informed decision-making on whether to keep or exclude specific outliers

**Rationale for IQR Method with 1.5 Multiplier:**

1. **Statistical Justification:**
   - The 1.5 × IQR rule is a widely accepted statistical standard in data analysis
   - Approximately captures 0.35% of data in a normal distribution as outliers
   - Balances sensitivity to true anomalies while avoiding over-flagging
   - More robust than standard deviation methods when data is skewed

2. **Appropriate for E-commerce Data:**
   - E-commerce returns and order values inherently follow non-normal distributions
   - High-value orders, bulk returns, and edge cases are legitimate business scenarios
   - The 1.5 multiplier captures statistical extremes while acknowledging valid business variance
   - Avoids bias against legitimate high-value or high-volume transactions

3. **Business Value Preservation:**
   - High-value customers and transactions are often the most important for profitability analysis
   - Outliers in return rates may indicate systematic issues worth investigating
   - Removing outliers could bias profit erosion analysis toward average transactions
   - Example: A bulk return of 100 units may be an outlier statistically but crucial for understanding erosion patterns

4. **Flagging vs. Removal:**
   - Outliers are added to a flag column (`is_outlier`) rather than removed
   - Allows analysts to:
     - Investigate outliers separately
     - Determine if they represent errors or legitimate extreme cases
     - Run analyses both with and without outliers
     - Make informed decisions on a case-by-case basis

**Implementation:**
- Function: `detect_outliers_iqr(df, multiplier=1.5, action="flag")`
- Output: `is_outlier` flag column marking each outlier row

---

## 3. Data Preservation and Flagging Strategy

### 3.1 Non-Destructive Validation Approach

The cleaning module uses a **flagging-first philosophy** to preserve all data while marking quality issues:

**Flagged Issues (Preserved in Dataset):**
- **Outliers**: `is_outlier`, `outlier_columns`, `outlier_values`
- **Price Inconsistencies**: `has_price_inconsistency`
- **Status Inconsistencies**: `has_status_inconsistency`
- **Temporal Inconsistencies**: `has_temporal_inconsistency`

All flagged records are retained in the main analysis dataset. A separate `data_to_review.csv` file is generated containing only the flagged records with their validation columns, allowing:
- Separate analysis of data quality issues
- Investigation of specific problems before removal
- Validation of cleaning decisions
- Sensitivity analysis (with/without flagged records)

**Only Removed Issues:**
- Exact duplicate records (`remove_duplicates=True`)
- All other removals are optional and configurable per validation type

### 3.2 Dataset Output Strategy

The `save_cleaned_dataset()` function implements a dual-output approach:

**Main Analysis Dataset** (`returns_eda_v1.parquet` / `returns_eda_v1.csv`)
- Contains: ALL records (even those with validation flags)
- Excludes: Validation flag columns only
- Purpose: Complete dataset for analysis with clean column structure
- Benefit: Analysts can choose to filter problematic records themselves

**Review Dataset** (`data_to_review.csv`)
- Contains: Only records with at least one validation flag
- Includes: `order_item_id` and all validation flag columns
- Purpose: Data quality audit trail
- Benefit: Easy identification of problematic records for investigation

This approach provides maximum flexibility while maintaining data integrity and traceability.

---

## 4. Implementation Integration

All cleaning steps are integrated into the `perform_deep_clean()` function, which provides:

- Flexible configuration of each cleaning step
- Parameterizable actions (flag vs. remove) for each validation type
- Comprehensive cleaning report with statistics on all operations
- Modular design allowing selective activation of specific validations

**Standard Configuration Used:**
```python
perform_deep_clean(
    df=raw_data,
    remove_duplicates=True,              # Step 1: Remove exact duplicates
    handle_missing="report",             # Step 2: Report only (no removal)
    detect_outliers=True,                # Step 3: Flag outliers
    validate_prices=True,                # Step 4: Flag price issues
    validate_status=True,                # Step 5: Flag status issues
    validate_temporal=True,              # Step 6: Flag temporal issues
    clean_categories=True,               # Step 7: Clean categorical values
    remove_low_variance=False,           # Step 8: Keep all columns
    outlier_action="flag",               # Outlier action (flag or remove)
    price_action="flag",                 # Price action (flag or remove)
    status_action="flag",                # Status action (flag or remove)
    temporal_action="flag",              # Temporal action (flag or remove)
)
```

**Key Functions:**

| Function | Purpose | Input | Output |
|----------|---------|-------|--------|
| `detect_and_handle_duplicates()` | Find and remove/flag exact duplicate rows | DataFrame, action | (DataFrame, report) |
| `handle_missing_values()` | Analyze missing data with configurable handling | DataFrame, strategy | (DataFrame, report) |
| `detect_outliers_iqr()` | Detect outliers using IQR method with detailed tracking | DataFrame, multiplier, action | (DataFrame with `is_outlier`, `outlier_columns`, `outlier_values`, report) |
| `validate_price_consistency()` | Check price logical hierarchy (retail ≥ sale ≥ cost ≥ 0) | DataFrame, action | (DataFrame with `has_price_inconsistency`, report) |
| `validate_status_consistency()` | Check status workflow consistency | DataFrame, action | (DataFrame with `has_status_inconsistency`, report) |
| `validate_temporal_consistency()` | Check timestamp ordering (shipped < delivered < returned) | DataFrame, action | (DataFrame with `has_temporal_inconsistency`, report) |
| `clean_categorical_values()` | Standardize categorical data (lowercase, whitespace removal) | DataFrame, columns, options | (DataFrame, report) |
| `remove_low_variance_columns()` | Remove near-constant columns | DataFrame, threshold | (DataFrame, report) |
| `perform_deep_clean()` | Orchestrate all cleaning steps with configurable actions | DataFrame, parameters | (DataFrame, comprehensive report) |
| `save_cleaned_dataset()` | Save main dataset (all rows, no flags) + review dataset (flagged rows only) | DataFrame | Main parquet/csv + data_to_review.csv |

---

## 5. Data Quality Assurance

The cleaning process generates comprehensive reports for each step, enabling:
- Transparency in data modifications
- Ability to audit each cleaning decision
- Identification of data quality patterns
- Justification for data removal or modification

All flagged records are retained in the dataset with indicator columns, allowing:
- Separate analysis of flagged vs. clean records
- Investigation of specific data quality issues
- Validation of cleaning decisions
- Sensitivity analysis with and without flagged records

---

## Conclusion

The data cleaning strategy balances preservation of data integrity with retention of complete information. By using flagging for subjective issues (outliers, missing values) and removal only for objective impossibilities (impossible coordinates, duplicates), we maintain data richness while ensuring analytical reliability.
