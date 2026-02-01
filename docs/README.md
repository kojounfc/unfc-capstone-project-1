# Technical Documentation

This folder contains technical documentation for the Profit Erosion E-commerce Capstone project.

## Contents

### Project Documentation

| Document | Description |
|----------|-------------|
| [MDA Capstone Project Proposal](MDA%20Capstone%20Project%20Proposal%20-%20Group%205%20-%20DAMO-699-4.pdf) | Project proposal submitted for DAMO-699-4 |
| [data_description.md](data_description.md) | Dataset source, structure, and suitability analysis |

### Data Documentation

| Document | Description |
|----------|-------------|
| [DATA_DICTIONARY.md](DATA_DICTIONARY.md) | Complete data dictionary with all features, formulas, and assumptions |
| [PROCESSING_COST_METHODOLOGY.md](PROCESSING_COST_METHODOLOGY.md) | Detailed methodology for return processing cost calculations |

### Module Technical References

| Document | Description |
|----------|-------------|
| [config.md](config.md) | Configuration module - paths, constants, and data types |
| [data_processing.md](data_processing.md) | Data processing module - ETL pipeline functions |
| [feature_engineering.md](feature_engineering.md) | Feature engineering module - return flags, margins, profit erosion |
| [analytics.md](analytics.md) | Analytics module - return rate analysis, customer segmentation, feature validation |
| [visualization.md](visualization.md) | Visualization module - plotting functions |

## Module Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        src/ Modules                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────┐     ┌───────────────────┐     ┌─────────────┐ │
│  │  config.py   │────▶│ data_processing.py│────▶│data_cleaning│ │
│  │  (constants) │     │    (ETL/merge)    │     │    .py      │ │
│  └──────────────┘     └───────────────────┘     └─────────────┘ │
│         │                      │                       │        │
│         │                      ▼                       │        │
│         │             ┌───────────────────┐            │        │
│         └────────────▶│feature_engineering│◀───────────┘        │
│                       │  (core features)  │                     │
│                       └───────────────────┘                     │
│                                │                                │
│                                ▼                                │
│                       ┌───────────────────┐                     │
│                       │   analytics.py    │                     │
│                       │ (analysis & agg)  │                     │
│                       └───────────────────┘                     │
│                                │                                │
│                                ▼                                │
│                       ┌───────────────────┐                     │
│                       │ visualization.py  │                     │
│                       │    (plotting)     │                     │
│                       └───────────────────┘                     │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Key Concepts

### Profit Erosion

Profit erosion is the total financial impact of a product return:

```
Profit_Erosion = Margin_Reversal + Processing_Cost
```

Where:
- **Margin Reversal**: `sale_price - cost` (the margin lost on the returned item)
- **Processing Cost**: Category-tiered cost ($12 base × tier multiplier)

### Processing Cost Tiers

| Tier | Multiplier | Effective Cost | Categories |
|------|------------|----------------|------------|
| Premium | 1.3x | $15.60 | Outerwear, Jeans, Suits, Dresses |
| Moderate | 1.15x | $13.80 | Active, Swim, Accessories |
| Standard | 1.0x | $12.00 | Tops & Tees, Intimates, Socks |

### Research Questions Supported

| RQ | Focus | Primary Module |
|----|-------|----------------|
| RQ1 | Profit erosion by category/brand | analytics.py |
| RQ2 | Customer segmentation | analytics.py |
| RQ3 | Predictive modeling | feature_engineering.py (targets) |
| RQ4 | Econometric regression | feature_engineering.py |

## Quick Links

- [Main Project README](../README.md)
- [CONTRIBUTING Guide](../CONTRIBUTING.md)
- [Source Code](../src/)
- [Tests](../tests/)
- [Notebooks](../notebooks/)
