# RQ1 SSL Validation Reference
**Capstone Project - Master of Data Analytics**  
**External Validation Dataset: `SSL_Returns_df_yoy.csv`**  
**Scope: RQ1 category validation only**

---

## 1. Purpose

This document defines the field mappings used to validate the RQ1 profit erosion pipeline on the SSL dataset.

The goal is directional and structural validation:

1. Confirm that `profit_erosion` can be produced from SSL data using a consistent economic-loss logic.
2. Confirm that category-level hypothesis testing on SSL leads to the same qualitative conclusion as the TheLook RQ1 workflow.

Brand and department are not validated independently.

---

## 2. Unit of Analysis and Filtering

- Unit of analysis: returned order line
- TheLook: filtered to returned items
- SSL: operational returns extract where rows represent return-related activity

Because SSL is not a full purchase population, any return-rate style visual should be interpreted as return composition rather than a directly comparable retail return rate.

---

## 3. SSL Category Construction

The canonical SSL category is constructed as:

```text
category = Pillar + "-" + Major Market Cat + "-" + Department
```

Missing values are filled with `"Unknown"` before concatenation.

---

## 4. Canonical RQ1 Fields

| Canonical field | Meaning | Used in |
|---|---|---|
| `order_id` | Unique order identifier | grouping and checks |
| `order_line_id` | Unique line identifier | item grain |
| `category` | Composite product category | hypothesis tests and visuals |
| `profit_erosion` | Loss per returned item | tests and visuals |
| `is_returned_item` | Return indicator | filtering |
| `date` | Transaction date | optional slicing |

---

## 5. Field Mappings: TheLook to SSL

| Canonical field | TheLook field | SSL field(s) | Notes |
|---|---|---|---|
| `order_id` | `order_id` | `Order Number` | Cast to string |
| `order_line_id` | `order_item_id` | `Order Line ID` | Cast to string |
| `category` | `category` | `Pillar + Major Market Cat + Department` | Hyphen-concatenated |
| `profit_erosion` | engineered `profit_erosion` | `total_loss` | Use absolute value |
| `is_returned_item` | return status flag | `Returns` | SSL is return-focused |
| `date` | `created_at` / `shipped_at` | `Booked Date` / `Billed Date` | Prefer `Booked Date` |

---

## 6. Profit Erosion Definition in SSL

TheLook models profit erosion as margin reversal plus processing cost.

For SSL validation:

- `profit_erosion = abs(total_loss)`

This is acceptable for validation because it captures realized financial loss at the returned-line level.

---

## 7. Known Limitations

1. SSL does not provide a full non-return population, so return-rate comparisons are limited.
2. The SSL composite category is more granular than TheLook categories.
3. `total_loss` may include accounting adjustments not present in the TheLook modeled cost structure.
4. Brand-level validation is intentionally out of scope.

---

## 8. Validation Artifacts Produced

The RQ1 SSL workflow now follows the same output discipline as the master notebook:

**Processed data (`data/processed/rq1_ssl/`):**

- `rq1_ssl_engineered.parquet`
- `rq1_ssl_returned_items.parquet`
- `rq1_ssl_base_canonical.parquet`

**Report CSVs (`reports/rq1_ssl/`):**

- `rq1_ssl_by_category.csv`
- `rq1_ssl_test_summary_category.csv`
- `rq1_ssl_posthoc_category.csv`
- `rq1_ssl_bootstrap_ci_category_mean.csv`

**Figures (`figures/rq1_ssl/`):**

- `fig1_top_categories_total_erosion.png`
- `fig2_return_rate_vs_mean_erosion_category.png`
- `fig3_severity_vs_volume_category.png`
- `fig4_profit_erosion_distribution_log.png`
- `fig5_bootstrap_ci_category_mean.png`

---

## References

- Looker. *TheLook eCommerce dataset*.
- School Specialty, Inc. *SSL_Returns_df_yoy*.
- Conover, W. J. *Practical Nonparametric Statistics*.
- Efron, B., and Tibshirani, R. J. *An Introduction to the Bootstrap*.
