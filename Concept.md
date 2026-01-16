Capstone Problem Statement (Common Across Both Data Sources)
Our proposed capstone focuses on profit erosion driven by product returns and post-transaction credits in e-commerce. While much of the existing analytics literature and applied work emphasizes return rates, customer satisfaction, or operational handling of returns, our project reframes returns as an economic problem. Specifically, we aim to quantify:
•	Margin reversal on returned items using observed sale price and product cost, and
•	Incremental profit erosion after incorporating return process costs (e.g., customer care, inspection, unpacking, shelving/restocking).
Our literature review indicates that while returns and reverse logistics are well studied, explicit integration of margin reversal and operational return costs as a unified profit erosion construct is limited, which differentiates our work from prior studies.

Data Source Option 1 (Preferred): BigQuery – thelook_ecommerce
Rationale & Strengths
•	Publicly available via Google BigQuery and not Kaggle-hosted, aligning with your recommendation.
•	Explicit product cost and sale price fields enable direct margin computation without assumptions.
•	Clear item-level return indicators allow clean identification of economic reversals.
•	Includes customer-level attributes and transactional history, enabling behavioral analysis through revealed behavior (e.g., repeat purchasing, repeat returns, discount exposure, channel/traffic source).
•	Although the dataset is synthetic, we did not generate it, nor did we fabricate any data; we are strictly using a publicly maintained dataset.
Considerations
•	Limited textual or sentiment-based behavioral signals.
•	Return timestamps are sparsely populated, which led us to intentionally avoid return-latency modeling and instead focus on economic exposure, reinforcing methodological discipline.

Data Source Option 2: Olist Brazilian E-Commerce Dataset (Kaggle)
Rationale & Strengths
•	Rich customer and transactional structure with explicit orders and order items.
•	Includes customer-level information and review data, enabling sentiment-based behavioral analysis.
•	While widely used for descriptive analytics, forecasting, and satisfaction studies, it has not been commonly applied to profit erosion modeling that integrates margin reversal and return process costs.
Concern
•	The dataset is hosted on Kaggle, which you advised the class to avoid where possible due to widespread reuse, despite the novelty of our framing and methodology.

Why Our Research Stands Out
Across both datasets, our work differs from existing analytics projects by:
•	Avoiding return-rate prediction as the primary objective.
•	Centering analysis on profit erosion, combining margin reversal with modeled return-processing costs.
•	Using customer-level revealed transactional behavior (pricing, repetition, aggregation) rather than relying solely on sentiment or synthetic labels.
•	Explicitly grounding assumptions and validating data limitations rather than imputing or generating synthetic behavior.

Request for Your Guidance
We would appreciate your guidance on the following, with BigQuery thelook_ecommerce as our preferred option:
1.	Whether proceeding with the BigQuery thelook_ecommerce dataset is appropriate, given its customer-level coverage, alignment with the problem statement, and your guidance on avoiding Kaggle and synthetic data generation.
2.	Whether the Olist dataset would still be acceptable as an alternative, given that our problem framing and analytical contribution differ materially from typical Kaggle-based analyses.
