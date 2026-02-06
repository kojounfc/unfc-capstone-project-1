from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

OUT_DIR = Path("reports/rq1")

def main():
    cat = pd.read_csv(OUT_DIR / "rq1_erosion_by_category.csv")
    brand = pd.read_csv(OUT_DIR / "rq1_erosion_by_brand.csv")

    # Fig 1: Top categories by total erosion
    cat_top = cat.sort_values("total_profit_erosion", ascending=False).head(15)
    plt.figure()
    plt.bar(cat_top["category"], cat_top["total_profit_erosion"])
    plt.xticks(rotation=70, ha="right")
    plt.title("Top Categories by Total Profit Erosion")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "fig1_top_categories_total_erosion.png", dpi=200)
    plt.close()

    # Fig 2: Top brands by total erosion
    brand_top = brand.sort_values("total_profit_erosion", ascending=False).head(15)
    plt.figure()
    plt.bar(brand_top["brand"], brand_top["total_profit_erosion"])
    plt.xticks(rotation=70, ha="right")
    plt.title("Top Brands by Total Profit Erosion")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "fig2_top_brands_total_erosion.png", dpi=200)
    plt.close()

    # Fig 3: Return rate vs mean erosion per return (category)
    if "return_rate" in cat.columns and "mean_profit_erosion_per_return" in cat.columns:
        df3 = cat.dropna(subset=["return_rate", "mean_profit_erosion_per_return"])
        plt.figure()
        plt.scatter(df3["return_rate"], df3["mean_profit_erosion_per_return"])
        plt.xlabel("Return Rate")
        plt.ylabel("Mean Profit Erosion per Return")
        plt.title("Category: Return Rate vs Mean Profit Erosion per Return")
        plt.tight_layout()
        plt.savefig(OUT_DIR / "fig3_return_rate_vs_mean_erosion_category.png", dpi=200)
        plt.close()

    print("RQ1 figures saved in reports/rq1/")

if __name__ == "__main__":
    main()