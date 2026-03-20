from pathlib import Path

import pandas as pd


def load_data(base_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    data_dir = base_dir / "data"
    sales_df = pd.read_csv(data_dir / "historical_sales.csv", parse_dates=["date"])
    warehouses_df = pd.read_csv(data_dir / "warehouses.csv")
    distances_df = pd.read_csv(data_dir / "distances.csv")
    return sales_df, warehouses_df, distances_df


def predict_demand(sales_df: pd.DataFrame, growth_factor: float = 1.1) -> pd.DataFrame:
    """Predict next-day demand per store-product using historical average with growth."""
    forecast_df = (
        sales_df.groupby(["store", "product"], as_index=False)["sales"]
        .mean()
        .rename(columns={"sales": "avg_historical_sales"})
    )
    forecast_df["predicted_demand"] = (forecast_df["avg_historical_sales"] * growth_factor).round(2)
    return forecast_df


def plan_inventory(forecast_df: pd.DataFrame, warehouses_df: pd.DataFrame) -> pd.DataFrame:
    """Compare predicted demand against warehouse stock for each product."""
    # Get max available stock per product across all warehouses
    max_stock = warehouses_df.groupby("product")["stock"].max().reset_index()
    max_stock.rename(columns={"stock": "available_stock"}, inplace=True)
    
    # Merge and check for shortage
    plan_df = forecast_df.merge(max_stock, on="product", how="left")
    plan_df["shortage"] = (plan_df["predicted_demand"] - plan_df["available_stock"]).clip(lower=0)
    plan_df["status"] = plan_df["shortage"].apply(lambda x: "shortage" if x > 0 else "ok")
    return plan_df


DELIVERY_COLS = ["base_handling_fee", "cost_per_km", "weight_kg_per_unit", "min_order_qty", "express_surcharge_pct"]


def assign_best_warehouse(plan_df: pd.DataFrame, warehouses_df: pd.DataFrame, distances_df: pd.DataFrame) -> pd.DataFrame:
    """Assign each store-product to closest warehouse that has stock for that product."""
    # Merge distances with warehouse stock and delivery pricing info
    wh_distances = distances_df.merge(warehouses_df, on="warehouse", how="left")

    # For each store-product, find closest warehouse with that product
    def get_best_warehouse(row):
        store = row["store"]
        product = row["product"]
        options = wh_distances[(wh_distances["store"] == store) & (wh_distances["product"] == product)]
        if options.empty:
            return pd.Series([None, None] + [None] * len(DELIVERY_COLS))
        best = options.sort_values("distance_km").iloc[0]
        return pd.Series([best["warehouse"], best["distance_km"]] + [best[c] for c in DELIVERY_COLS])

    plan_df[["warehouse", "distance_km"] + DELIVERY_COLS] = plan_df.apply(
        get_best_warehouse, axis=1, result_type="expand"
    )

    plan_df["ship_qty"] = plan_df["shortage"].round(2)

    # Amazon-style delivery cost:
    #   base_handling_fee  (fixed: picking, packing, labelling per shipment)
    # + weight_kg_per_unit × ship_qty × cost_per_km × distance_km  (variable: weight × distance)
    # Only charged when there is an actual shipment (ship_qty > 0)
    has_shipment = plan_df["ship_qty"] > 0
    variable = plan_df["weight_kg_per_unit"] * plan_df["ship_qty"] * plan_df["cost_per_km"] * plan_df["distance_km"]
    plan_df["delivery_cost"] = ((plan_df["base_handling_fee"] * has_shipment) + variable).round(2)
    return plan_df


def run_pipeline() -> pd.DataFrame:
    base_dir = Path(__file__).resolve().parents[1]
    sales_df, warehouses_df, distances_df = load_data(base_dir)

    forecast_df = predict_demand(sales_df, growth_factor=1.1)
    plan_df = plan_inventory(forecast_df, warehouses_df)
    output_df = assign_best_warehouse(plan_df, warehouses_df, distances_df)

    output_path = base_dir / "data" / "output.csv"
    output_df.to_csv(output_path, index=False)

    print("Pipeline complete!")
    print(f"Saved: {output_path}")
    print(f"\nProcessed {len(output_df)} store-product combinations")
    print(f"From {len(sales_df)} historical sales records")
    print(f"\nShortages identified: {len(output_df[output_df['status'] == 'shortage'])}")
    print(f"Total delivery cost: ${output_df['delivery_cost'].sum():,.2f}")
    print(f"\nOutput preview:")
    print(output_df.head(10))
    return output_df


if __name__ == "__main__":
    run_pipeline()
    
