"""
Part 2: Strategic Data Engineering & AI-Driven Insights (Pandas)
Industrial Implementation for ShopEase Retail Dataset

Key Modules:
1. AI-Powered Data Hygiene (Null detection, domain imputation: FillNA with 0.0 for discounts, median/clean)
2. Relational Data Merging (Orders ⨝ Products ⨝ Customers -> Master Sales Table)
3. Chronological Data Handling (Datetime parsing, Seasonality, Monthly Troughs/Peaks, Day-of-Week load)
4. Strategic Business Deliverables:
   - Top 10 Customers by Lifetime Value (CLV)
   - Most Profitable Product Categories (Sports vs Electronics vs Fashion vs Home)
   - Revenue vs Discount Correlation Analysis
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def run_part2_pipeline(data_dir="data", output_dir="reports"):
    os.makedirs(output_dir, exist_ok=True)
    print("=" * 70)
    print("PART 2: STRATEGIC DATA ENGINEERING & AI-DRIVEN INSIGHTS (PANDAS)")
    print("=" * 70)

    # -------------------------------------------------------------
    # 1. LOAD DISPARATE DATASETS
    # -------------------------------------------------------------
    print("\n[Step 1/5] Ingesting Siloed Raw Datasets...")
    orders_path = os.path.join(data_dir, "orders.csv")
    products_path = os.path.join(data_dir, "products.csv")
    customers_path = os.path.join(data_dir, "customers.csv")

    df_orders = pd.read_csv(orders_path)
    df_products = pd.read_csv(products_path)
    df_customers = pd.read_csv(customers_path)

    print(f" -> Ingested Orders:    {len(df_orders):,} rows, {df_orders.shape[1]} columns")
    print(f" -> Ingested Products:  {len(df_products):,} rows, {df_products.shape[1]} columns")
    print(f" -> Ingested Customers: {len(df_customers):,} rows, {df_customers.shape[1]} columns")

    # -------------------------------------------------------------
    # 2. AI-POWERED DATA HYGIENE
    # -------------------------------------------------------------
    print("\n[Step 2/5] AI-Powered Data Hygiene Audit...")
    null_audit_before = {
        "orders_missing_discounts": int(df_orders["DiscountPct"].isna().sum()),
        "orders_missing_pct": round(float(df_orders["DiscountPct"].isna().mean() * 100), 2),
        "customers_missing_emails": int(df_customers["Email"].isna().sum()),
        "products_missing_prices": int(df_products["UnitPrice"].isna().sum())
    }
    print(f" -> Missing DiscountPct in Orders: {null_audit_before['orders_missing_discounts']:,} rows ({null_audit_before['orders_missing_pct']}%)")
    print(f" -> Missing Email in Customers:     {null_audit_before['customers_missing_emails']:,} rows")

    # AI Data Hygiene Imputation Strategy:
    # 1. In retail transaction databases, a null DiscountPct means zero promotional discount applied (0.0).
    df_orders_cleaned = df_orders.copy()
    df_orders_cleaned["DiscountPct"] = df_orders_cleaned["DiscountPct"].fillna(0.0)

    # 2. Clean customer missing emails with fallback synthetic handle
    df_customers_cleaned = df_customers.copy()
    df_customers_cleaned["Email"] = df_customers_cleaned["Email"].fillna(
        df_customers_cleaned["CustomerID"].apply(lambda cid: f"user.{cid.lower()}@guest.shopease.internal")
    )

    null_audit_after = {
        "orders_missing_discounts": int(df_orders_cleaned["DiscountPct"].isna().sum()),
        "customers_missing_emails": int(df_customers_cleaned["Email"].isna().sum()),
        "products_missing_prices": int(df_products["UnitPrice"].isna().sum())
    }
    print(" -> Data Hygiene applied: DiscountPct NULLs replaced with 0.0 (Standard retail full-MSRP rule)")
    print(f" -> Remaining NULLs in Orders: {null_audit_after['orders_missing_discounts']}")

    # -------------------------------------------------------------
    # 3. CHRONOLOGICAL DATA HANDLING
    # -------------------------------------------------------------
    print("\n[Step 3/5] Chronological Feature Engineering & Operational Patterns...")
    # Convert mixed string date representations into robust pandas Timestamp objects
    df_orders_cleaned["OrderDate"] = pd.to_datetime(df_orders_cleaned["OrderDate"], format="mixed", errors="coerce")
    
    # Extract chronological attributes
    df_orders_cleaned["Year"] = df_orders_cleaned["OrderDate"].dt.year
    df_orders_cleaned["Month"] = df_orders_cleaned["OrderDate"].dt.month
    df_orders_cleaned["MonthName"] = df_orders_cleaned["OrderDate"].dt.strftime("%b")
    df_orders_cleaned["Quarter"] = "Q" + df_orders_cleaned["OrderDate"].dt.quarter.astype(str)
    df_orders_cleaned["DayOfWeek"] = df_orders_cleaned["OrderDate"].dt.day_name()
    df_orders_cleaned["DayOfWeekNum"] = df_orders_cleaned["OrderDate"].dt.day_of_week
    df_orders_cleaned["IsWeekend"] = df_orders_cleaned["DayOfWeekNum"].isin([5, 6])

    # -------------------------------------------------------------
    # 4. RELATIONAL DATA MERGING (MASTER TABLE)
    # -------------------------------------------------------------
    print("\n[Step 4/5] Relational Data Merging: Orders JOIN Products JOIN Customers...")
    # Step 4a: Left join Orders with Products on ProductID
    merged_op = pd.merge(df_orders_cleaned, df_products, on="ProductID", how="left")

    # Step 4b: Left join with Customers on CustomerID
    df_master = pd.merge(merged_op, df_customers_cleaned, on="CustomerID", how="left")

    # Compute financial measures
    df_master["GrossRevenue"] = df_master["Quantity"] * df_master["UnitPrice"]
    df_master["DiscountAmount"] = df_master["GrossRevenue"] * df_master["DiscountPct"]
    df_master["NetRevenue"] = df_master["GrossRevenue"] - df_master["DiscountAmount"]
    df_master["TotalCost"] = df_master["Quantity"] * df_master["UnitCost"]
    df_master["GrossProfit"] = df_master["NetRevenue"] - df_master["TotalCost"]
    df_master["ProfitMarginPct"] = (df_master["GrossProfit"] / df_master["NetRevenue"]) * 100

    # Save Master Table
    master_path = os.path.join(data_dir, "master_sales_table.csv")
    df_master.to_csv(master_path, index=False)
    print(f" -> Master Table created successfully: {len(df_master):,} records across {df_master.shape[1]} attributes")
    print(f" -> Saved Master Table -> {master_path}")

    # -------------------------------------------------------------
    # 5. STRATEGIC DELIVERABLES & BUSINESS INTELLIGENCE
    # -------------------------------------------------------------
    print("\n[Step 5/5] Synthesizing Strategic Deliverables...")

    # DELIVERABLE 1: Top 10 Customers by Lifetime Value (CLV)
    customer_clv = df_master.groupby(["CustomerID", "CustomerName", "City", "Region"]).agg(
        TotalOrders=("OrderID", "count"),
        TotalUnits=("Quantity", "sum"),
        TotalNetRevenue=("NetRevenue", "sum"),
        TotalGrossProfit=("GrossProfit", "sum"),
        AvgOrderValue=("NetRevenue", "mean")
    ).reset_index()

    customer_clv["CustomerMarginPct"] = (customer_clv["TotalGrossProfit"] / customer_clv["TotalNetRevenue"]) * 100
    top_10_customers = customer_clv.sort_values(by="TotalNetRevenue", ascending=False).head(10)

    print("\n--- TOP 10 CUSTOMERS BY LIFETIME VALUE (CLV) ---")
    for rank, (_, row) in enumerate(top_10_customers.iterrows(), 1):
        print(f" #{rank:02d} | {row['CustomerID']} | {row['CustomerName']:<20} | {row['City']:<14} | Spent: ${row['TotalNetRevenue']:>10,.2f} | Profit: ${row['TotalGrossProfit']:>9,.2f} | Orders: {row['TotalOrders']}")

    # DELIVERABLE 2: Category Profitability (Sports vs Electronics vs Fashion vs Home, etc.)
    cat_summary = df_master.groupby("Category").agg(
        OrderCount=("OrderID", "count"),
        TotalUnits=("Quantity", "sum"),
        GrossRevenue=("GrossRevenue", "sum"),
        DiscountAmount=("DiscountAmount", "sum"),
        NetRevenue=("NetRevenue", "sum"),
        TotalCost=("TotalCost", "sum"),
        GrossProfit=("GrossProfit", "sum"),
        AvgDiscountPct=("DiscountPct", "mean")
    ).reset_index()

    cat_summary["ProfitMarginPct"] = (cat_summary["GrossProfit"] / cat_summary["NetRevenue"]) * 100
    cat_summary["AvgDiscountPct"] = cat_summary["AvgDiscountPct"] * 100
    cat_summary = cat_summary.sort_values(by="GrossProfit", ascending=False)

    print("\n--- CATEGORY PROFITABILITY BREAKDOWN ---")
    for _, row in cat_summary.iterrows():
        print(f" {row['Category']:<20} | Net Rev: ${row['NetRevenue']:>11,.2f} | Profit: ${row['GrossProfit']:>10,.2f} | Margin: {row['ProfitMarginPct']:>5.1f}% | Avg Disc: {row['AvgDiscountPct']:>4.1f}%")

    # DELIVERABLE 3: Revenue vs Discount Correlation Analysis
    corr_matrix = df_master[["DiscountPct", "Quantity", "GrossRevenue", "NetRevenue", "GrossProfit", "ProfitMarginPct"]].corr()
    corr_discount_revenue = float(corr_matrix.loc["DiscountPct", "NetRevenue"])
    corr_discount_profit = float(corr_matrix.loc["DiscountPct", "GrossProfit"])
    corr_discount_margin = float(corr_matrix.loc["DiscountPct", "ProfitMarginPct"])
    corr_discount_quantity = float(corr_matrix.loc["DiscountPct", "Quantity"])

    print("\n--- REVENUE VS. DISCOUNT CORRELATION METRICS ---")
    print(f" -> Correlation (Discount % vs Net Revenue):     {corr_discount_revenue:+.4f}")
    print(f" -> Correlation (Discount % vs Gross Profit):    {corr_discount_profit:+.4f}")
    print(f" -> Correlation (Discount % vs Profit Margin %): {corr_discount_margin:+.4f}")
    print(f" -> Correlation (Discount % vs Order Quantity):  {corr_discount_quantity:+.4f}")

    # Discount bucket elasticity analysis
    df_master["DiscountBucket"] = pd.cut(
        df_master["DiscountPct"],
        bins=[-0.01, 0.001, 0.101, 0.201, 0.35],
        labels=["0% (Full Price)", "1% - 10%", "11% - 20%", "21% - 30%"]
    )
    discount_elasticity = df_master.groupby("DiscountBucket", observed=False).agg(
        OrderCount=("OrderID", "count"),
        AvgUnitsPerOrder=("Quantity", "mean"),
        AvgNetRevenue=("NetRevenue", "mean"),
        AvgProfitMargin=("ProfitMarginPct", "mean"),
        TotalProfitContribution=("GrossProfit", "sum")
    ).reset_index()

    # Chronological Analysis: Monthly & Day-of-Week
    month_order = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    monthly_trend = df_master.groupby(["Month", "MonthName"]).agg(
        OrderCount=("OrderID", "count"),
        NetRevenue=("NetRevenue", "sum"),
        GrossProfit=("GrossProfit", "sum")
    ).reset_index().sort_values(by="Month")
    
    lowest_month_row = monthly_trend.loc[monthly_trend["NetRevenue"].idxmin()]
    highest_month_row = monthly_trend.loc[monthly_trend["NetRevenue"].idxmax()]

    day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    dow_summary = df_master.groupby("DayOfWeek").agg(
        OrderCount=("OrderID", "count"),
        NetRevenue=("NetRevenue", "sum")
    ).reindex(day_order).reset_index()
    busiest_day_row = dow_summary.loc[dow_summary["OrderCount"].idxmax()]

    # Regional Demographics Summary
    regional_summary = df_master.groupby("Region").agg(
        TotalCustomers=("CustomerID", "nunique"),
        TotalOrders=("OrderID", "count"),
        NetRevenue=("NetRevenue", "sum"),
        GrossProfit=("GrossProfit", "sum")
    ).reset_index().sort_values(by="NetRevenue", ascending=False)
    regional_summary["ProfitMarginPct"] = (regional_summary["GrossProfit"] / regional_summary["NetRevenue"]) * 100

    print(f"\n -> Lowest Revenue Month:  {lowest_month_row['MonthName']} (${lowest_month_row['NetRevenue']:,.2f})")
    print(f" -> Peak Revenue Month:    {highest_month_row['MonthName']} (${highest_month_row['NetRevenue']:,.2f})")
    print(f" -> Busiest Day of Week:   {busiest_day_row['DayOfWeek']} ({busiest_day_row['OrderCount']:,} orders)")

    # -------------------------------------------------------------
    # GENERATE HIGH-IMPACT MATPLOTLIB VISUALIZATIONS
    # -------------------------------------------------------------
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

    # Chart 1: Top 10 Customers by Lifetime Value
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    cust_labels = [f"{row['CustomerName']}\n({row['City']})" for _, row in top_10_customers.iterrows()]
    y_pos = np.arange(len(cust_labels))
    bars1 = ax1.barh(y_pos, top_10_customers["TotalNetRevenue"], color='#2563eb', edgecolor='#1e40af')
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(cust_labels, fontsize=9)
    ax1.invert_yaxis()
    ax1.set_xlabel('Lifetime Net Revenue ($)', fontsize=11, fontweight='bold')
    ax1.set_title('Top 10 Customers by Lifetime Value (ShopEase Annual)', fontsize=13, fontweight='bold', pad=15)
    for i, v in enumerate(top_10_customers["TotalNetRevenue"]):
        profit = top_10_customers.iloc[i]["TotalGrossProfit"]
        ax1.text(v + 500, i, f'${v:,.0f} (Profit: ${profit:,.0f})', va='center', fontsize=9, fontweight='bold', color='#0f172a')
    plt.tight_layout()
    chart1_path = os.path.join(output_dir, "part2_top_customers.png")
    fig1.savefig(chart1_path, dpi=200)
    plt.close(fig1)

    # Chart 2: Category Profitability (Net Revenue vs Gross Profit)
    fig2, ax2 = plt.subplots(figsize=(11, 5.5))
    x = np.arange(len(cat_summary))
    width = 0.35
    rects1 = ax2.bar(x - width/2, cat_summary["NetRevenue"] / 1000, width, label='Net Revenue ($K)', color='#3b82f6', edgecolor='#1e3a8a')
    rects2 = ax2.bar(x + width/2, cat_summary["GrossProfit"] / 1000, width, label='Gross Profit ($K)', color='#10b981', edgecolor='#065f46')
    ax2.set_ylabel('Amount in Thousands USD ($K)', fontsize=11, fontweight='bold')
    ax2.set_title('Category Profitability & Revenue Benchmark', fontsize=13, fontweight='bold', pad=15)
    ax2.set_xticks(x)
    ax2.set_xticklabels(cat_summary["Category"], fontsize=10, fontweight='bold')
    ax2.legend(frameon=True, facecolor='#f8fafc', edgecolor='#cbd5e1')

    # Add margin badge above bars
    for idx, row in cat_summary.iterrows():
        cat_x = x[list(cat_summary["Category"]).index(row["Category"])]
        margin_val = row["ProfitMarginPct"]
        ax2.annotate(f"{margin_val:.1f}% Margin", 
                     xy=(cat_x, (row["NetRevenue"] / 1000)),
                     xytext=(0, 6), textcoords="offset points",
                     ha='center', va='bottom', fontsize=9, fontweight='bold', color='#047857')
    plt.tight_layout()
    chart2_path = os.path.join(output_dir, "part2_category_profitability.png")
    fig2.savefig(chart2_path, dpi=200)
    plt.close(fig2)

    # Chart 3: Revenue vs Discount Correlation & Margin Impact
    fig3, (ax3a, ax3b) = plt.subplots(1, 2, figsize=(14, 5.5))
    
    # Scatter sample of 600 points for clear visual rendering
    sample_df = df_master.sample(n=min(800, len(df_master)), random_state=42)
    scatter = ax3a.scatter(sample_df["DiscountPct"] * 100, sample_df["NetRevenue"], 
                           c=sample_df["ProfitMarginPct"], cmap="viridis", alpha=0.6, edgecolors="none")
    cbar = plt.colorbar(scatter, ax=ax3a)
    cbar.set_label('Profit Margin %', fontsize=10, fontweight='bold')
    ax3a.set_xlabel('Discount Percentage Applied (%)', fontsize=11, fontweight='bold')
    ax3a.set_ylabel('Net Revenue per Order ($)', fontsize=11, fontweight='bold')
    ax3a.set_title('Order Net Revenue vs. Discount Level', fontsize=12, fontweight='bold')

    # Elasticity Bar Chart: Profit Margin by Discount Tier
    ax3b.bar(discount_elasticity["DiscountBucket"].astype(str), discount_elasticity["AvgProfitMargin"],
             color=['#059669', '#10b981', '#f59e0b', '#ef4444'], edgecolor='#334155', width=0.55)
    ax3b.set_ylabel('Average Profit Margin (%)', fontsize=11, fontweight='bold')
    ax3b.set_xlabel('Discount Tier', fontsize=11, fontweight='bold')
    ax3b.set_title('Margin Erosion Across Discount Tiers', fontsize=12, fontweight='bold')
    for i, val in enumerate(discount_elasticity["AvgProfitMargin"]):
        ax3b.text(i, val + 1, f'{val:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=10)

    plt.tight_layout()
    chart3_path = os.path.join(output_dir, "part2_revenue_discount_corr.png")
    fig3.savefig(chart3_path, dpi=200)
    plt.close(fig3)

    # Chart 4: Monthly Operational Revenue Trend & Day of Week
    fig4, (ax4a, ax4b) = plt.subplots(1, 2, figsize=(14, 5))
    ax4a.plot(monthly_trend["MonthName"], monthly_trend["NetRevenue"] / 1000, 
              marker='o', color='#2563eb', linewidth=2.5, markersize=7)
    ax4a.set_ylabel('Net Revenue ($K)', fontsize=11, fontweight='bold')
    ax4a.set_title('Annual Revenue Curve: Seasonal Demand Peaks', fontsize=12, fontweight='bold')
    
    # Highlight Trough & Peak
    min_idx = monthly_trend.index[monthly_trend["NetRevenue"] == monthly_trend["NetRevenue"].min()][0]
    max_idx = monthly_trend.index[monthly_trend["NetRevenue"] == monthly_trend["NetRevenue"].max()][0]
    ax4a.scatter([lowest_month_row["MonthName"]], [lowest_month_row["NetRevenue"] / 1000], color='#ef4444', s=120, zorder=5)
    ax4a.scatter([highest_month_row["MonthName"]], [highest_month_row["NetRevenue"] / 1000], color='#10b981', s=120, zorder=5)
    ax4a.annotate(f'Lowest: {lowest_month_row["MonthName"]}\n(${lowest_month_row["NetRevenue"]/1000:,.0f}K)', 
                  xy=(lowest_month_row["MonthName"], lowest_month_row["NetRevenue"]/1000),
                  xytext=(-20, 25), textcoords="offset points",
                  bbox=dict(boxstyle="round,pad=0.3", fc="#fee2e2", ec="#ef4444"),
                  fontsize=8, fontweight='bold', arrowprops=dict(arrowstyle="->", color='#ef4444'))
    ax4a.annotate(f'Peak: {highest_month_row["MonthName"]}\n(${highest_month_row["NetRevenue"]/1000:,.0f}K)', 
                  xy=(highest_month_row["MonthName"], highest_month_row["NetRevenue"]/1000),
                  xytext=(-20, -35), textcoords="offset points",
                  bbox=dict(boxstyle="round,pad=0.3", fc="#dcfce7", ec="#10b981"),
                  fontsize=8, fontweight='bold', arrowprops=dict(arrowstyle="->", color='#10b981'))

    # Day of Week Orders
    ax4b.bar(dow_summary["DayOfWeek"], dow_summary["OrderCount"], color='#8b5cf6', edgecolor='#5b21b6', width=0.55)
    ax4b.set_ylabel('Order Count', fontsize=11, fontweight='bold')
    ax4b.set_title('Weekly Operational Fulfillment Load', fontsize=12, fontweight='bold')
    ax4b.tick_params(axis='x', rotation=30)
    for i, count in enumerate(dow_summary["OrderCount"]):
        ax4b.text(i, count + 15, f'{count:,}', ha='center', va='bottom', fontsize=8, fontweight='bold')

    plt.tight_layout()
    chart4_path = os.path.join(output_dir, "part2_monthly_operations.png")
    fig4.savefig(chart4_path, dpi=200)
    plt.close(fig4)

    # -------------------------------------------------------------
    # CONSOLIDATE JSON DELIVERABLES FOR WEB UI & EXECUTIVE REPORTS
    # -------------------------------------------------------------
    deliverable_payload = {
        "kpis": {
            "total_orders": len(df_master),
            "total_customers": int(df_master["CustomerID"].nunique()),
            "total_products": int(df_master["ProductID"].nunique()),
            "gross_revenue": round(float(df_master["GrossRevenue"].sum()), 2),
            "total_discounts_granted": round(float(df_master["DiscountAmount"].sum()), 2),
            "net_revenue": round(float(df_master["NetRevenue"].sum()), 2),
            "total_cogs": round(float(df_master["TotalCost"].sum()), 2),
            "gross_profit": round(float(df_master["GrossProfit"].sum()), 2),
            "overall_profit_margin_pct": round(float((df_master["GrossProfit"].sum() / df_master["NetRevenue"].sum()) * 100), 2),
            "avg_order_value": round(float(df_master["NetRevenue"].mean()), 2)
        },
        "hygiene_audit": {
            "before": null_audit_before,
            "after": null_audit_after,
            "strategy": "Replaced missing discount percentages with 0.0 under the industrial assumption that unlisted discounts indicate regular full-MSRP checkout without coupon codes."
        },
        "top_10_customers": [
            {
                "rank": i + 1,
                "customer_id": r["CustomerID"],
                "name": r["CustomerName"],
                "city": r["City"],
                "region": r["Region"],
                "orders_count": int(r["TotalOrders"]),
                "units_purchased": int(r["TotalUnits"]),
                "lifetime_net_revenue": round(float(r["TotalNetRevenue"]), 2),
                "lifetime_profit": round(float(r["TotalGrossProfit"]), 2),
                "margin_pct": round(float(r["CustomerMarginPct"]), 1),
                "avg_order_value": round(float(r["AvgOrderValue"]), 2)
            }
            for i, (_, r) in enumerate(top_10_customers.iterrows())
        ],
        "category_profitability": [
            {
                "category": r["Category"],
                "order_count": int(r["OrderCount"]),
                "units_sold": int(r["TotalUnits"]),
                "gross_revenue": round(float(r["GrossRevenue"]), 2),
                "net_revenue": round(float(r["NetRevenue"]), 2),
                "gross_profit": round(float(r["GrossProfit"]), 2),
                "profit_margin_pct": round(float(r["ProfitMarginPct"]), 2),
                "avg_discount_pct": round(float(r["AvgDiscountPct"]), 2)
            }
            for _, r in cat_summary.iterrows()
        ],
        "discount_correlation": {
            "corr_discount_net_revenue": round(corr_discount_revenue, 4),
            "corr_discount_gross_profit": round(corr_discount_profit, 4),
            "corr_discount_profit_margin": round(corr_discount_margin, 4),
            "corr_discount_order_quantity": round(corr_discount_quantity, 4),
            "elasticity_tiers": [
                {
                    "tier": str(r["DiscountBucket"]),
                    "order_count": int(r["OrderCount"]),
                    "avg_units": round(float(r["AvgUnitsPerOrder"]), 2),
                    "avg_net_rev": round(float(r["AvgNetRevenue"]), 2),
                    "avg_margin_pct": round(float(r["AvgProfitMargin"]), 2),
                    "total_profit": round(float(r["TotalProfitContribution"]), 2)
                }
                for _, r in discount_elasticity.iterrows()
            ]
        },
        "chronological_insights": {
            "lowest_revenue_month": {
                "month": lowest_month_row["MonthName"],
                "net_revenue": round(float(lowest_month_row["NetRevenue"]), 2)
            },
            "peak_revenue_month": {
                "month": highest_month_row["MonthName"],
                "net_revenue": round(float(highest_month_row["NetRevenue"]), 2)
            },
            "busiest_day_of_week": {
                "day": busiest_day_row["DayOfWeek"],
                "order_count": int(busiest_day_row["OrderCount"])
            },
            "monthly_breakdown": [
                {
                    "month": r["MonthName"],
                    "order_count": int(r["OrderCount"]),
                    "net_revenue": round(float(r["NetRevenue"]), 2),
                    "gross_profit": round(float(r["GrossProfit"]), 2)
                }
                for _, r in monthly_trend.iterrows()
            ],
            "day_of_week_breakdown": [
                {
                    "day": r["DayOfWeek"],
                    "order_count": int(r["OrderCount"]),
                    "net_revenue": round(float(r["NetRevenue"]), 2)
                }
                for _, r in dow_summary.iterrows()
            ]
        },
        "regional_breakdown": [
            {
                "region": r["Region"],
                "total_customers": int(r["TotalCustomers"]),
                "order_count": int(r["TotalOrders"]),
                "net_revenue": round(float(r["NetRevenue"]), 2),
                "gross_profit": round(float(r["GrossProfit"]), 2),
                "profit_margin_pct": round(float(r["ProfitMarginPct"]), 2)
            }
            for _, r in regional_summary.iterrows()
        ]
    }

    report_json_path = os.path.join(output_dir, "part2_report.json")
    with open(report_json_path, "w") as f:
        json.dump(deliverable_payload, f, indent=2)
    print(f"\n[OK] Saved Part 2 Deliverables JSON -> {report_json_path}")
    print(f"[OK] Generated 4 Matplotlib Charts in {output_dir}/")
    return deliverable_payload

if __name__ == "__main__":
    run_part2_pipeline()
