import re
import sqlite3
import pandas as pd
import numpy as np

con = sqlite3.connect("data/instamart.db")
parts = re.split(r"^-- (Q\d+):.*$", open("sql/queries.sql").read(), flags=re.M)
q = {parts[i]: pd.read_sql_query(parts[i + 1], con) for i in range(1, len(parts), 2)}

# ===== Report results against references =====
print("Q1:", len(q["Q1"]), "(30)")
print("Q2:", len(q["Q2"]), round(q["Q2"]["revenue"].sum(), 2), q["Q2"].iloc[0].tolist())
print("Q3:", len(q["Q3"]), "(11)")
print("Q4:", len(q["Q4"]), "(25)")
print("Q5:", len(q["Q5"]), q["Q5"]["avg_total"].iloc[0], "(550, 2552.06)")
print(q["Q6"].query("row_num == 1")[["city", "customer_id", "total_spend"]])
print("Q6 rank != row_number:", (q["Q6"]["rnk"] != q["Q6"]["row_num"]).sum(), "(8)")
print("Q7:", len(q["Q7"]), q["Q7"]["cumulative_revenue"].iloc[-1], "(183, 2998668.8)")

#pandas cross-check and exploratory analysis

conn = sqlite3.connect("data/instamart.db")
orders = pd.read_sql_query("SELECT * FROM orders;", conn)
customers = pd.read_sql_query("SELECT * FROM customers;", conn)
products = pd.read_sql_query("SELECT * FROM products;", conn)
conn.close()

# Completed orders dataset merged with dimensions
completed = orders[orders["status"] == "completed"].copy()
df_merged = completed.merge(customers, on="customer_id").merge(
    products, on="product_id"
)

# --- (a) Cross-checks ---
# 1. Pivot Table cross-check
pivot = pd.pivot_table(
    df_merged,
    values="order_amount",
    index="city",
    columns="category",
    aggfunc="sum",
    margins=True,
    margins_name="Total",
)
print("Grand Total:", round(pivot.loc["Total", "Total"], 2))

# 2. Named Aggregation by City
city_summary = df_merged.groupby("city").agg(
    sum_amount=("order_amount", "sum"),
    mean_amount=("order_amount", "mean"),
    count_orders=("order_amount", "count"),
)
print("\nCity Aggregation:")
print(city_summary.round(2))

# --- (b) Summary Statistics, IQR & Outliers ---
amount = completed["order_amount"]
mean_val = amount.mean()
median_val = amount.median()
skew_val = amount.skew()
q1 = amount.quantile(0.25)
q3 = amount.quantile(0.75)
iqr = q3 - q1
lower_fence = q1 - 1.5 * iqr
upper_fence = q3 + 1.5 * iqr

low_outliers = (amount < lower_fence).sum()
high_outliers = (amount > upper_fence).sum()

print("\n--- Order Amount Statistics ---")
print(
    f"Mean: {mean_val:.2f} | Median: {median_val:.2f} | Skewness: {skew_val:.2f}"
)
print(f"Q1: {q1:.2f} | Q3: {q3:.2f} | IQR: {iqr:.2f}")
print(f"Fences: [{lower_fence:.2f}, {upper_fence:.2f}]")
print(
    f"Outliers: {low_outliers} below lower fence, {high_outliers} above upper"
    " fence"
)

# Correlation Matrix
corr_cols = ["order_amount", "discount_pct", "quantity", "delivery_minutes"]
print("\nCorrelation Matrix:")
print(completed[corr_cols].corr().round(3))