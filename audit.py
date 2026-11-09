import pandas as pd
import sqlite3
from pathlib import Path

data = "data/raw"

customers = pd.read_csv(f"{data}/customers.csv")
orders = pd.read_csv(f"{data}/orders.csv")
products = pd.read_csv(f"{data}/products.csv")
tickets = pd.read_csv(f"{data}/tickets.csv", keep_default_na=False)

for name, df in [("customers", customers), ("orders", orders), ("products", products), ("tickets", tickets)]:
    
    print(f"\n ----- {name}: shape {df.shape} -----")
    print(df.dtypes)

    print(pd.DataFrame({"---null---": df.isna().sum(), "---percentage---": (df.isna().mean() * 100).round(2) }))



figures = {
    "Raw orders rows": (len(orders)),
    "Exact duplicate rows": (orders.duplicated().sum()),
    "Status Pending": ((orders["status"] == "pending").sum()),
    "Distinct order_id": (orders["order_id"].nunique()),
}

table = pd.DataFrame(list(figures.items()), columns=["Figure", "Value"])
print(table.to_string(index=False))

amt_num = pd.to_numeric(orders["order_amount"], errors="coerce")
dm = pd.to_numeric(orders["delivery_minutes"], errors="coerce")



figures2 = {
    "Non-numeric order_amount": (amt_num.isna()).sum(),
    "Negative delivery_minutes": (dm < 0).sum(),
    "Blank delivery_minutes": dm.isna().sum(),
}

table2 = pd.DataFrame(list(figures2.items()), columns=["Figure", "Value"])
print(table2.to_string(index=False))

age = pd.to_numeric(customers["age"], errors="coerce")
g = customers["gender"]
gender_blank = g.isna() | (g.astype(str).str.strip() == "")

figures3 = {
    "Blank age": age.isna().sum(),
    "Age outside 10-100": ((age < 10) | (age > 100)).sum(),
    "Distinct raw city spellings": customers["city"].nunique(),
    "Distinct non-blank gender spellings": g[~gender_blank].nunique(),
    "Blank gender": gender_blank.sum(),
}

table3 = pd.DataFrame(list(figures3.items()), columns=["Figure", "Value"])
print(table3.to_string(index=False))


customers["city"] = customers["city"].str.strip().str.title()

print(customers["city"].value_counts())

#Mapping gender

customers["gender"] = (
    customers["gender"]
    .str.strip()
    .str.lower()
    .map({
        "female": "female",
        "f": "female",
        "woman": "female",
        "male": "male",
        "m": "male",
        "man": "male",
        "": "unknown",
    })
    .fillna("unknown")
)

print(customers["gender"].value_counts())

#adding missing age

age_limit = (customers["age"] < 10 ) | (customers["age"] > 100 )

customers.loc[age_limit, "age"] = None

customers["age"] = customers["age"].fillna(
    customers.groupby("city")["age"].transform("median")
)

print(customers["age"].isna().sum())  
print(customers["age"].min(), customers["age"].max())

#Converting sign up date to date and time

customers["signup_date"] = pd.to_datetime(customers["signup_date"])

# question 5

# order_ts to datetime

orders["order_ts"] = pd.to_datetime(orders["order_ts"])

# dropping duplicates

print("Before:", len(orders))

orders = orders.drop_duplicates()
print("After_Duplicates:",len(orders))

#keeping the latest status of orders

n = len(orders)
orders = orders.sort_values("order_ts", kind="stable").drop_duplicates("order_id", keep="last")
orders = orders.sort_values("order_id").reset_index(drop=True)


#Order amount to numeric

orders["order_amount"] = pd.to_numeric(orders["order_amount"], errors="coerce")
price = orders["product_id"].map(products.set_index("product_id")["price_inr"])
calc = (price * orders["quantity"] * (1 - orders["discount_pct"] / 100)).round(2)
missing_amt = orders["order_amount"].isna()
orders.loc[missing_amt, "order_amount"] = calc[missing_amt]


#Step 5: delivery_minutes (negatives, completed median fill, cancelled stays missing) =====
orders["delivery_minutes"] = pd.to_numeric(orders["delivery_minutes"], errors="coerce")
neg = orders["delivery_minutes"] < 0
orders.loc[neg, "delivery_minutes"] = pd.NA
print("negative delivery_minutes set to missing:", neg.sum())

category = orders["product_id"].map(products.set_index("product_id")["category"])
done = orders["status"] == "completed"
category_median = orders["delivery_minutes"].where(done).groupby(category).transform("median")
to_fill = done & orders["delivery_minutes"].isna()
orders.loc[to_fill, "delivery_minutes"] = category_median[to_fill]
print("completed delivery_minutes filled with category median:", to_fill.sum())


#Store: write clean tables to CSV and SQLite

CLEAN_DIR = Path("data/clean")
DB_PATH = Path("data/instamart.db")


def store_clean_tables(tables, clean_dir=CLEAN_DIR, db_path=DB_PATH):
    """tables: dict of {table_name: DataFrame}. Writes each to CSV and to SQLite.
    Datetime columns are converted to text copies, so the DataFrames in memory stay unchanged."""
    clean_dir.mkdir(parents=True, exist_ok=True)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(db_path) as con:
        for name, df in tables.items():
            out = df.copy()
            # Datetime columns -> text (order_ts as YYYY-MM-DD HH:MM:SS, dates as YYYY-MM-DD)
            if "order_ts" in out.columns:
                out["order_ts"] = pd.to_datetime(out["order_ts"]).dt.strftime("%Y-%m-%d %H:%M:%S")
            if "signup_date" in out.columns:
                out["signup_date"] = pd.to_datetime(out["signup_date"]).dt.strftime("%Y-%m-%d")

            out.to_csv(clean_dir / f"{name}.csv", index=False)
            out.to_sql(name, con, if_exists="replace", index=False)
            print(f"{name}: {len(out)} rows written")


store_clean_tables({"customers": customers, "products": products, "orders": orders})

con = sqlite3.connect("data/instamart.db")
orders = pd.read_sql_query("SELECT * FROM orders", con)
customers = pd.read_sql_query("SELECT * FROM customers", con)
done = orders["status"] == "completed"
canc = orders["status"] == "cancelled"

assert len(orders) == 12309 and orders["order_id"].is_unique
assert done.sum() == 11110 and canc.sum() == 1199
assert (orders["status"] == "pending").sum() == 0
assert orders["order_amount"].isna().sum() == 0
assert orders.loc[done, "delivery_minutes"].isna().sum() == 0
assert orders.loc[canc, "delivery_minutes"].isna().sum() == 1199
assert round(orders.loc[done, "order_amount"].sum(), 2) == 2998668.80
assert customers["city"].nunique() == 6
assert customers["gender"].value_counts().to_dict() == {"female": 610, "male": 560, "unknown": 30}
assert customers["age"].isna().sum() == 0 and customers["age"].between(18, 60).all()

print("ALL CHECKPOINTS PASSED")



conn = sqlite3.connect("data/instamart.db")
with open("data/sql/queries.sql", "r") as f:
  queries = [q.strip() for q in f.read().split(";") if q.strip()]

for i, query in enumerate(queries, 1):
  df = pd.read_sql_query(query, conn)
  print(f"--- Q{i} Result ({len(df)} rows) ---")
  print(df.head(2), "\n")

conn.close()