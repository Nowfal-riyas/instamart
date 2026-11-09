# ===== Imports and settings =====
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.base import clone
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, silhouette_score)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

SNAPSHOT = pd.Timestamp("2025-10-01 00:00:00")
CUTOFF = pd.Timestamp("2025-09-01")
FINAL_K = 4
Path("outputs").mkdir(exist_ok=True)


# ===== Load clean data =====
orders = pd.read_csv("data/clean/orders.csv", parse_dates=["order_ts"])
customers = pd.read_csv("data/clean/customers.csv", parse_dates=["signup_date"])
products = pd.read_csv("data/clean/products.csv")


# ===== Task 1: RFM table =====
done = orders[orders["status"] == "completed"]
rfm = done.groupby("customer_id").agg(
    last_order_ts=("order_ts", "max"),
    frequency=("order_id", "nunique"),
    monetary=("order_amount", "sum"),
).reset_index()
rfm["recency"] = (SNAPSHOT - rfm["last_order_ts"]).dt.days
rfm = rfm[["customer_id", "recency", "frequency", "monetary"]]
print(len(rfm), rfm["recency"].agg(["median", "min", "max"]).tolist(),
      round(rfm["frequency"].mean(), 2), rfm["frequency"].max(),
      round(rfm["monetary"].mean(), 2), round(rfm["monetary"].max(), 2))


# ===== Task 2: scale RFM, K-means for K = 2..8 =====
X_rfm = StandardScaler().fit_transform(rfm[["recency", "frequency", "monetary"]])
rows = []
for k in range(2, 9):
    km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(X_rfm)
    rows.append((k, round(km.inertia_, 1), round(silhouette_score(X_rfm, km.labels_), 3)))
k_table = pd.DataFrame(rows, columns=["K", "inertia", "silhouette"])
print(k_table.to_string(index=False))


# ===== Task 3a: final clusters and segment profile =====
km = KMeans(n_clusters=FINAL_K, n_init=10, random_state=42).fit(X_rfm)
rfm["segment_id"] = km.labels_
profile = rfm.groupby("segment_id").agg(
    size=("customer_id", "count"), recency=("recency", "mean"),
    frequency=("frequency", "mean"), monetary=("monetary", "mean")).round(2)
print(profile)


# ===== Task 3b: segment names, actions, Premium flag =====
premium_id = profile["monetary"].idxmax()
rest = profile.drop(premium_id)
atrisk_id = rest["recency"].idxmax()
loyal_id = rest.drop(atrisk_id)["frequency"].idxmax()
names = {premium_id: "Premium", atrisk_id: "At-Risk", loyal_id: "Loyal Regulars"}
actions = {"Premium": "VIP perks and early access",
           "At-Risk": "Win-back offer before they lapse",
           "Loyal Regulars": "Loyalty points and referral rewards",
           "Occasional Buyers": "Bundle deals to raise order frequency"}
rfm["segment_name"] = rfm["segment_id"].map(names).fillna("Occasional Buyers")
rfm["is_premium"] = rfm["segment_id"] == premium_id
print(rfm.groupby("segment_name").size())
print(pd.Series(actions))


# ===== Task 3c: PCA and segment file =====
pca = PCA(n_components=2).fit(X_rfm)
print(pca.explained_variance_ratio_.round(3), pca.explained_variance_ratio_.sum().round(3))
rfm[["customer_id", "recency", "frequency", "monetary", "segment_id",
     "segment_name", "is_premium"]].to_csv("outputs/customer_segments.csv", index=False)


# ===== Task 4: lapse features (orders before the cutoff only) and label =====
def build_lapse_dataset(orders, customers, products, cutoff, window_end="2025-10-01"):
    cutoff, window_end = pd.Timestamp(cutoff), pd.Timestamp(window_end)
    o = orders.merge(products[["product_id", "category"]], on="product_id", how="left")
    pre = o[o["order_ts"] < cutoff]
    comp = pre[pre["status"] == "completed"]

    X = comp.groupby("customer_id").agg(
        last=("order_ts", "max"), frequency=("order_id", "nunique"),
        monetary=("order_amount", "sum"), avg_discount=("discount_pct", "mean"),
        avg_delivery=("delivery_minutes", "mean"),
        fav_category=("category", lambda s: s.mode().iloc[0]))
    X["recency"] = (cutoff - X.pop("last")).dt.days

    final = pre[pre["status"].isin(["completed", "cancelled"])]
    X["cancel_rate"] = ((final["status"] == "cancelled").groupby(final["customer_id"]).mean()
                        .reindex(X.index).fillna(0))

    c = customers.set_index("customer_id")[["city", "age", "gender", "signup_date"]]
    X = X.join(c)
    X["tenure_days"] = (cutoff - X.pop("signup_date")).dt.days

    active = o[(o["status"] == "completed") & (o["order_ts"] >= cutoff)
               & (o["order_ts"] < window_end)]["customer_id"].unique()
    y = pd.Series((~X.index.isin(active)).astype(int), index=X.index, name="lapsed")
    return X, y


X, y = build_lapse_dataset(orders, customers, products, CUTOFF)
print(len(X), y.sum(), round(y.mean() * 100, 2))                 # 1156, 443, 38.32

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42)
print(len(X_train), len(X_test))                                  # 924, 232


# ===== Task 5: preprocessing (fitted on training rows only) =====
nominal = ["city", "gender", "fav_category"]
numeric = [c for c in X.columns if c not in nominal]
assert X.isna().sum().sum() == 0, "missing values in features"

def make_prep():
    return ColumnTransformer([
        ("num", StandardScaler(), numeric),
        ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), nominal)])

prep_check = make_prep().fit(X_train)
idx = numeric.index("frequency")
print("scaler mean frequency (train):", round(prep_check.named_transformers_["num"].mean_[idx], 3),
      "| full cohort:", round(X["frequency"].mean(), 3))


# ===== Task 6a: baseline (always predict "not lapsed") =====
def report(name, y_true, y_pred):
    print(f"{name}: acc={accuracy_score(y_true, y_pred):.3f} "
          f"P={precision_score(y_true, y_pred, zero_division=0):.3f} "
          f"R={recall_score(y_true, y_pred, zero_division=0):.3f} "
          f"F1={f1_score(y_true, y_pred, zero_division=0):.3f}")
    print(confusion_matrix(y_true, y_pred))

report("baseline", y_test, np.zeros(len(y_test), dtype=int))


# ===== Task 6b/6c: tune logistic regression and random forest =====
lr_grid = GridSearchCV(
    Pipeline([("prep", make_prep()), ("model", LogisticRegression(max_iter=1000))]),
    {"model__C": [0.01, 0.1, 1, 10, 100]}, cv=5, scoring="f1").fit(X_train, y_train)
rf_grid = GridSearchCV(
    Pipeline([("prep", make_prep()),
              ("model", RandomForestClassifier(n_estimators=100, random_state=42))]),
    {"model__max_depth": [3, 5, 8], "model__min_samples_leaf": [5, 10, 20]},
    cv=5, scoring="f1").fit(X_train, y_train)
print("LR:", lr_grid.best_params_, round(lr_grid.best_score_, 3))
print("RF:", rf_grid.best_params_, round(rf_grid.best_score_, 3))


# ===== Task 6d: test-set evaluation and model choice =====
report("LogReg test", y_test, lr_grid.predict(X_test))
report("RandForest test", y_test, rf_grid.predict(X_test))
f1_lr, f1_rf = f1_score(y_test, lr_grid.predict(X_test)), f1_score(y_test, rf_grid.predict(X_test))
best_model = rf_grid.best_estimator_ if f1_rf >= f1_lr else lr_grid.best_estimator_
print("selected:", "RandomForest" if f1_rf >= f1_lr else "LogisticRegression")


# ===== Task 7: over-sample lapsed class in training only =====
extra = y_train[y_train == 1].sample(
    (y_train == 0).sum() - (y_train == 1).sum(), replace=True, random_state=42).index
X_bal = pd.concat([X_train, X_train.loc[extra]])
y_bal = pd.concat([y_train, y_train.loc[extra]])
print(y_bal.value_counts().to_dict())                             # 570 each
lr_bal = clone(lr_grid.best_estimator_).fit(X_bal, y_bal)
report("LogReg unbalanced", y_test, lr_grid.predict(X_test))
report("LogReg balanced", y_test, lr_bal.predict(X_test))


# ===== Task 8a: feature importance and coefficients =====
rf_pipe, lr_pipe = rf_grid.best_estimator_, lr_grid.best_estimator_
imp = pd.Series(rf_pipe["model"].feature_importances_,
                index=rf_pipe["prep"].get_feature_names_out()).sort_values(ascending=False)
print(imp.head(5))
coef = pd.Series(lr_pipe["model"].coef_[0], index=lr_pipe["prep"].get_feature_names_out())
print(coef[["num__recency", "num__frequency", "num__cancel_rate"]])


# ===== Task 8b: save, reload, verify identical predictions =====
joblib.dump(best_model, "outputs/lapse_model.joblib")
reloaded = joblib.load("outputs/lapse_model.joblib")
assert np.array_equal(best_model.predict(X_test), reloaded.predict(X_test))
print("reloaded model reproduces test predictions exactly")


# ===== Task 8c: Premium retention list from test set =====
premium_ids = set(rfm.loc[rfm["is_premium"], "customer_id"])
test_prem = X_test[X_test.index.isin(premium_ids)]
retention = pd.DataFrame({
    "customer_id": test_prem.index,
    "lapse_probability": best_model.predict_proba(test_prem)[:, 1],
    "actually_lapsed": y_test.loc[test_prem.index].values,
}).sort_values("lapse_probability", ascending=False).head(10)
retention.to_csv("outputs/retention_list.csv", index=False)
print(retention, "| truly lapsed:", retention["actually_lapsed"].sum())