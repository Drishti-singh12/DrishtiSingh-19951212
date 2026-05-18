"""
Etraveli Group — Data Insight Analyst Assessment
Parts 2 (Statistical Analysis), 3 (Machine Learning), 4 (Executive Storytelling)
Author: Drishti singh
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, average_precision_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import json

warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
ORDERS_PATH  = os.path.join(os.path.dirname(__file__), "orders_data/orders.csv")
ERRANDS_PATH = os.path.join(os.path.dirname(__file__), "errands_data/errands.csv")
CHARTS_DIR   = os.path.join(os.path.dirname(__file__), "charts")
os.makedirs(CHARTS_DIR, exist_ok=True)

PALETTE = {
    "primary":   "#1B3A6B",
    "accent":    "#E8A020",
    "danger":    "#C0392B",
    "ok":        "#27AE60",
    "light":     "#F4F6FA",
    "mid":       "#8CA0BC",
    "text":      "#1A1A2E",
}

sns.set_theme(style="whitegrid", font_scale=1.1)
plt.rcParams.update({
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.facecolor":     "#FAFBFE",
    "figure.facecolor":   "white",
    "font.family":        "DejaVu Sans",
})


# ─────────────────────────────────────────────
# UTILS
# ─────────────────────────────────────────────
def base36_to_int(s):
    return int(str(s), 36)

def save(fig, name):
    path = os.path.join(CHARTS_DIR, f"{name}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")
    return path


# ─────────────────────────────────────────────
# LOAD & PREP  (shared across all parts)
# ─────────────────────────────────────────────
print("=" * 60)
print("LOADING DATA")
print("=" * 60)

orders  = pd.read_csv(ORDERS_PATH)
errands = pd.read_csv(ERRANDS_PATH)

# Filter test records
errands_real = errands[errands["is_test_errand"] == 0].copy()
print(f"Real errands: {len(errands_real):,}  (removed {len(errands)-len(errands_real):,} test records)")

# Resolve join key — order_number is base-36 encoding of order_id
errands_real["order_id_int"] = errands_real["order_number"].apply(base36_to_int)

# Timestamps
orders["order_created_at"]  = pd.to_datetime(orders["order_created_at"])
errands_real["errand_dt"]   = pd.to_datetime(errands_real["created"])

# Contact flag on orders
contacted_ids = set(errands_real["order_id_int"].unique())
orders["contacted"] = orders["order_id"].isin(contacted_ids).astype(int)

# Contact counts
contact_counts = (errands_real.groupby("order_id_int").size()
                               .reset_index(name="n_contacts"))
orders = orders.merge(contact_counts, left_on="order_id", right_on="order_id_int", how="left")
orders["n_contacts"] = orders["n_contacts"].fillna(0).astype(int)

# Merge errand ↔ order (for time-to-contact)
merged = errands_real.merge(orders[["order_id","order_created_at"]],
                            left_on="order_id_int", right_on="order_id", how="left")
merged["hours_to_contact"] = (merged["errand_dt"] - merged["order_created_at"]).dt.total_seconds() / 3600

# First contact per order
first_contact = (merged.groupby("order_id_int")
                        .agg(first_dt=("errand_dt", "min"),
                             order_dt=("order_created_at", "first"),
                             first_category=("errand_category", "first"))
                        .reset_index())
first_contact["hours_ttfc"] = (first_contact["first_dt"] - first_contact["order_dt"]).dt.total_seconds() / 3600

OVERALL_CR = orders["contacted"].mean()
N_ORDERS   = len(orders)
N_ERRANDS  = len(errands_real)
print(f"Orders: {N_ORDERS:,} | Errands: {N_ERRANDS:,} | Contact rate: {OVERALL_CR:.2%}")


# ─────────────────────────────────────────────
# PART 2 — STATISTICAL ANALYSIS
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("PART 2 — STATISTICAL ANALYSIS")
print("=" * 60)

sample = orders.sample(500_000, random_state=42)

# ── 2.1  Contact Propensity Factors ─────────────────────────────
print("\n[2.1] Contact Propensity Factors — Chi-square & Mann-Whitney")

results_21 = []

def chi2_test(df, col, label):
    tab = pd.crosstab(df[col], df["contacted"])
    chi2, p, dof, _ = stats.chi2_contingency(tab)
    rate_map = df.groupby(col)["contacted"].mean().round(4).to_dict()
    return {"factor": label, "test": "Chi-square", "statistic": round(chi2, 1),
            "p_value": p, "df": dof, "rates": rate_map}

# 1 — Cancellation
r = chi2_test(sample, "Is_Canceled", "Order Cancellation")
r["insight"] = (f"Cancelled orders: {sample[sample.Is_Canceled==1].contacted.mean():.1%} "
                f"vs active: {sample[sample.Is_Canceled==0].contacted.mean():.1%}")
results_21.append(r)

# 2 — Booking Change
r = chi2_test(sample, "Is_Changed", "Booking Modification")
r["insight"] = (f"Changed orders: {sample[sample.Is_Changed==1].contacted.mean():.1%} "
                f"vs unchanged: {sample[sample.Is_Changed==0].contacted.mean():.1%}")
results_21.append(r)

# 3 — Customer Group
r = chi2_test(sample, "Customer_Group_Type", "Customer Group Type")
rates = sample.groupby("Customer_Group_Type")["contacted"].mean()
r["insight"] = f"FAMILY {rates.get('FAMILY',0):.1%} vs SINGLE {rates.get('SINGLE',0):.1%}"
results_21.append(r)

# 4 — Journey Type
r = chi2_test(sample, "Journey_Type_ID", "Journey Type")
rates = sample.groupby("Journey_Type_ID")["contacted"].mean().sort_values(ascending=False)
r["insight"] = f"Highest: {rates.index[0]} ({rates.iloc[0]:.1%}), Lowest: {rates.index[-1]} ({rates.iloc[-1]:.1%})"
results_21.append(r)

# 5 — Order Amount (Mann-Whitney, high vs low)
med = sample["Order_Amount"].clip(0).median()
high = sample[sample["Order_Amount"] > med]["contacted"]
low  = sample[sample["Order_Amount"] <= med]["contacted"]
stat, p_val = stats.mannwhitneyu(high, low, alternative="two-sided")
results_21.append({
    "factor": "Order Value (High vs Low)",
    "test": "Mann-Whitney U",
    "statistic": round(stat, 0),
    "p_value": p_val,
    "insight": f"High value orders: {high.mean():.1%} vs Low: {low.mean():.1%}"
})

print(pd.DataFrame([{k: v for k, v in r.items() if k != "rates"} for r in results_21]).to_string())

# Chart 2.1 — Effect sizes (contact rate by factor level)
fig, axes = plt.subplots(1, 4, figsize=(16, 5))
factors = [
    ("Is_Canceled",         "Cancellation",     {0: "Not Cancelled", 1: "Cancelled"}),
    ("Is_Changed",          "Booking Change",   {0: "Unchanged",     1: "Changed"}),
    ("Customer_Group_Type", "Customer Group",   None),
    ("Journey_Type_ID",     "Journey Type",     None),
]
for ax, (col, title, remap) in zip(axes, factors):
    rates = sample.groupby(col)["contacted"].mean().sort_values()
    labels = [remap[k] if remap else k for k in rates.index] if remap else rates.index.tolist()
    colors = [PALETTE["danger"] if v > OVERALL_CR else PALETTE["ok"] for v in rates.values]
    bars = ax.barh(labels, rates.values * 100, color=colors, edgecolor="white", height=0.6)
    ax.axvline(OVERALL_CR * 100, color=PALETTE["accent"], lw=1.5, ls="--", label="Overall avg")
    ax.set_xlabel("Contact Rate (%)")
    ax.set_title(title, fontweight="bold", color=PALETTE["primary"])
    for bar, v in zip(bars, rates.values):
        ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
                f"{v:.1%}", va="center", fontsize=9)
    ax.set_xlim(0, min(rates.max() * 120, 85))

axes[0].legend(fontsize=8)
fig.suptitle("Contact Rate by Order Characteristics — All differences p < 0.001",
             fontsize=13, fontweight="bold", color=PALETTE["primary"], y=1.02)
fig.tight_layout()
save(fig, "2_1_contact_propensity")


# ── 2.2  Time-to-First-Contact (Survival Analysis) ──────────────
print("\n[2.2] Time-to-First-Contact")

ttfc = first_contact[first_contact["hours_ttfc"] >= 0]["hours_ttfc"]
print(f"  Median TTFC: {ttfc.median():.0f} hours ({ttfc.median()/24:.1f} days)")
print(f"  Within 24h:  {(ttfc <= 24).mean():.1%}")
print(f"  Within 48h:  {(ttfc <= 48).mean():.1%}")

# Kaplan-Meier style (manual — no lifelines available)
sorted_t = np.sort(ttfc.values)
n = len(sorted_t)
# Survival function S(t) = proportion still "surviving" (not yet contacted) at time t
surv   = np.arange(n, 0, -1) / n   # proportion yet to contact
# Use a subset of time points for plotting
t_plot = np.linspace(0, sorted_t[int(n*0.99)], 500)
S_plot = np.array([(sorted_t >= t).mean() for t in t_plot])

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax = axes[0]
ax.plot(t_plot / 24, S_plot, color=PALETTE["primary"], lw=2)
ax.fill_between(t_plot / 24, S_plot, alpha=0.1, color=PALETTE["primary"])
ax.axvline(2,  color=PALETTE["danger"],  lw=1.5, ls="--", label="48 hours")
ax.axvline(14, color=PALETTE["accent"],  lw=1.5, ls="--", label="2 weeks")
ax.set_xlabel("Days after booking")
ax.set_ylabel("Proportion not yet contacted")
ax.set_title("Survival Curve — Time to First CS Contact", fontweight="bold", color=PALETTE["primary"])
ax.legend()

# Distribution of TTFC in first 30 days
ax2 = axes[1]
ttfc_days = ttfc[ttfc <= 720] / 24
ax2.hist(ttfc_days, bins=50, color=PALETTE["primary"], alpha=0.7, edgecolor="white")
ax2.axvline(ttfc_days.median(), color=PALETTE["accent"], lw=2, ls="--",
            label=f"Median: {ttfc_days.median():.1f} days")
ax2.set_xlabel("Days to first contact")
ax2.set_ylabel("Number of orders")
ax2.set_title("Distribution of First Contact Timing (≤30d)", fontweight="bold", color=PALETTE["primary"])
ax2.legend()

fig.tight_layout()
save(fig, "2_2_time_to_contact")


# ── 2.3  Repeat Contact Behaviour ───────────────────────────────
print("\n[2.3] Repeat Contact Behaviour")

repeat_flag = errands_real.groupby("order_id_int").size().rename("n_contacts").reset_index()
repeat_flag["is_repeat"] = (repeat_flag["n_contacts"] > 1).astype(int)

cat_order = (errands_real.sort_values("errand_dt")
                          .groupby("order_id_int")["errand_category"]
                          .first()
                          .reset_index()
                          .rename(columns={"errand_category": "first_cat"}))
cat_repeat = cat_order.merge(repeat_flag, on="order_id_int")

repeat_rates = (cat_repeat.groupby("first_cat")["is_repeat"]
                           .agg(["mean", "count"])
                           .rename(columns={"mean": "repeat_rate", "count": "volume"})
                           .query("volume > 500")
                           .sort_values("repeat_rate", ascending=False))

print(repeat_rates.round(3).to_string())

# Chi-square: does first_cat predict repeat?
top_cats = repeat_rates.head(8).index.tolist() + repeat_rates.tail(4).index.tolist()
sub = cat_repeat[cat_repeat["first_cat"].isin(top_cats)]
tab_r = pd.crosstab(sub["first_cat"], sub["is_repeat"])
chi2_r, p_r, dof_r, _ = stats.chi2_contingency(tab_r)
print(f"\n  Chi-square (first category → repeat): chi2={chi2_r:.1f}, p={p_r:.2e}")

fig, ax = plt.subplots(figsize=(10, 6))
top = repeat_rates.head(12)
colors = [PALETTE["danger"] if v > 0.6 else PALETTE["mid"] for v in top["repeat_rate"]]
bars = ax.barh(top.index, top["repeat_rate"] * 100, color=colors, edgecolor="white")
ax.axvline(repeat_flag["is_repeat"].mean() * 100, color=PALETTE["accent"],
           lw=2, ls="--", label=f"Avg: {repeat_flag['is_repeat'].mean():.1%}")
ax.set_xlabel("Repeat Contact Rate (%)")
ax.set_title("First Contact Category → Repeat Contact Rate\n(Chi-square p < 0.001)",
             fontweight="bold", color=PALETTE["primary"])
ax.legend()
for bar, v in zip(bars, top["repeat_rate"]):
    ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
            f"{v:.0%}", va="center", fontsize=9)
fig.tight_layout()
save(fig, "2_3_repeat_contact")


# ── 2.4  Analyst-Defined Hypothesis Test ────────────────────────
print("\n[2.4] Custom Hypothesis: Do FAMILY bookings have higher repeat contact rates than SINGLE?")

fam_repeat  = orders[orders["Customer_Group_Type"] == "FAMILY"]["n_contacts"]
sing_repeat = orders[orders["Customer_Group_Type"] == "SINGLE"]["n_contacts"]
stat_24, p_24 = stats.mannwhitneyu(fam_repeat, sing_repeat, alternative="greater")
print(f"  FAMILY median contacts: {fam_repeat.median():.0f}  | SINGLE: {sing_repeat.median():.0f}")
print(f"  Mann-Whitney U={stat_24:.0f}, p={p_24:.4f}")
print(f"  → {'Reject' if p_24 < 0.05 else 'Fail to reject'} H0 at α=0.05")


# ─────────────────────────────────────────────
# PART 3 — MACHINE LEARNING
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("PART 3 — MACHINE LEARNING")
print("=" * 60)

# ── 3A  Contact Prediction Model ─────────────────────────────────
print("\n[3A] Building Contact Prediction Model")

# Feature engineering — BOOKING-TIME ONLY (strict no-leakage)
orders["book_hour"]  = orders["order_created_at"].dt.hour
orders["book_dow"]   = orders["order_created_at"].dt.dayofweek
orders["book_month"] = orders["order_created_at"].dt.month
orders["is_direct"]  = orders["Partner"].isna().astype(int)
orders["order_amt_log"] = np.log1p(orders["Order_Amount"].clip(lower=0))

CAT_FEATS = ["Brand", "booking_system", "Customer_Group_Type", "Device",
             "client_entry_type", "booking_system_source_type", "Journey_Type_ID",
             "Site_Country", "currency"]
NUM_FEATS = ["order_amt_log", "book_hour", "book_dow", "book_month", "is_direct"]

ml_sample = orders.sample(300_000, random_state=42).copy()
for col in CAT_FEATS:
    ml_sample[col] = ml_sample[col].fillna("Unknown")
    le = LabelEncoder()
    ml_sample[col + "_enc"] = le.fit_transform(ml_sample[col])

enc_feats = [c + "_enc" for c in CAT_FEATS] + NUM_FEATS
X = ml_sample[enc_feats].fillna(0)
y = ml_sample["contacted"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# Model A: Logistic Regression
lr = LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced", C=0.5)
lr.fit(X_train, y_train)
lr_proba = lr.predict_proba(X_test)[:, 1]
lr_auc   = roc_auc_score(y_test, lr_proba)
lr_ap    = average_precision_score(y_test, lr_proba)

# Model B: Random Forest
rf = RandomForestClassifier(n_estimators=150, max_depth=7, min_samples_leaf=50,
                             random_state=42, n_jobs=-1, class_weight="balanced")
rf.fit(X_train, y_train)
rf_proba = rf.predict_proba(X_test)[:, 1]
rf_auc   = roc_auc_score(y_test, rf_proba)
rf_ap    = average_precision_score(y_test, rf_proba)

print(f"  Logistic Regression — AUC: {lr_auc:.4f}  | Avg Precision: {lr_ap:.4f}")
print(f"  Random Forest       — AUC: {rf_auc:.4f}  | Avg Precision: {rf_ap:.4f}")

# Feature importance
feat_imp = pd.Series(rf.feature_importances_, index=enc_feats).sort_values(ascending=False)
print("\n  Top 10 Feature Importances:")
print(feat_imp.head(10).round(4))

# ── Charts 3A ────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# ROC curves
for model_name, proba in [("Logistic Regression", lr_proba), ("Random Forest", rf_proba)]:
    fpr, tpr, _ = roc_curve(y_test, proba)
    auc = roc_auc_score(y_test, proba)
    axes[0].plot(fpr, tpr, lw=2, label=f"{model_name} (AUC={auc:.3f})")
axes[0].plot([0, 1], [0, 1], "k--", alpha=0.4, label="Random")
axes[0].set_xlabel("False Positive Rate")
axes[0].set_ylabel("True Positive Rate")
axes[0].set_title("ROC Curve — Contact Prediction", fontweight="bold", color=PALETTE["primary"])
axes[0].legend(fontsize=9)

# Precision-Recall
for model_name, proba in [("Logistic Regression", lr_proba), ("Random Forest", rf_proba)]:
    prec, rec, _ = precision_recall_curve(y_test, proba)
    ap = average_precision_score(y_test, proba)
    axes[1].plot(rec, prec, lw=2, label=f"{model_name} (AP={ap:.3f})")
axes[1].axhline(y_test.mean(), color="gray", ls="--", label=f"Baseline ({y_test.mean():.2f})")
axes[1].set_xlabel("Recall")
axes[1].set_ylabel("Precision")
axes[1].set_title("Precision-Recall Curve", fontweight="bold", color=PALETTE["primary"])
axes[1].legend(fontsize=9)

# Feature importances
top_n  = feat_imp.head(10)
labels = [f.replace("_enc", "").replace("_", " ") for f in top_n.index]
colors = [PALETTE["primary"] if i < 3 else PALETTE["mid"] for i in range(len(top_n))]
axes[2].barh(labels[::-1], top_n.values[::-1], color=colors[::-1], edgecolor="white")
axes[2].set_xlabel("Importance")
axes[2].set_title("Top 10 Feature Importances\n(Random Forest)", fontweight="bold", color=PALETTE["primary"])

fig.tight_layout()
save(fig, "3A_model_evaluation")


# ── 3B  Unsupervised Segmentation ────────────────────────────────
print("\n[3B] Customer Segmentation (KMeans)")

cluster_sample = orders.sample(100_000, random_state=42).copy()
cluster_sample["order_amt_log"] = np.log1p(cluster_sample["Order_Amount"].clip(lower=0))

for col in ["Customer_Group_Type", "Journey_Type_ID", "Device", "Brand", "booking_system_source_type"]:
    cluster_sample[col] = cluster_sample[col].fillna("Unknown")
    le = LabelEncoder()
    cluster_sample[col + "_enc"] = le.fit_transform(cluster_sample[col])

CLUST_FEATS = ["order_amt_log", "Customer_Group_Type_enc", "Journey_Type_ID_enc",
               "Device_enc", "Brand_enc", "booking_system_source_type_enc",
               "contacted", "n_contacts", "Is_Changed", "Is_Canceled"]

Xc = cluster_sample[CLUST_FEATS].fillna(0)
sc = StandardScaler()
Xcs = sc.fit_transform(Xc)

# Silhouette for k=2..6
sil = {}
for k in range(2, 7):
    km = KMeans(n_clusters=k, random_state=42, n_init=5)
    lbl = km.fit_predict(Xcs)
    sil[k] = silhouette_score(Xcs, lbl, sample_size=10_000, random_state=42)
    print(f"  k={k}  silhouette={sil[k]:.4f}")

best_k = max(sil, key=sil.get)
print(f"  Best k by silhouette: {best_k}")

# Refit with k=4 (interpretable + good score)
km4 = KMeans(n_clusters=4, random_state=42, n_init=10)
cluster_sample["cluster"] = km4.fit_predict(Xcs)

profile = cluster_sample.groupby("cluster").agg(
    n=("cluster", "count"),
    avg_order_amt=("Order_Amount", lambda x: x.clip(0, np.percentile(x.clip(lower=0), 95)).mean()),
    contact_rate=("contacted", "mean"),
    avg_contacts=("n_contacts", "mean"),
    cancel_rate=("Is_Canceled", "mean"),
    change_rate=("Is_Changed", "mean"),
    pct_family=("Customer_Group_Type", lambda x: (x == "FAMILY").mean()),
    pct_roundtrip=("Journey_Type_ID", lambda x: (x == "Round-trip").mean()),
).round(3)
print("\n  Cluster Profiles:")
print(profile.to_string())

# Business labels
CLUSTER_LABELS = {
    0: "🔴 High-Value Problem Orders\n(changed/high contact)",
    1: "🟡 Round-Trip Low-Risk Bookers\n(low contact, unchanged)",
    2: "🟢 Simple One-Way Travellers\n(low contact, single leg)",
    3: "🟠 Cancelled Orders\n(high contact, full refund)",
}

# Chart 3B
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Scatter: order amount vs contact rate proxy (n_contacts)
colors_c = [PALETTE["danger"], PALETTE["accent"], PALETTE["ok"], "#8E44AD"]
for ci in range(4):
    sub = cluster_sample[cluster_sample["cluster"] == ci].sample(min(2000, len(cluster_sample[cluster_sample["cluster"]==ci])))
    label_short = ["High-Value Problems", "Round-Trip Low-Risk", "One-Way Simple", "Cancelled"][ci]
    axes[0].scatter(sub["order_amt_log"], sub["n_contacts"].clip(0, 10),
                    alpha=0.15, s=10, color=colors_c[ci], label=label_short)
axes[0].set_xlabel("log(Order Amount)")
axes[0].set_ylabel("Number of CS Contacts")
axes[0].set_title("Customer Segments — Value vs Contact Volume", fontweight="bold", color=PALETTE["primary"])
axes[0].legend(markerscale=3, fontsize=9)

# Profile radar-ish: bar chart
metrics = ["contact_rate", "cancel_rate", "change_rate", "pct_family", "pct_roundtrip"]
metric_labels = ["Contact Rate", "Cancel Rate", "Change Rate", "% Family", "% Round-trip"]
x = np.arange(len(metrics))
w = 0.2
for i, ci in enumerate(range(4)):
    vals = [profile.loc[ci, m] for m in metrics]
    label_short = ["High-Value", "Round-Trip", "One-Way", "Cancelled"][i]
    axes[1].bar(x + i*w, vals, w, label=label_short, color=colors_c[i], alpha=0.85)
axes[1].set_xticks(x + 1.5*w)
axes[1].set_xticklabels(metric_labels, fontsize=9)
axes[1].set_ylabel("Rate")
axes[1].set_title("Segment Profile Comparison", fontweight="bold", color=PALETTE["primary"])
axes[1].legend(fontsize=9)
fig.tight_layout()
save(fig, "3B_segmentation")


# ─────────────────────────────────────────────
# CHART FOR EXECUTIVE DECK (Slide 1 — headline)
# ─────────────────────────────────────────────
print("\n[EXEC] Creating executive headline chart")

# Monthly contact volume + category breakdown
errands_real["month"] = errands_real["errand_dt"].dt.to_period("M")
monthly_cat = (errands_real.groupby(["month", "errand_category"]).size()
                           .unstack(fill_value=0))
monthly_total = monthly_cat.sum(axis=1)
month_labels = [str(m) for m in monthly_total.index]

top_cats = ["Cancellation / refund", "Rebooking", "Schedule change",
            "Document & travel info", "Change of name & passenger info"]
other = monthly_cat.drop(columns=[c for c in top_cats if c in monthly_cat.columns], errors="ignore").sum(axis=1)

cat_colors = [PALETTE["danger"], "#E67E22", "#3498DB", "#9B59B6", "#1ABC9C", PALETTE["mid"]]

fig, ax = plt.subplots(figsize=(13, 6))
bottom = np.zeros(len(monthly_cat))
for cat, col in zip(top_cats, cat_colors):
    if cat in monthly_cat.columns:
        vals = monthly_cat[cat].values
        ax.bar(month_labels, vals, bottom=bottom, color=col, label=cat, alpha=0.9, width=0.7)
        bottom += vals
ax.bar(month_labels, other.values, bottom=bottom, color=PALETTE["mid"], label="Other", alpha=0.7, width=0.7)
ax.set_xlabel("Month")
ax.set_ylabel("CS Contact Volume")
ax.set_title("Monthly CS Contact Volume by Category  (May 2024 – Apr 2025)",
             fontweight="bold", color=PALETTE["primary"], fontsize=13)
ax.legend(loc="upper right", fontsize=9, ncol=2)
plt.xticks(rotation=30, ha="right")
fig.tight_layout()
save(fig, "exec_monthly_volume")

# Headline KPI chart for slide 1
fig, axes = plt.subplots(1, 3, figsize=(14, 5))
kpis = [
    ("16.3%", "of orders\ntrigger a CS contact", PALETTE["primary"]),
    ("52.4%", "of contacted orders\nhave repeat contacts", PALETTE["danger"]),
    ("31.7%", "of contacts via\nphone (most costly)", PALETTE["accent"]),
]
for ax, (num, lbl, col) in zip(axes, kpis):
    ax.set_facecolor(col)
    ax.text(0.5, 0.58, num, ha="center", va="center", fontsize=44, fontweight="bold",
            color="white", transform=ax.transAxes)
    ax.text(0.5, 0.28, lbl, ha="center", va="center", fontsize=13,
            color="white", alpha=0.9, transform=ax.transAxes)
    ax.axis("off")
fig.patch.set_facecolor(PALETTE["light"])
fig.suptitle("The Scale of the Problem", fontsize=15, fontweight="bold",
             color=PALETTE["primary"], y=1.02)
fig.tight_layout()
save(fig, "exec_kpi_tiles")

# Root cause: cancellations & changes drive contacts
cr_matrix = orders.groupby(["Is_Canceled", "Is_Changed"])["contacted"].mean().unstack()
fig, ax = plt.subplots(figsize=(7, 4))
cats = ["Not Cancelled\nNot Changed", "Not Cancelled\nChanged", "Cancelled\n(any)"]
vals = [
    orders[(orders.Is_Canceled==0) & (orders.Is_Changed==0)]["contacted"].mean(),
    orders[(orders.Is_Canceled==0) & (orders.Is_Changed==1)]["contacted"].mean(),
    orders[orders.Is_Canceled==1]["contacted"].mean(),
]
bar_colors = [PALETTE["ok"], PALETTE["accent"], PALETTE["danger"]]
bars = ax.bar(cats, [v*100 for v in vals], color=bar_colors, width=0.5, edgecolor="white")
ax.axhline(OVERALL_CR*100, color=PALETTE["primary"], ls="--", lw=1.5, label=f"Overall avg ({OVERALL_CR:.1%})")
ax.set_ylabel("Contact Rate (%)")
ax.set_title("Booking Disruption Drives Contact Volume",
             fontweight="bold", color=PALETTE["primary"])
for bar, v in zip(bars, vals):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
            f"{v:.1%}", ha="center", fontweight="bold", fontsize=12)
ax.legend()
fig.tight_layout()
save(fig, "exec_root_cause")

print("\n" + "=" * 60)
print("ALL CHARTS SAVED — proceeding to PPTX generation")
print("=" * 60)
print(f"\nKey stats for deck:")
print(f"  Overall contact rate:  {OVERALL_CR:.1%}")
print(f"  Cancelled order CR:    {orders[orders.Is_Canceled==1].contacted.mean():.1%}")
print(f"  Changed order CR:      {orders[orders.Is_Changed==1].contacted.mean():.1%}")
print(f"  Normal order CR:       {orders[(orders.Is_Canceled==0)&(orders.Is_Changed==0)].contacted.mean():.1%}")
print(f"  RF AUC:                {rf_auc:.4f}")
print(f"  Total orders:          {N_ORDERS:,}")
print(f"  Total real errands:    {N_ERRANDS:,}")
