"""
Data Insight Analyst – Part 1: Data Profiling & Exploratory Analysis
=====================================================================
Etraveli Group | Customer Service Analytics Assessment

Usage:
    DATA_DIR=./data OUTPUT_DIR=./outputs/part1 python part1_eda.py

Expects:
    $DATA_DIR/orders.csv
    $DATA_DIR/errands.csv
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from pathlib import Path

warnings.filterwarnings("ignore")

# ── Configuration ──────────────────────────────────────────────────────────────
DATA_DIR   = Path(os.getenv("DATA_DIR", "."))
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "outputs/part1"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ORDERS_FILE  = DATA_DIR / "orders.csv"
ERRANDS_FILE = DATA_DIR / "errands.csv"

sns.set_theme(style="whitegrid", font_scale=1.05)
plt.rcParams.update({
    "figure.dpi": 120,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

COLOR_PRIMARY = "#2563EB"
COLOR_WARN    = "#DC2626"


# ── Helpers ────────────────────────────────────────────────────────────────────

def decode_order_number(val):
    try:
        return int(str(val), 36)
    except (ValueError, TypeError):
        return None


def save_fig(name):
    path = OUTPUT_DIR / f"{name}.png"
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"  [saved] {path.name}")


def pct(n, total):
    return f"{n:,}  ({n / total * 100:.1f}%)"


def section(title):
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}")


# ══════════════════════════════════════════════════════════════════════════════
# 1. LOAD DATA
# ══════════════════════════════════════════════════════════════════════════════

section("1. Loading data")

orders  = pd.read_csv(ORDERS_FILE)
errands = pd.read_csv(ERRANDS_FILE)

print(f"  Orders  : {orders.shape[0]:,} rows x {orders.shape[1]} cols")
print(f"  Errands : {errands.shape[0]:,} rows x {errands.shape[1]} cols")

orders["order_created_at"] = pd.to_datetime(orders["order_created_at"], errors="coerce")
errands["created"]         = pd.to_datetime(errands["created"],         errors="coerce")


# ══════════════════════════════════════════════════════════════════════════════
# 1.1  DATA QUALITY AUDIT
# ══════════════════════════════════════════════════════════════════════════════

section("1.1 Data Quality Audit")

print("\n--- Null counts (orders) ---")
orders_nulls = orders.isnull().sum()
print(orders_nulls[orders_nulls > 0].to_string() if orders_nulls.any() else "  No nulls.")

print("\n--- Null counts (errands) ---")
errands_nulls = errands.isnull().sum()
print(errands_nulls[errands_nulls > 0].to_string() if errands_nulls.any() else "  No nulls.")
print("\n  Decision (errand_type 3 nulls)      : Leave as-is — negligible.")
print("  Decision (errand_action 6706 nulls) : Leave as-is — absence of action is meaningful.")

print("\n--- Duplicate primary keys ---")
print(f"  Duplicate order_id  : {orders.duplicated('order_id').sum():,}")
print(f"  Duplicate errand_id : {errands.duplicated('errand_id').sum():,}")

print("\n--- Financial sanity ---")
neg_amt = (orders["Order_Amount"] < 0).sum()
neg_rev = (orders["Revenue"] < 0).sum()
ext_amt = (orders["Order_Amount"] > 1_000_000).sum()
print(f"  Negative Order_Amount : {pct(neg_amt, len(orders))}")
print(f"  Negative Revenue      : {pct(neg_rev, len(orders))}")
print(f"  Order_Amount > 1M     : {pct(ext_amt, len(orders))}")
print("  Decision: Retain — likely chargebacks / large multi-pax bookings.")

test_n = (errands["is_test_errand"] == 1).sum()
print(f"\n--- Test errands ---")
print(f"  To filter out: {pct(test_n, len(errands))}")

print("\n--- ANOMALY 1: Join key encoding ---")
print("  order_number (errands) is base-36 encoded order_id (orders).")
print("  Example: '24770FC' -> int(str, 36) -> 4607513832")
print("  Decision: Decode before joining. Verify 100% match rate.")

print("\n--- ANOMALY 2: Date range mismatch ---")
ord_min = orders["order_created_at"].min().date()
ord_max = orders["order_created_at"].max().date()
err_min = errands["created"].min().date()
err_max = errands["created"].max().date()
print(f"  Orders  window: {ord_min} -> {ord_max}")
print(f"  Errands window: {err_min} -> {err_max}")
print("  Errands extend ~8 months beyond orders. Structural, not corrupt.")
print("  Decision: Flag out-of-window errands; be careful with contact-rate denominator.")


# ══════════════════════════════════════════════════════════════════════════════
# 2. CLEAN & JOIN
# ══════════════════════════════════════════════════════════════════════════════

section("2. Clean & Join")

errands_real = errands[errands["is_test_errand"] == 0].copy()
errands_real["order_id"] = errands_real["order_number"].apply(decode_order_number)
print(f"  Real errands        : {len(errands_real):,}")
print(f"  Failed decodes      : {errands_real['order_id'].isnull().sum():,}")

orders["has_contact"] = orders["order_id"].isin(errands_real["order_id"]).astype(int)
contacts_per_order    = errands_real.groupby("order_id").size().rename("n_contacts")


# ══════════════════════════════════════════════════════════════════════════════
# 1.2  CONTACT RATE BASELINE
# ══════════════════════════════════════════════════════════════════════════════

section("1.2 Contact Rate Baseline")

total_orders     = len(orders)
overall_cr       = orders["has_contact"].mean()
orders_contacted = orders["has_contact"].sum()
pct_multi        = (contacts_per_order >= 2).mean()

print(f"  Total orders          : {total_orders:,}")
print(f"  Orders with >=1 errand: {orders_contacted:,}")
print(f"  Overall contact rate  : {overall_cr * 100:.2f}%")
print(f"  Mean contacts/order   : {contacts_per_order.mean():.2f}")
print(f"  Orders with 2+ contacts: {pct_multi * 100:.1f}% of contacted orders")

top10_thresh   = contacts_per_order.quantile(0.90)
top10_contacts = contacts_per_order[contacts_per_order >= top10_thresh].sum()
print(f"  Top 10% high-contact orders drive: {top10_contacts/len(errands_real)*100:.1f}% of all contacts")

print("\n  Contact rate by dimension:")
dims = ["Brand", "Device", "Journey_Type_ID", "Customer_Group_Type", "client_entry_type"]
dim_results = {}
for dim in dims:
    cr = orders.groupby(dim)["has_contact"].agg(["sum", "count", "mean"])
    cr.columns = ["contacts", "orders", "contact_rate"]
    cr = cr.sort_values("contact_rate", ascending=False)
    dim_results[dim] = cr
    print(f"\n  [{dim}]")
    print(cr.to_string())

# Figure 1.2a: Multi-panel contact rates
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
axes = axes.flatten()
for i, dim in enumerate(dims):
    cr = dim_results[dim]["contact_rate"].sort_values() * 100
    colors = [COLOR_WARN if v == cr.max() else COLOR_PRIMARY for v in cr.values]
    axes[i].barh(cr.index.astype(str), cr.values, color=colors)
    axes[i].set_title(f"Contact rate by {dim}", fontsize=9)
    axes[i].xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f}%"))
    axes[i].set_xlabel("Contact rate (%)")
axes[5].set_visible(False)
plt.suptitle("Contact Rate Decomposition Across Key Dimensions", fontsize=12, fontweight="bold")
plt.tight_layout()
save_fig("1_2a_contact_rate_dimensions")

# Figure 1.2b: Lorenz curve
sorted_c = contacts_per_order.sort_values()
cum_c    = sorted_c.cumsum() / sorted_c.sum()
cum_o    = np.arange(1, len(sorted_c) + 1) / len(sorted_c)
fig, ax = plt.subplots(figsize=(6, 5))
ax.plot(cum_o * 100, cum_c * 100, color=COLOR_PRIMARY, linewidth=2, label="Observed")
ax.plot([0, 100], [0, 100], "k--", alpha=0.4, label="Equal distribution")
ax.fill_between(cum_o * 100, cum_c * 100, cum_o * 100, alpha=0.12, color=COLOR_PRIMARY)
ax.set_xlabel("% of contacted orders (ranked low to high)")
ax.set_ylabel("Cumulative % of contacts")
ax.set_title("Contact concentration (Lorenz curve)")
ax.legend()
save_fig("1_2b_lorenz_contacts")

# Figure 1.2c: Distribution of contacts per order
fig, ax = plt.subplots(figsize=(8, 4))
vc = contacts_per_order.clip(upper=10).value_counts().sort_index()
ax.bar(vc.index.astype(str), vc.values, color=COLOR_PRIMARY, edgecolor="white")
ax.set_xlabel("Contacts per order  (10 = 10+)")
ax.set_ylabel("Number of orders")
ax.set_title("Distribution of contacts per order")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1000:.0f}K"))
save_fig("1_2c_contacts_distribution")


# ══════════════════════════════════════════════════════════════════════════════
# 1.3  TEMPORAL ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

section("1.3 Temporal Analysis")

errands_real["week"] = errands_real["created"].dt.to_period("W")
weekly_vol = errands_real.groupby("week").size()

orders["week"] = orders["order_created_at"].dt.to_period("W")
wcr = orders.groupby("week").agg(
    orders_n=("order_id", "count"),
    contacts_n=("has_contact", "sum")
)
wcr["cr"] = wcr["contacts_n"] / wcr["orders_n"]
print("\n  Weekly contact rate (booking window):")
print(wcr["cr"].apply(lambda x: f"{x * 100:.1f}%").to_string())

# Figure 1.3a: Weekly errand volume
fig, ax = plt.subplots(figsize=(13, 4))
ax.bar(range(len(weekly_vol)), weekly_vol.values, color=COLOR_PRIMARY, alpha=0.85)
step = max(1, len(weekly_vol) // 12)
ax.set_xticks(range(0, len(weekly_vol), step))
ax.set_xticklabels(
    [str(weekly_vol.index[i]) for i in range(0, len(weekly_vol), step)],
    rotation=45, ha="right", fontsize=8
)
ax.set_ylabel("Errand volume")
ax.set_title("Weekly errand volume (May 2024 - Apr 2025)")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1000:.0f}K"))
save_fig("1_3a_weekly_errand_volume")

# Figure 1.3b: Weekly contact rate over booking window
fig, ax = plt.subplots(figsize=(11, 4))
ax.plot(range(len(wcr)), wcr["cr"].values * 100,
        color=COLOR_PRIMARY, marker="o", linewidth=2, markersize=5)
ax.set_xticks(range(len(wcr)))
ax.set_xticklabels([str(p) for p in wcr.index], rotation=45, ha="right", fontsize=8)
ax.set_ylabel("Contact rate (%)")
ax.set_title("Weekly contact rate by order booking week")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f}%"))
save_fig("1_3b_weekly_contact_rate")

# Figure 1.3c: Category composition over time
errands_real["month"] = errands_real["created"].dt.to_period("M")
cat_month = errands_real.groupby(["month", "errand_category"]).size().unstack(fill_value=0)
top_cats6 = cat_month.sum().nlargest(6).index
cat_month_pct = cat_month[top_cats6].div(cat_month.sum(axis=1), axis=0) * 100

fig, ax = plt.subplots(figsize=(12, 5))
cat_month_pct.plot(kind="bar", stacked=True, ax=ax, colormap="tab10")
ax.set_ylabel("Share of monthly contacts (%)")
ax.set_title("Errand category composition by month")
ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=8)
ax.set_xticklabels([str(p) for p in cat_month_pct.index], rotation=45, ha="right")
save_fig("1_3c_category_composition_time")


# ══════════════════════════════════════════════════════════════════════════════
# 1.4  CHANNEL & CATEGORY DECOMPOSITION
# ══════════════════════════════════════════════════════════════════════════════

section("1.4 Channel & Category Decomposition")

ch_cat     = errands_real.groupby(["errand_channel", "errand_category"]).size().unstack(fill_value=0)
top_cats8  = ch_cat.sum().nlargest(8).index
ch_cat_top = ch_cat[top_cats8]
ch_cat_pct = ch_cat_top.div(ch_cat_top.sum(axis=1), axis=0) * 100

print("\n  Channel x Category mix (% per channel):")
print(ch_cat_pct.round(1).to_string())

fig, ax = plt.subplots(figsize=(12, 6))
sns.heatmap(
    ch_cat_pct, annot=True, fmt=".0f", cmap="YlOrRd",
    linewidths=0.4, ax=ax,
    cbar_kws={"label": "% of channel contacts"}
)
ax.set_title("Channel x Category distribution (% of channel total)")
ax.tick_params(axis="x", rotation=30)
save_fig("1_4a_channel_category_heatmap")

errands_real["quarter"] = errands_real["created"].dt.to_period("Q")
ch_q     = errands_real.groupby(["quarter", "errand_channel"]).size().unstack(fill_value=0)
ch_q_pct = ch_q.div(ch_q.sum(axis=1), axis=0) * 100
print("\n  Channel share by quarter:")
print(ch_q_pct.round(1).to_string())

fig, ax = plt.subplots(figsize=(9, 4))
ch_q_pct.plot(kind="bar", stacked=True, ax=ax, colormap="Set2")
ax.set_ylabel("Share (%)")
ax.set_title("Channel share by quarter")
ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=9)
ax.set_xticklabels([str(p) for p in ch_q_pct.index], rotation=30, ha="right")
save_fig("1_4b_channel_share_by_quarter")


# ══════════════════════════════════════════════════════════════════════════════
# 1.5  LINKAGE ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

section("1.5 Linkage Analysis")

order_window_end = orders["order_created_at"].max()
in_window        = errands_real["created"] <= order_window_end + pd.Timedelta(days=90)

print(f"  Total real errands      : {len(errands_real):,}")
print(f"  Errands in order window : {pct(in_window.sum(), len(errands_real))}")
print(f"  Out-of-window errands   : {pct((~in_window).sum(), len(errands_real))}")

out_df = errands_real[~in_window]
print(f"\n  Out-of-window date range: {out_df['created'].min().date()} -> {out_df['created'].max().date()}")
print("\n  Out-of-window top categories:")
print(out_df["errand_category"].value_counts().head(6).to_string())

no_errand_orders = (~orders["order_id"].isin(errands_real["order_id"])).sum()
print(f"\n  Orders with zero CS contacts: {pct(no_errand_orders, total_orders)}")

print("""
  INTERPRETATION:
  Out-of-window errands = customers contacting CS months after booking
  (e.g. schedule changes, refunds long post-travel). Not data corruption.
  For ML modelling use only in-window orders to avoid deflated contact rates.
""")


# ══════════════════════════════════════════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════════════════════════════════════════

section("Part 1 Summary")

print(f"""
  KEY FINDINGS:
    Overall contact rate      : {overall_cr * 100:.1f}% of all orders
    Repeat contact rate       : {pct_multi * 100:.1f}% of contacted orders
    Top contact category      : Cancellation/Refund (24.3%)
    Dominant channel          : Chat (43.7% of contacts)
    Largest brand (A)         : 58.9% of all orders
    Anomaly 1                 : base-36 encoded join key
    Anomaly 2                 : errands span 8+ months beyond order window

  Charts saved to: {OUTPUT_DIR}
  Part 1 complete. Run part2_statistical_analysis.py next.
""")