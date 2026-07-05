
import os
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PLOT_DIR = os.path.join(BASE_DIR, "plots")
BEFORE_DIR = os.path.join(PLOT_DIR, "before_preprocessing")
AFTER_DIR = os.path.join(PLOT_DIR, "after_preprocessing")
os.makedirs(BEFORE_DIR, exist_ok=True)
os.makedirs(AFTER_DIR, exist_ok=True)

# =========================
# 1. IMPORT DỮ LIỆU
# =========================
stores = pd.read_csv(os.path.join(BASE_DIR, "stores.csv"))
oil = pd.read_csv(os.path.join(BASE_DIR, "oil.csv"))
holidays = pd.read_csv(os.path.join(BASE_DIR, "holidays_events.csv"))
transactions = pd.read_csv(os.path.join(BASE_DIR, "transactions.csv"))
test = pd.read_csv(os.path.join(BASE_DIR, "test.csv"))
sample_submission = pd.read_csv(os.path.join(BASE_DIR, "sample_submission.csv"))

# =========================
# 2. HÀM HỖ TRỢ VẼ BIỂU ĐỒ
# =========================
def save_current_plot(path):
    plt.tight_layout()
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.close()

def histogram(df, col, path, title=None):
    s = df[col].dropna()
    if s.empty:
        return
    plt.figure(figsize=(8, 5))
    plt.hist(s, bins=30)
    plt.title(title or f"Histogram of {col}")
    plt.xlabel(col)
    plt.ylabel("Frequency")
    save_current_plot(path)

def boxplot(df, col, path, title=None):
    s = df[col].dropna()
    if s.empty:
        return
    plt.figure(figsize=(7, 5))
    plt.boxplot(s, vert=True)
    plt.title(title or f"Boxplot of {col}")
    plt.ylabel(col)
    save_current_plot(path)

def scatterplot(df, x, y, path, title=None):
    temp = df[[x, y]].dropna()
    if temp.empty:
        return
    plt.figure(figsize=(8, 5))
    plt.scatter(temp[x], temp[y], alpha=0.35)
    plt.title(title or f"Scatter Plot: {x} vs {y}")
    plt.xlabel(x)
    plt.ylabel(y)
    save_current_plot(path)

def barchart(series, path, title, xlabel, ylabel):
    if series.empty:
        return
    plt.figure(figsize=(10, 5))
    series.plot(kind="bar")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(rotation=45, ha="right")
    save_current_plot(path)

def linechart(df, x, y, path, title=None):
    temp = df[[x, y]].dropna().sort_values(x)
    if temp.empty:
        return
    plt.figure(figsize=(11, 5))
    plt.plot(temp[x], temp[y])
    plt.title(title or f"Line Chart: {y} by {x}")
    plt.xlabel(x)
    plt.ylabel(y)
    plt.xticks(rotation=45)
    save_current_plot(path)

def heatmap_corr(df, cols, path, title="Correlation Heatmap"):
    temp = df[cols].dropna()
    if temp.empty or len(cols) < 2:
        return
    corr = temp.corr(numeric_only=True)
    plt.figure(figsize=(7, 6))
    plt.imshow(corr, aspect="auto")
    plt.colorbar()
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=45, ha="right")
    plt.yticks(range(len(corr.index)), corr.index)
    plt.title(title)
    for i in range(len(corr.index)):
        for j in range(len(corr.columns)):
            plt.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center")
    save_current_plot(path)

# =========================
# 3. PHÂN LOẠI BIẾN
# =========================
def classify_columns(df, name):
    rows = []
    for col in df.columns:
        dtype = str(df[col].dtype)
        if pd.api.types.is_bool_dtype(df[col]):
            nature = "Định tính nhị phân"
        elif pd.api.types.is_numeric_dtype(df[col]):
            # ID/mã số thường là định tính định danh dù kiểu số
            if "id" in col.lower() or "nbr" in col.lower() or col.lower() in ["cluster"]:
                nature = "Định tính định danh / mã hóa số"
            else:
                nature = "Định lượng"
        elif "date" in col.lower():
            nature = "Thời gian"
        else:
            nature = "Định tính"
        rows.append({
            "file": name,
            "column": col,
            "dtype": dtype,
            "missing": int(df[col].isna().sum()),
            "unique": int(df[col].nunique(dropna=True)),
            "classification": nature
        })
    return rows

classification = []
for name, df in {
    "stores": stores,
    "oil": oil,
    "holidays_events": holidays,
    "transactions": transactions,
    "test": test,
    "sample_submission": sample_submission
}.items():
    classification.extend(classify_columns(df, name))

classification_df = pd.DataFrame(classification)
classification_df.to_csv(os.path.join(BASE_DIR, "variable_classification.csv"), index=False, encoding="utf-8-sig")

# =========================
# 4. PHÂN TÍCH TRƯỚC TIỀN XỬ LÝ
# =========================
# Chuyển ngày tạm cho biểu đồ trước xử lý
oil_before = oil.copy()
transactions_before = transactions.copy()
oil_before["date"] = pd.to_datetime(oil_before["date"], errors="coerce")
transactions_before["date"] = pd.to_datetime(transactions_before["date"], errors="coerce")

before_merged = transactions_before.merge(stores, on="store_nbr", how="left")
before_merged = before_merged.merge(oil_before, on="date", how="left")

histogram(before_merged, "transactions", os.path.join(BEFORE_DIR, "histogram_transactions_before.png"), "Histogram giao dịch - trước xử lý")
boxplot(before_merged, "transactions", os.path.join(BEFORE_DIR, "boxplot_transactions_before.png"), "Boxplot giao dịch - trước xử lý")
scatterplot(before_merged, "dcoilwtico", "transactions", os.path.join(BEFORE_DIR, "scatter_oil_transactions_before.png"), "Giá dầu và giao dịch - trước xử lý")

barchart(
    before_merged.groupby("city")["transactions"].sum().sort_values(ascending=False).head(15),
    os.path.join(BEFORE_DIR, "bar_transactions_by_city_before.png"),
    "Top 15 thành phố theo tổng giao dịch - trước xử lý",
    "city",
    "transactions"
)

daily_trans_before = before_merged.groupby("date", as_index=False)["transactions"].sum()
linechart(daily_trans_before, "date", "transactions", os.path.join(BEFORE_DIR, "line_daily_transactions_before.png"), "Tổng giao dịch theo ngày - trước xử lý")

heatmap_corr(
    before_merged,
    ["store_nbr", "cluster", "transactions", "dcoilwtico"],
    os.path.join(BEFORE_DIR, "heatmap_corr_before.png"),
    "Heatmap tương quan - trước xử lý"
)

# =========================
# 5. TIỀN XỬ LÝ
# =========================
stores_clean = stores.copy()
oil_clean = oil.copy()
holidays_clean = holidays.copy()
transactions_clean = transactions.copy()
test_clean = test.copy()

# Chuyển kiểu ngày
for df in [oil_clean, holidays_clean, transactions_clean, test_clean]:
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

# Xử lý missing giá dầu: nội suy + lấp trước/lấp sau
oil_clean = oil_clean.sort_values("date")
oil_clean["dcoilwtico_before_missing"] = oil_clean["dcoilwtico"].isna().astype(int)
oil_clean["dcoilwtico"] = oil_clean["dcoilwtico"].interpolate(method="linear")
oil_clean["dcoilwtico"] = oil_clean["dcoilwtico"].ffill().bfill()

# Chuẩn hóa text
for col in ["city", "state", "type"]:
    stores_clean[col] = stores_clean[col].astype(str).str.strip()

for col in ["type", "locale", "locale_name", "description"]:
    holidays_clean[col] = holidays_clean[col].astype(str).str.strip()

test_clean["family"] = test_clean["family"].astype(str).str.strip()

# Gom holiday theo ngày để tránh nhân bản dòng khi merge
holiday_daily = holidays_clean.groupby("date", as_index=False).agg(
    holiday_count=("description", "count"),
    transferred_count=("transferred", "sum")
)

# Merge dữ liệu chính
clean = transactions_clean.merge(stores_clean, on="store_nbr", how="left")
clean = clean.merge(oil_clean[["date", "dcoilwtico", "dcoilwtico_before_missing"]], on="date", how="left")
clean = clean.merge(holiday_daily, on="date", how="left")
clean["holiday_count"] = clean["holiday_count"].fillna(0).astype(int)
clean["transferred_count"] = clean["transferred_count"].fillna(0).astype(int)

# Sau merge, nếu giá dầu còn thiếu do ngày giao dịch không có trong oil thì lấp theo thời gian
clean = clean.sort_values("date")
clean["dcoilwtico"] = clean["dcoilwtico"].ffill().bfill()

# Feature thời gian
clean["year"] = clean["date"].dt.year
clean["month"] = clean["date"].dt.month
clean["dayofweek"] = clean["date"].dt.dayofweek
clean["is_weekend"] = clean["dayofweek"].isin([5, 6]).astype(int)

clean.to_csv(os.path.join(BASE_DIR, "cleaned_transactions_dataset.csv"), index=False, encoding="utf-8-sig")

# Summary trước/sau
summary = pd.DataFrame({
    "metric": [
        "Số dòng transactions",
        "Số cột trước merge",
        "Missing oil gốc",
        "Missing dcoilwtico sau merge + xử lý",
        "Missing holiday_count sau xử lý",
        "Số dòng dữ liệu sạch",
        "Số cột dữ liệu sạch"
    ],
    "before": [
        len(transactions),
        transactions.shape[1],
        int(oil["dcoilwtico"].isna().sum()),
        int(before_merged["dcoilwtico"].isna().sum()),
        "Chưa có biến",
        len(before_merged),
        before_merged.shape[1]
    ],
    "after": [
        len(transactions_clean),
        transactions_clean.shape[1],
        int(oil_clean["dcoilwtico"].isna().sum()),
        int(clean["dcoilwtico"].isna().sum()),
        int(clean["holiday_count"].isna().sum()),
        len(clean),
        clean.shape[1]
    ]
})
summary.to_csv(os.path.join(BASE_DIR, "preprocessing_comparison.csv"), index=False, encoding="utf-8-sig")

# =========================
# 6. PHÂN TÍCH SAU TIỀN XỬ LÝ
# =========================
histogram(clean, "transactions", os.path.join(AFTER_DIR, "histogram_transactions_after.png"), "Histogram giao dịch - sau xử lý")
histogram(clean, "dcoilwtico", os.path.join(AFTER_DIR, "histogram_oil_after.png"), "Histogram giá dầu - sau xử lý")

boxplot(clean, "transactions", os.path.join(AFTER_DIR, "boxplot_transactions_after.png"), "Boxplot giao dịch - sau xử lý")
boxplot(clean, "dcoilwtico", os.path.join(AFTER_DIR, "boxplot_oil_after.png"), "Boxplot giá dầu - sau xử lý")

scatterplot(clean, "dcoilwtico", "transactions", os.path.join(AFTER_DIR, "scatter_oil_transactions_after.png"), "Giá dầu và giao dịch - sau xử lý")
scatterplot(clean, "cluster", "transactions", os.path.join(AFTER_DIR, "scatter_cluster_transactions_after.png"), "Cluster và giao dịch - sau xử lý")

barchart(
    clean.groupby("city")["transactions"].sum().sort_values(ascending=False).head(15),
    os.path.join(AFTER_DIR, "bar_transactions_by_city_after.png"),
    "Top 15 thành phố theo tổng giao dịch - sau xử lý",
    "city",
    "transactions"
)

barchart(
    clean.groupby("type")["transactions"].sum().sort_values(ascending=False),
    os.path.join(AFTER_DIR, "bar_transactions_by_store_type_after.png"),
    "Tổng giao dịch theo loại cửa hàng - sau xử lý",
    "store type",
    "transactions"
)

daily_trans_after = clean.groupby("date", as_index=False)["transactions"].sum()
linechart(daily_trans_after, "date", "transactions", os.path.join(AFTER_DIR, "line_daily_transactions_after.png"), "Tổng giao dịch theo ngày - sau xử lý")

daily_oil_after = clean.groupby("date", as_index=False)["dcoilwtico"].mean()
linechart(daily_oil_after, "date", "dcoilwtico", os.path.join(AFTER_DIR, "line_oil_after.png"), "Giá dầu trung bình theo ngày - sau xử lý")

heatmap_corr(
    clean,
    ["store_nbr", "cluster", "transactions", "dcoilwtico", "holiday_count", "transferred_count", "month", "dayofweek", "is_weekend"],
    os.path.join(AFTER_DIR, "heatmap_corr_after.png"),
    "Heatmap tương quan - sau xử lý"
)

# Biểu đồ riêng cho test.csv
test_clean["date"] = pd.to_datetime(test_clean["date"], errors="coerce")
barchart(
    test_clean.groupby("family")["onpromotion"].sum().sort_values(ascending=False).head(15),
    os.path.join(AFTER_DIR, "bar_onpromotion_by_family_test.png"),
    "Top 15 nhóm hàng theo số lượng khuyến mãi trong test.csv",
    "family",
    "onpromotion"
)

daily_promo = test_clean.groupby("date", as_index=False)["onpromotion"].sum()
linechart(daily_promo, "date", "onpromotion", os.path.join(AFTER_DIR, "line_onpromotion_test.png"), "Tổng khuyến mãi theo ngày trong test.csv")

print("DONE")
print("Output folder:", BASE_DIR)
print("Variable classification:", os.path.join(BASE_DIR, "variable_classification.csv"))
print("Cleaned dataset:", os.path.join(BASE_DIR, "cleaned_transactions_dataset.csv"))
print("Preprocessing comparison:", os.path.join(BASE_DIR, "preprocessing_comparison.csv"))
print("Plots folder:", PLOT_DIR)
