import os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

input_file = Path("online_shoppers_intention.csv")
output_dir = Path("online_shoppers_outputs")
charts_dir = output_dir / "charts"
before_dir = charts_dir / "before_preprocessing"
after_dir = charts_dir / "after_preprocessing"
reports_dir = output_dir / "reports"

for p in [before_dir, after_dir, reports_dir]:
    p.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(input_file)

target_col = "Revenue"
numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
bool_cols = df.select_dtypes(include=["bool"]).columns.tolist()
categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()

coded_categorical = ["OperatingSystems", "Browser", "Region", "TrafficType"]
numeric_analysis_cols = [c for c in numeric_cols if c not in coded_categorical]
categorical_analysis_cols = categorical_cols + bool_cols + coded_categorical

classification = []
for col in df.columns:
    if col == target_col:
        nature = "Biến mục tiêu / định tính nhị phân"
    elif col in numeric_analysis_cols:
        nature = "Biến định lượng"
    elif col in categorical_analysis_cols:
        nature = "Biến định tính"
    else:
        nature = "Khác"
    classification.append({
        "ten_bien": col,
        "kieu_du_lieu_python": str(df[col].dtype),
        "ban_chat_bien": nature,
        "so_gia_tri_khuyet": int(df[col].isna().sum()),
        "so_gia_tri_khac_nhau": int(df[col].nunique(dropna=True))
    })

pd.DataFrame(classification).to_csv(reports_dir / "01_phan_loai_bien.csv", index=False, encoding="utf-8-sig")

def basic_report(data, name):
    report = {
        "ten_bo_du_lieu": name,
        "so_dong": data.shape[0],
        "so_cot": data.shape[1],
        "so_du_lieu_khuyet": int(data.isna().sum().sum()),
        "so_dong_trung_lap": int(data.duplicated().sum())
    }
    pd.DataFrame([report]).to_csv(reports_dir / f"summary_{name}.csv", index=False, encoding="utf-8-sig")

basic_report(df, "before")
df[numeric_analysis_cols].describe().T.to_csv(reports_dir / "02_mo_ta_bien_dinh_luong_before.csv", encoding="utf-8-sig")

df_clean = df.copy()
df_clean = df_clean.drop_duplicates()

for col in numeric_cols:
    if df_clean[col].isna().sum() > 0:
        df_clean[col] = df_clean[col].fillna(df_clean[col].median())

for col in categorical_cols + bool_cols:
    if df_clean[col].isna().sum() > 0:
        mode_value = df_clean[col].mode(dropna=True)
        fill_value = mode_value.iloc[0] if len(mode_value) else "Unknown"
        df_clean[col] = df_clean[col].fillna(fill_value)

outlier_report_rows = []
for col in numeric_analysis_cols:
    q1 = df_clean[col].quantile(0.25)
    q3 = df_clean[col].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    before_outliers = int(((df_clean[col] < lower) | (df_clean[col] > upper)).sum())
    df_clean[col] = df_clean[col].clip(lower=lower, upper=upper)
    after_outliers = int(((df_clean[col] < lower) | (df_clean[col] > upper)).sum())
    outlier_report_rows.append({
        "bien": col,
        "q1": q1,
        "q3": q3,
        "iqr": iqr,
        "can_duoi": lower,
        "can_tren": upper,
        "so_ngoai_le_truoc_clip": before_outliers,
        "so_ngoai_le_sau_clip": after_outliers
    })

pd.DataFrame(outlier_report_rows).to_csv(reports_dir / "03_outlier_iqr_report.csv", index=False, encoding="utf-8-sig")

basic_report(df_clean, "after")
df_clean[numeric_analysis_cols].describe().T.to_csv(reports_dir / "04_mo_ta_bien_dinh_luong_after.csv", encoding="utf-8-sig")

compare_rows = []
for col in numeric_analysis_cols:
    compare_rows.append({
        "bien": col,
        "mean_before": df[col].mean(),
        "mean_after": df_clean[col].mean(),
        "median_before": df[col].median(),
        "median_after": df_clean[col].median(),
        "std_before": df[col].std(),
        "std_after": df_clean[col].std(),
        "min_before": df[col].min(),
        "min_after": df_clean[col].min(),
        "max_before": df[col].max(),
        "max_after": df_clean[col].max()
    })
pd.DataFrame(compare_rows).to_csv(reports_dir / "05_so_sanh_truoc_sau_tien_xu_ly.csv", index=False, encoding="utf-8-sig")
df_clean.to_csv(output_dir / "online_shoppers_intention_cleaned.csv", index=False, encoding="utf-8-sig")

def safe_name(s):
    return "".join(ch if ch.isalnum() or ch in ["_", "-"] else "_" for ch in str(s))

def save_histograms(data, folder, label):
    for col in numeric_analysis_cols:
        plt.figure(figsize=(8, 5))
        plt.hist(data[col].dropna(), bins=30)
        plt.title(f"Histogram - {col} ({label})")
        plt.xlabel(col)
        plt.ylabel("Tần suất")
        plt.tight_layout()
        plt.savefig(folder / f"histogram_{safe_name(col)}_{label}.png", dpi=160)
        plt.close()

def save_boxplots(data, folder, label):
    for col in numeric_analysis_cols:
        plt.figure(figsize=(7, 5))
        plt.boxplot(data[col].dropna(), vert=True)
        plt.title(f"Boxplot - {col} ({label})")
        plt.ylabel(col)
        plt.tight_layout()
        plt.savefig(folder / f"boxplot_{safe_name(col)}_{label}.png", dpi=160)
        plt.close()

def save_bar_charts(data, folder, label):
    for col in categorical_analysis_cols:
        counts = data[col].astype(str).value_counts().head(20)
        plt.figure(figsize=(10, 5))
        counts.plot(kind="bar")
        plt.title(f"Bar Chart - {col} ({label})")
        plt.xlabel(col)
        plt.ylabel("Số lượng")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(folder / f"bar_{safe_name(col)}_{label}.png", dpi=160)
        plt.close()

def save_scatter_plots(data, folder, label):
    pairs = [
        ("ProductRelated", "ProductRelated_Duration"),
        ("Administrative", "Administrative_Duration"),
        ("Informational", "Informational_Duration"),
        ("BounceRates", "ExitRates"),
        ("PageValues", "ProductRelated_Duration"),
        ("PageValues", "ExitRates")
    ]
    for x, y in pairs:
        if x in data.columns and y in data.columns:
            plt.figure(figsize=(8, 5))
            plt.scatter(data[x], data[y], alpha=0.35)
            plt.title(f"Scatter Plot - {x} vs {y} ({label})")
            plt.xlabel(x)
            plt.ylabel(y)
            plt.tight_layout()
            plt.savefig(folder / f"scatter_{safe_name(x)}_vs_{safe_name(y)}_{label}.png", dpi=160)
            plt.close()

def save_line_chart(data, folder, label):
    month_order = ["Jan", "Feb", "Mar", "Apr", "May", "June", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    if "Month" in data.columns and "Revenue" in data.columns:
        tmp = data.copy()
        tmp["Revenue_num"] = tmp["Revenue"].astype(int)
        monthly = tmp.groupby("Month")["Revenue_num"].mean().reindex(month_order).dropna()
        plt.figure(figsize=(9, 5))
        monthly.plot(kind="line", marker="o")
        plt.title(f"Line Chart - Tỷ lệ Revenue theo tháng ({label})")
        plt.xlabel("Tháng")
        plt.ylabel("Tỷ lệ mua hàng")
        plt.tight_layout()
        plt.savefig(folder / f"line_revenue_rate_by_month_{label}.png", dpi=160)
        plt.close()

def save_heatmap(data, folder, label):
    corr = data[numeric_analysis_cols].corr()
    plt.figure(figsize=(11, 8))
    plt.imshow(corr, aspect="auto")
    plt.colorbar()
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
    plt.yticks(range(len(corr.index)), corr.index)
    plt.title(f"Heatmap - Tương quan biến định lượng ({label})")
    plt.tight_layout()
    plt.savefig(folder / f"heatmap_correlation_{label}.png", dpi=180)
    plt.close()

def save_all_charts(data, folder, label):
    save_histograms(data, folder, label)
    save_boxplots(data, folder, label)
    save_bar_charts(data, folder, label)
    save_scatter_plots(data, folder, label)
    save_line_chart(data, folder, label)
    save_heatmap(data, folder, label)

save_all_charts(df, before_dir, "before")
save_all_charts(df_clean, after_dir, "after")

print("Hoàn tất. Kết quả nằm trong thư mục:", output_dir)
