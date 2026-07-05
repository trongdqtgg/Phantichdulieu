from pathlib import Path
import re, zipfile, textwrap
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

possible_inputs = [Path('Womens Clothing E-Commerce Reviews.csv'), Path('Womens Clothing E-Commerce Reviews(2).csv')]
INPUT = next((p for p in possible_inputs if p.exists()), None)
if INPUT is None:
    raise FileNotFoundError('Không tìm thấy file CSV. Hãy đặt Womens Clothing E-Commerce Reviews.csv cùng thư mục với script.')
BASE = Path('womens_ecommerce_outputs')
CHARTS = BASE/'charts'
BEFORE = CHARTS/'before_preprocessing'
AFTER = CHARTS/'after_preprocessing'
REPORTS = BASE/'reports'
for p in [BEFORE, AFTER, REPORTS]:
    p.mkdir(parents=True, exist_ok=True)

def safe(s):
    s = str(s).strip().replace(' ', '_').replace('/', '_')
    s = re.sub(r'[^A-Za-z0-9_\-]+', '', s)
    return s[:90]

def savefig(path):
    plt.tight_layout()
    plt.savefig(path, dpi=160, bbox_inches='tight')
    plt.close()

def add_features(d):
    out = d.copy()
    # text derived features, keep NaN-aware for before charts but usable with fillna('')
    review_text = out.get('Review Text', pd.Series('', index=out.index)).fillna('').astype(str)
    title = out.get('Title', pd.Series('', index=out.index)).fillna('').astype(str)
    out['Review_Text_Length'] = review_text.str.len()
    out['Review_Word_Count'] = review_text.str.split().apply(len)
    out['Title_Length'] = title.str.len()
    out['Has_Title'] = np.where(title.str.strip().eq(''), 'No title', 'Has title')
    if 'Rating' in out.columns:
        out['Sentiment_Label'] = pd.cut(out['Rating'], bins=[0,2,3,5], labels=['Negative (1-2)','Neutral (3)','Positive (4-5)'], include_lowest=True)
    if 'Age' in out.columns:
        out['Age_Group'] = pd.cut(out['Age'], bins=[0,25,35,45,55,65,120], labels=['<=25','26-35','36-45','46-55','56-65','>65'])
    if 'Recommended IND' in out.columns:
        out['Recommended_Label'] = out['Recommended IND'].map({0:'Not recommended', 1:'Recommended'}).fillna('Unknown')
    return out

raw = pd.read_csv(INPUT)
before = add_features(raw)

# Define variable nature based on the paper/dataset context
id_cols = ['Unnamed: 0', 'Clothing ID']
target_cols = ['Rating', 'Recommended IND', 'Sentiment_Label', 'Recommended_Label']
quant_cols_before = ['Age', 'Positive Feedback Count', 'Review_Text_Length', 'Review_Word_Count', 'Title_Length']
cat_cols_before = ['Title', 'Review Text', 'Division Name', 'Department Name', 'Class Name', 'Has_Title', 'Age_Group']

class_rows = []
for col in before.columns:
    if col in id_cols:
        nature = 'Biến định danh / mã hóa, không dùng để tính trung bình'
        role = 'ID / chỉ số dòng'
    elif col in ['Age','Positive Feedback Count','Review_Text_Length','Review_Word_Count','Title_Length']:
        nature = 'Biến định lượng'
        role = 'Phân tích mô tả, histogram, boxplot, scatter, line'
    elif col in ['Rating']:
        nature = 'Biến định lượng rời rạc kiêm nhãn cảm xúc theo thang 1-5'
        role = 'Mục tiêu phân tích đánh giá / sentiment proxy'
    elif col in ['Recommended IND']:
        nature = 'Biến định tính nhị phân được mã hóa 0/1'
        role = 'Nhãn khuyến nghị sản phẩm'
    elif col in ['Title','Review Text']:
        nature = 'Biến định tính dạng văn bản tự do'
        role = 'Nguồn tạo đặc trưng độ dài và phân tích NLP'
    else:
        nature = 'Biến định tính'
        role = 'Phân nhóm / so sánh tần suất'
    class_rows.append({
        'ten_bien': col,
        'kieu_du_lieu_python': str(before[col].dtype),
        'ban_chat_bien': nature,
        'vai_tro_phan_tich': role,
        'so_gia_tri_khuyet': int(before[col].isna().sum()),
        'ty_le_khuyet_%': round(before[col].isna().mean()*100, 2),
        'so_gia_tri_khac_nhau': int(before[col].nunique(dropna=True))
    })
pd.DataFrame(class_rows).to_csv(REPORTS/'01_phan_loai_ban_chat_bien.csv', index=False, encoding='utf-8-sig')

def summary(data, name):
    pd.DataFrame([{
        'giai_doan': name,
        'so_dong': data.shape[0],
        'so_cot': data.shape[1],
        'tong_gia_tri_khuyet': int(data.isna().sum().sum()),
        'so_dong_trung_lap': int(data.duplicated().sum()),
        'ty_le_recommended_%': round(data['Recommended IND'].mean()*100, 2) if 'Recommended IND' in data else np.nan,
        'rating_trung_binh': round(data['Rating'].mean(), 3) if 'Rating' in data else np.nan,
        'tuoi_trung_binh': round(data['Age'].mean(), 3) if 'Age' in data else np.nan
    }]).to_csv(REPORTS/f'summary_{name}.csv', index=False, encoding='utf-8-sig')

summary(before, 'before')
before[[c for c in quant_cols_before + ['Rating','Recommended IND'] if c in before.columns]].describe().T.to_csv(REPORTS/'02_mo_ta_bien_dinh_luong_before.csv', encoding='utf-8-sig')

# categorical frequency report before
freq_rows=[]
for col in [c for c in cat_cols_before + ['Rating','Recommended_Label','Sentiment_Label'] if c in before.columns]:
    vc = before[col].astype('object').fillna('Missing').value_counts(dropna=False).head(15)
    for val, cnt in vc.items():
        freq_rows.append({'giai_doan':'before','ten_bien':col,'gia_tri':val,'tan_suat':int(cnt),'ty_le_%':round(cnt/len(before)*100,2)})
pd.DataFrame(freq_rows).to_csv(REPORTS/'03_tan_suat_bien_dinh_tinh_before.csv', index=False, encoding='utf-8-sig')

# preprocessing
clean = raw.copy()
if 'Unnamed: 0' in clean.columns:
    clean = clean.drop(columns=['Unnamed: 0'])
clean = clean.drop_duplicates()
# strip text columns
for col in ['Title','Review Text','Division Name','Department Name','Class Name']:
    if col in clean.columns:
        clean[col] = clean[col].astype('object').where(clean[col].notna(), np.nan)
        clean[col] = clean[col].apply(lambda x: x.strip() if isinstance(x,str) else x)
# fill text/category
if 'Title' in clean.columns:
    clean['Title'] = clean['Title'].fillna('No title')
if 'Review Text' in clean.columns:
    clean['Review Text'] = clean['Review Text'].fillna('No review text')
for col in ['Division Name','Department Name','Class Name']:
    if col in clean.columns:
        mode = clean[col].mode(dropna=True)
        clean[col] = clean[col].fillna(mode.iloc[0] if len(mode) else 'Unknown')
# numeric fill
for col in ['Age','Rating','Recommended IND','Positive Feedback Count','Clothing ID']:
    if col in clean.columns and clean[col].isna().sum():
        clean[col] = clean[col].fillna(clean[col].median())
clean = add_features(clean)

# cap outliers with IQR for key continuous/count variables
outlier_rows=[]
cap_cols = ['Age','Positive Feedback Count','Review_Text_Length','Review_Word_Count','Title_Length']
for col in cap_cols:
    if col in clean.columns:
        q1, q3 = clean[col].quantile(0.25), clean[col].quantile(0.75)
        iqr = q3-q1
        lower, upper = q1 - 1.5*iqr, q3 + 1.5*iqr
        before_out = int(((clean[col] < lower) | (clean[col] > upper)).sum())
        clean[col] = clean[col].clip(lower=lower, upper=upper)
        after_out = int(((clean[col] < lower) | (clean[col] > upper)).sum())
        outlier_rows.append({'ten_bien':col,'Q1':round(q1,3),'Q3':round(q3,3),'IQR':round(iqr,3),'can_duoi':round(lower,3),'can_tren':round(upper,3),'so_outlier_truoc_cap':before_out,'so_outlier_sau_cap':after_out})
pd.DataFrame(outlier_rows).to_csv(REPORTS/'04_outlier_iqr_report.csv', index=False, encoding='utf-8-sig')
summary(clean, 'after')
clean.to_csv(BASE/'womens_clothing_reviews_cleaned.csv', index=False, encoding='utf-8-sig')

quant_cols_after = [c for c in ['Age','Rating','Recommended IND','Positive Feedback Count','Review_Text_Length','Review_Word_Count','Title_Length'] if c in clean.columns]
clean[quant_cols_after].describe().T.to_csv(REPORTS/'05_mo_ta_bien_dinh_luong_after.csv', encoding='utf-8-sig')
freq_rows=[]
for col in [c for c in ['Division Name','Department Name','Class Name','Rating','Recommended_Label','Sentiment_Label','Age_Group','Has_Title'] if c in clean.columns]:
    vc = clean[col].astype('object').fillna('Missing').value_counts(dropna=False).head(20)
    for val, cnt in vc.items():
        freq_rows.append({'giai_doan':'after','ten_bien':col,'gia_tri':val,'tan_suat':int(cnt),'ty_le_%':round(cnt/len(clean)*100,2)})
pd.DataFrame(freq_rows).to_csv(REPORTS/'06_tan_suat_bien_dinh_tinh_after.csv', index=False, encoding='utf-8-sig')

compare = pd.DataFrame([
    {'chi_tieu':'Số dòng','before':len(before),'after':len(clean),'nhan_xet':'Sau tiền xử lý loại trùng lặp nếu có.'},
    {'chi_tieu':'Số cột','before':before.shape[1],'after':clean.shape[1],'nhan_xet':'Sau tiền xử lý bỏ cột chỉ số và thêm biến đặc trưng văn bản.'},
    {'chi_tieu':'Tổng giá trị khuyết','before':int(before.isna().sum().sum()),'after':int(clean.isna().sum().sum()),'nhan_xet':'Đã điền thiếu cho tiêu đề, nội dung đánh giá và nhóm sản phẩm.'},
    {'chi_tieu':'Tuổi trung bình','before':round(before['Age'].mean(),3),'after':round(clean['Age'].mean(),3),'nhan_xet':'Thay đổi nhẹ vì winsorize/capping outlier.'},
    {'chi_tieu':'Rating trung bình','before':round(before['Rating'].mean(),3),'after':round(clean['Rating'].mean(),3),'nhan_xet':'Rating không bị thay đổi về bản chất.'},
    {'chi_tieu':'Positive Feedback Count trung bình','before':round(before['Positive Feedback Count'].mean(),3),'after':round(clean['Positive Feedback Count'].mean(),3),'nhan_xet':'Giảm ảnh hưởng của giá trị cực đoan nhờ capping IQR.'},
    {'chi_tieu':'Tỷ lệ Recommended (%)','before':round(before['Recommended IND'].mean()*100,2),'after':round(clean['Recommended IND'].mean()*100,2),'nhan_xet':'Tỷ lệ khuyến nghị được giữ nguyên gần như hoàn toàn.'},
])
compare.to_csv(REPORTS/'07_so_sanh_truoc_sau_tien_xu_ly.csv', index=False, encoding='utf-8-sig')

# plot functions

def plot_hist(data, cols, outdir, suffix):
    for col in cols:
        if col in data.columns:
            s = pd.to_numeric(data[col], errors='coerce').dropna()
            if len(s)==0: continue
            plt.figure(figsize=(8,5))
            plt.hist(s, bins=30, edgecolor='black')
            plt.title(f'Histogram - {col} ({suffix})')
            plt.xlabel(col); plt.ylabel('Tần suất')
            savefig(outdir/f'histogram_{safe(col)}_{suffix}.png')

def plot_box(data, cols, outdir, suffix):
    for col in cols:
        if col in data.columns:
            s = pd.to_numeric(data[col], errors='coerce').dropna()
            if len(s)==0: continue
            plt.figure(figsize=(7,5))
            plt.boxplot(s, vert=True, patch_artist=True)
            plt.title(f'Boxplot - {col} ({suffix})')
            plt.ylabel(col)
            savefig(outdir/f'boxplot_{safe(col)}_{suffix}.png')

def plot_bar(data, cols, outdir, suffix, topn=20):
    for col in cols:
        if col in data.columns:
            vc = data[col].astype('object').fillna('Missing').value_counts().head(topn).sort_values()
            if len(vc)==0: continue
            plt.figure(figsize=(9, max(4, len(vc)*0.35)))
            plt.barh(vc.index.astype(str), vc.values)
            plt.title(f'Bar Chart - {col} ({suffix})')
            plt.xlabel('Số lượng'); plt.ylabel(col)
            savefig(outdir/f'bar_{safe(col)}_{suffix}.png')

def plot_scatter(data, pairs, outdir, suffix):
    for x,y in pairs:
        if x in data.columns and y in data.columns:
            d = data[[x,y]].apply(pd.to_numeric, errors='coerce').dropna()
            if len(d)==0: continue
            sample = d.sample(min(len(d), 6000), random_state=42) if len(d)>6000 else d
            plt.figure(figsize=(8,5))
            plt.scatter(sample[x], sample[y], alpha=0.35, s=12)
            plt.title(f'Scatter Plot - {x} vs {y} ({suffix})')
            plt.xlabel(x); plt.ylabel(y)
            savefig(outdir/f'scatter_{safe(x)}_vs_{safe(y)}_{suffix}.png')

def plot_heatmap(data, cols, outdir, suffix):
    cols = [c for c in cols if c in data.columns]
    corr = data[cols].apply(pd.to_numeric, errors='coerce').corr()
    if corr.empty: return
    plt.figure(figsize=(9,7))
    im = plt.imshow(corr, aspect='auto')
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=45, ha='right')
    plt.yticks(range(len(corr.index)), corr.index)
    for i in range(len(corr.index)):
        for j in range(len(corr.columns)):
            val = corr.iloc[i,j]
            if pd.notna(val):
                plt.text(j, i, f'{val:.2f}', ha='center', va='center', fontsize=8)
    plt.title(f'Heatmap tương quan ({suffix})')
    savefig(outdir/f'heatmap_correlation_{suffix}.png')

def plot_line_charts(data, outdir, suffix):
    # count by age
    if 'Age' in data.columns:
        g = data.groupby('Age').size().sort_index()
        plt.figure(figsize=(9,5))
        plt.plot(g.index, g.values, marker='o', linewidth=1)
        plt.title(f'Line Chart - Số review theo tuổi ({suffix})')
        plt.xlabel('Age'); plt.ylabel('Số review')
        savefig(outdir/f'line_review_count_by_age_{suffix}.png')
    if {'Age','Rating'}.issubset(data.columns):
        g = data.groupby('Age')['Rating'].mean().sort_index()
        plt.figure(figsize=(9,5))
        plt.plot(g.index, g.values, marker='o', linewidth=1)
        plt.title(f'Line Chart - Rating trung bình theo tuổi ({suffix})')
        plt.xlabel('Age'); plt.ylabel('Rating trung bình')
        savefig(outdir/f'line_avg_rating_by_age_{suffix}.png')
    if {'Rating','Recommended IND'}.issubset(data.columns):
        g = data.groupby('Rating')['Recommended IND'].mean().sort_index()*100
        plt.figure(figsize=(8,5))
        plt.plot(g.index, g.values, marker='o')
        plt.title(f'Line Chart - Tỷ lệ recommended theo rating ({suffix})')
        plt.xlabel('Rating'); plt.ylabel('Recommended (%)')
        savefig(outdir/f'line_recommended_rate_by_rating_{suffix}.png')
    if {'Rating','Positive Feedback Count'}.issubset(data.columns):
        g = data.groupby('Rating')['Positive Feedback Count'].mean().sort_index()
        plt.figure(figsize=(8,5))
        plt.plot(g.index, g.values, marker='o')
        plt.title(f'Line Chart - Feedback tích cực TB theo rating ({suffix})')
        plt.xlabel('Rating'); plt.ylabel('Positive Feedback Count TB')
        savefig(outdir/f'line_avg_feedback_by_rating_{suffix}.png')
    if {'Age_Group','Recommended IND'}.issubset(data.columns):
        g = data.groupby('Age_Group', observed=False)['Recommended IND'].mean()*100
        plt.figure(figsize=(8,5))
        plt.plot(g.index.astype(str), g.values, marker='o')
        plt.title(f'Line Chart - Tỷ lệ recommended theo nhóm tuổi ({suffix})')
        plt.xlabel('Nhóm tuổi'); plt.ylabel('Recommended (%)')
        savefig(outdir/f'line_recommended_rate_by_age_group_{suffix}.png')

# chart lists
hist_cols = ['Age','Rating','Recommended IND','Positive Feedback Count','Review_Text_Length','Review_Word_Count','Title_Length']
box_cols = hist_cols
bar_cols = ['Rating','Recommended IND','Recommended_Label','Sentiment_Label','Division Name','Department Name','Class Name','Age_Group','Has_Title']
scatter_pairs = [
    ('Age','Rating'), ('Age','Positive Feedback Count'), ('Age','Review_Word_Count'),
    ('Rating','Positive Feedback Count'), ('Rating','Review_Word_Count'),
    ('Review_Text_Length','Positive Feedback Count'), ('Review_Word_Count','Positive Feedback Count'),
    ('Review_Text_Length','Rating'), ('Review_Word_Count','Rating'),
    ('Title_Length','Rating')
]

for data,outdir,suffix in [(before,BEFORE,'before'),(clean,AFTER,'after')]:
    plot_hist(data, hist_cols, outdir, suffix)
    plot_box(data, box_cols, outdir, suffix)
    plot_bar(data, bar_cols, outdir, suffix)
    plot_scatter(data, scatter_pairs, outdir, suffix)
    plot_heatmap(data, hist_cols, outdir, suffix)
    plot_line_charts(data, outdir, suffix)

# extra grouped bar charts after only
if {'Department Name','Rating'}.issubset(clean.columns):
    pivot = clean.pivot_table(index='Department Name', columns='Rating', values='Clothing ID', aggfunc='count', fill_value=0)
    pivot = pivot.loc[pivot.sum(axis=1).sort_values(ascending=False).head(10).index]
    pivot.plot(kind='bar', figsize=(10,6))
    plt.title('Grouped Bar - Rating theo Department Name (after)')
    plt.xlabel('Department Name'); plt.ylabel('Số review'); plt.xticks(rotation=45, ha='right')
    savefig(AFTER/'grouped_bar_rating_by_department_after.png')

if {'Class Name','Recommended IND'}.issubset(clean.columns):
    g = clean.groupby('Class Name')['Recommended IND'].mean().sort_values(ascending=False).head(15)*100
    plt.figure(figsize=(9,6))
    plt.barh(g.sort_values().index.astype(str), g.sort_values().values)
    plt.title('Bar Chart - Top Class Name theo tỷ lệ Recommended (after)')
    plt.xlabel('Recommended (%)'); plt.ylabel('Class Name')
    savefig(AFTER/'bar_top_class_by_recommended_rate_after.png')

# Markdown/text reports
n = len(raw); n_after=len(clean)
missing = raw.isna().sum().sort_values(ascending=False)
missing_txt = '\n'.join([f'- {idx}: {val} giá trị thiếu' for idx,val in missing.items() if val>0]) or '- Không có giá trị thiếu.'
chart_count_before = len(list(BEFORE.glob('*.png')))
chart_count_after = len(list(AFTER.glob('*.png')))

report_md = f'''# Báo cáo phân tích dữ liệu Women's E-Commerce Clothing Reviews

## Thông tin dữ liệu và bài báo tham chiếu
- Dataset: Women's E-Commerce Clothing Reviews trên Kaggle.
- Bài báo tham chiếu: Statistical Analysis on E-Commerce Reviews, with Sentiment Classification using Bidirectional RNN (Agarap, A. F., 2018).
- Link arXiv: https://arxiv.org/abs/1805.03687
- Số dòng ban đầu: {raw.shape[0]:,}
- Số cột ban đầu: {raw.shape[1]:,}

## 1. Phân loại bản chất của biến
Chi tiết nằm trong `reports/01_phan_loai_ban_chat_bien.csv`.

Tóm tắt:
- Biến định danh: `Unnamed: 0`, `Clothing ID`.
- Biến định lượng: `Age`, `Rating`, `Positive Feedback Count`, các biến tạo thêm như `Review_Text_Length`, `Review_Word_Count`, `Title_Length`.
- Biến định tính: `Title`, `Review Text`, `Division Name`, `Department Name`, `Class Name`, `Age_Group`, `Sentiment_Label`, `Recommended_Label`.
- Biến mục tiêu/nhãn quan trọng: `Rating`, `Recommended IND`, `Sentiment_Label`.

## 2. Phân tích các biến định lượng
Các biến định lượng được mô tả bằng count, mean, std, min, Q1, median, Q3, max trong:
- `reports/02_mo_ta_bien_dinh_luong_before.csv`
- `reports/05_mo_ta_bien_dinh_luong_after.csv`

Nhận xét chính:
- `Rating` có xu hướng lệch về mức cao, cho thấy phần lớn khách hàng đánh giá tích cực.
- `Positive Feedback Count` lệch phải mạnh, đa số review ít phản hồi tích cực nhưng có một số review nhận nhiều phản hồi.
- `Age` tập trung nhiều ở nhóm người trưởng thành, phù hợp với dữ liệu mua sắm quần áo nữ.
- Các biến độ dài văn bản giúp bổ sung góc nhìn NLP: review dài có thể phản ánh trải nghiệm chi tiết hơn.

## 3. Phân tích các biến định tính
Báo cáo tần suất nằm trong:
- `reports/03_tan_suat_bien_dinh_tinh_before.csv`
- `reports/06_tan_suat_bien_dinh_tinh_after.csv`

Nhận xét chính:
- `Class Name`, `Department Name`, `Division Name` cho biết nhóm sản phẩm nào có nhiều review.
- `Recommended IND` cho biết khách hàng có khuyến nghị sản phẩm hay không.
- `Sentiment_Label` được suy ra từ `Rating`: 1-2 tiêu cực, 3 trung lập, 4-5 tích cực.
- Các cột văn bản `Title` và `Review Text` có giá trị thiếu đáng kể nên cần xử lý trước khi phân tích văn bản.

## 4. Trực quan hóa dữ liệu
Đã xuất ảnh PNG theo 2 giai đoạn:
- Trước tiền xử lý: `charts/before_preprocessing/` ({chart_count_before} biểu đồ)
- Sau tiền xử lý: `charts/after_preprocessing/` ({chart_count_after} biểu đồ)

Đủ các loại biểu đồ yêu cầu:
- Histogram: phân phối Age, Rating, Positive Feedback Count, Review length, Word count, Title length.
- Boxplot: phát hiện ngoại lệ ở Age, Positive Feedback Count, độ dài review.
- Scatter Plot: quan hệ giữa Age, Rating, Positive Feedback Count, Review length, Word count.
- Bar Chart: tần suất Rating, Recommended, Division, Department, Class, Sentiment, Age Group.
- Line Chart: số review theo tuổi, rating trung bình theo tuổi, recommended rate theo rating/nhóm tuổi.
- Heatmap: tương quan giữa các biến định lượng.

## 5. Python và Excel
- Code Python hoàn chỉnh: `phan_tich_womens_ecommerce.py`.
- Hướng dẫn làm bằng Excel: `reports/09_huong_dan_excel_step_by_step.txt`.
- Ưu/khuyết điểm Python và Excel: `reports/10_uu_nhuoc_diem_python_excel.txt`.

## 6. So sánh trước và sau tiền xử lý
Chi tiết: `reports/07_so_sanh_truoc_sau_tien_xu_ly.csv`.

Giá trị thiếu ban đầu:
{missing_txt}

Các bước tiền xử lý đã thực hiện:
1. Xóa cột chỉ số `Unnamed: 0` vì không mang ý nghĩa phân tích.
2. Xóa dòng trùng lặp nếu có.
3. Điền thiếu `Title` bằng `No title`, `Review Text` bằng `No review text`.
4. Điền thiếu nhóm sản phẩm bằng mode.
5. Tạo biến mới: `Review_Text_Length`, `Review_Word_Count`, `Title_Length`, `Has_Title`, `Sentiment_Label`, `Age_Group`, `Recommended_Label`.
6. Capping outlier bằng IQR cho Age, Positive Feedback Count và các biến độ dài văn bản.

## 7. Nhận xét tổng hợp
Dataset phù hợp để phân tích thống kê mô tả, trực quan hóa hành vi đánh giá sản phẩm và làm bài toán phân loại cảm xúc/khuyến nghị. Sau tiền xử lý, dữ liệu sạch hơn, giảm ảnh hưởng của missing value và outlier, đồng thời có thêm biến đặc trưng văn bản để phục vụ mô hình sentiment classification như định hướng của bài báo Agarap (2018).
'''
(REPORTS/'08_bao_cao_phan_tich_day_du.md').write_text(report_md, encoding='utf-8')

excel_steps = '''HƯỚNG DẪN LÀM BẰNG EXCEL - WOMEN'S E-COMMERCE CLOTHING REVIEWS

1. Import dữ liệu
- Mở Excel > Data > Get Data > From Text/CSV.
- Chọn file Womens Clothing E-Commerce Reviews.csv.
- Chọn encoding UTF-8 nếu có lỗi font.
- Bấm Load để đưa dữ liệu vào sheet.

2. Kiểm tra dữ liệu thiếu
- Dùng Filter ở hàng tiêu đề.
- Với từng cột Title, Review Text, Division Name, Department Name, Class Name, lọc Blank để xem số ô trống.
- Có thể dùng công thức =COUNTBLANK(A:A) cho từng cột.

3. Phân loại biến
- Tạo sheet Variable_Type.
- Liệt kê tên cột, kiểu dữ liệu, bản chất biến.
- Age, Rating, Positive Feedback Count là biến định lượng.
- Title, Review Text, Division Name, Department Name, Class Name là biến định tính.
- Recommended IND là biến nhị phân 0/1.

4. Tạo biến mới
- Review_Text_Length: =LEN([@[Review Text]])
- Review_Word_Count: =IF(TRIM([@[Review Text]])="",0,LEN(TRIM([@[Review Text]]))-LEN(SUBSTITUTE(TRIM([@[Review Text]])," ",""))+1)
- Has_Title: =IF([@Title]="","No title","Has title")
- Sentiment_Label: =IF([@Rating]<=2,"Negative",IF([@Rating]=3,"Neutral","Positive"))
- Age_Group: dùng hàm IFS, ví dụ =IFS([@Age]<=25,"<=25",[@Age]<=35,"26-35",[@Age]<=45,"36-45",[@Age]<=55,"46-55",[@Age]<=65,"56-65",TRUE,">65")

5. Phân tích biến định lượng
- Chọn cột Age/Rating/Positive Feedback Count.
- Vào Data > Data Analysis > Descriptive Statistics.
- Nếu chưa có Data Analysis: File > Options > Add-ins > Analysis ToolPak.
- Tính Mean, Median, Min, Max, Standard Deviation.

6. Phân tích biến định tính
- Insert > PivotTable.
- Kéo Department Name/Class Name/Rating vào Rows.
- Kéo cùng biến vào Values để đếm tần suất.
- Đổi Values thành Count nếu Excel đang để Sum.

7. Vẽ Histogram
- Chọn cột Age hoặc Rating.
- Insert > Statistic Chart > Histogram.
- Đặt tiêu đề: Histogram - Age hoặc Histogram - Rating.

8. Vẽ Boxplot
- Chọn cột Age hoặc Positive Feedback Count.
- Insert > Statistic Chart > Box & Whisker.
- Dùng để phát hiện ngoại lệ.

9. Vẽ Scatter Plot
- Chọn 2 cột, ví dụ Age và Positive Feedback Count.
- Insert > Scatter.
- Dùng để xem quan hệ giữa 2 biến định lượng.

10. Vẽ Bar Chart
- Tạo PivotTable đếm Rating, Department Name, Class Name.
- Chọn PivotTable > Insert > Bar/Column Chart.

11. Vẽ Line Chart
- Tạo PivotTable: Rows = Age, Values = Average of Rating hoặc Count of Review.
- Insert > Line Chart.

12. Vẽ Heatmap tương quan
- Tạo bảng correlation bằng Data Analysis > Correlation.
- Chọn các biến Age, Rating, Positive Feedback Count, Review_Text_Length, Review_Word_Count.
- Bôi đen bảng correlation > Home > Conditional Formatting > Color Scales.

13. So sánh trước và sau tiền xử lý
- Sheet Before: dữ liệu gốc.
- Sheet After: dữ liệu đã xử lý thiếu, xóa trùng, thêm biến mới.
- Tạo bảng so sánh: số dòng, số cột, số missing, mean rating, mean age, mean feedback.

14. Xuất ảnh biểu đồ
- Bấm phải vào biểu đồ > Save as Picture.
- Lưu theo tên rõ ràng như histogram_Age_before.png, bar_Rating_after.png.
'''
(REPORTS/'09_huong_dan_excel_step_by_step.txt').write_text(excel_steps, encoding='utf-8')

pros_cons = '''ƯU ĐIỂM - KHUYẾT ĐIỂM PYTHON VÀ EXCEL

1. Python
Ưu điểm:
- Xử lý dữ liệu lớn tốt hơn Excel.
- Có thể tự động hóa toàn bộ quy trình: import, tiền xử lý, thống kê, vẽ biểu đồ, xuất ảnh.
- Dễ tái lập kết quả vì toàn bộ thao tác nằm trong code.
- Phù hợp phân tích NLP/sentiment classification vì có thể tạo biến từ văn bản và huấn luyện mô hình.
- Xuất hàng loạt biểu đồ nhanh, tránh thao tác thủ công.

Khuyết điểm:
- Cần biết lập trình Python và thư viện pandas/matplotlib.
- Người mới có thể khó chỉnh sửa biểu đồ theo giao diện kéo thả.
- Cần kiểm soát lỗi đường dẫn, encoding, tên cột.

2. Excel
Ưu điểm:
- Dễ dùng, trực quan, phù hợp người mới.
- PivotTable giúp phân tích biến định tính rất nhanh.
- Biểu đồ có thể chỉnh sửa bằng chuột, dễ đưa vào báo cáo.
- Phù hợp khi dữ liệu vừa và nhỏ.

Khuyết điểm:
- Khó tái lập nếu thao tác thủ công nhiều bước.
- Dễ sai nếu lọc/xóa/sắp xếp nhầm.
- Không mạnh bằng Python khi xử lý văn bản, dữ liệu lớn hoặc tự động xuất nhiều biểu đồ.
- Vẽ heatmap/correlation và xử lý outlier cần nhiều thao tác hơn.

Kết luận:
- Python phù hợp cho bài phân tích đầy đủ, nhiều biểu đồ, cần lặp lại và mở rộng sang machine learning.
- Excel phù hợp để trình bày nhanh, kiểm tra dữ liệu và tạo PivotTable/bar chart thủ công.
- Nên dùng kết hợp: Python để xử lý và xuất biểu đồ hàng loạt; Excel để kiểm tra, trình bày và tinh chỉnh báo cáo.
'''
(REPORTS/'10_uu_nhuoc_diem_python_excel.txt').write_text(pros_cons, encoding='utf-8')

summary_txt = f'''README - Kết quả phân tích Women's E-Commerce Clothing Reviews

Thư mục gồm:
- phan_tich_womens_ecommerce.py: code Python hoàn chỉnh để chạy lại từ file CSV.
- womens_clothing_reviews_cleaned.csv: dữ liệu sau tiền xử lý.
- reports/*.csv, *.txt, *.md: báo cáo phân loại biến, mô tả định lượng, tần suất định tính, outlier, so sánh trước/sau, hướng dẫn Excel.
- charts/before_preprocessing/*.png: {chart_count_before} ảnh biểu đồ trước tiền xử lý.
- charts/after_preprocessing/*.png: {chart_count_after} ảnh biểu đồ sau tiền xử lý.

Đủ loại biểu đồ bắt buộc:
- Histogram
- Boxplot
- Scatter Plot
- Bar Chart
- Line Chart
- Heatmap

Cách chạy lại:
1. Đặt file Womens Clothing E-Commerce Reviews.csv cùng thư mục với phan_tich_womens_ecommerce.py.
2. Cài thư viện nếu cần: pip install pandas numpy matplotlib
3. Chạy: python phan_tich_womens_ecommerce.py
4. Xem kết quả trong thư mục womens_ecommerce_outputs.
'''
(BASE/'README.txt').write_text(summary_txt, encoding='utf-8')


print(f'Hoàn tất. Kết quả nằm trong thư mục: {BASE.resolve()}')
print(f'Số biểu đồ before: {chart_count_before}; after: {chart_count_after}')
