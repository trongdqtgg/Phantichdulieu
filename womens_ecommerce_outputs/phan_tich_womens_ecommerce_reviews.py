import os, shutil, textwrap, zipfile, json, math, re
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

INPUT = Path('/mnt/data/Womens Clothing E-Commerce Reviews(1).csv')
OUT = Path('/mnt/data/womens_ecommerce_outputs_like_mau')
ZIP = Path('/mnt/data/womens_ecommerce_outputs_like_mau.zip')
if OUT.exists(): shutil.rmtree(OUT)
(OUT/'charts'/'before_preprocessing').mkdir(parents=True)
(OUT/'charts'/'after_preprocessing').mkdir(parents=True)
(OUT/'reports').mkdir(parents=True)

def safe(s):
    return re.sub(r'[^A-Za-z0-9_\-]+','_',str(s)).strip('_')

def savefig(path):
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()

def hist(df, col, folder, suffix):
    s=df[col].dropna()
    if len(s)==0: return
    plt.figure(figsize=(8,5)); plt.hist(s, bins=30)
    plt.title(f'Histogram - {col} ({suffix})'); plt.xlabel(col); plt.ylabel('Frequency')
    savefig(folder/f'histogram_{safe(col)}_{suffix}.png')

def box(df, col, folder, suffix):
    s=df[col].dropna()
    if len(s)==0: return
    plt.figure(figsize=(7,5)); plt.boxplot(s, vert=True, showfliers=True)
    plt.title(f'Boxplot - {col} ({suffix})'); plt.ylabel(col)
    savefig(folder/f'boxplot_{safe(col)}_{suffix}.png')

def bar_counts(df, col, folder, suffix, top=15):
    if col not in df.columns: return
    vc=df[col].astype(str).replace('nan','Missing').value_counts().head(top).sort_values()
    if vc.empty: return
    plt.figure(figsize=(10,6)); plt.barh(vc.index, vc.values)
    plt.title(f'Bar chart - {col} ({suffix})'); plt.xlabel('Count'); plt.ylabel(col)
    savefig(folder/f'bar_{safe(col)}_{suffix}.png')

def scatter(df, x, y, folder, suffix, sample=5000):
    if x not in df.columns or y not in df.columns: return
    dd=df[[x,y]].dropna()
    if dd.empty: return
    if len(dd)>sample: dd=dd.sample(sample, random_state=42)
    plt.figure(figsize=(8,5)); plt.scatter(dd[x], dd[y], alpha=0.35, s=12)
    plt.title(f'Scatter - {x} vs {y} ({suffix})'); plt.xlabel(x); plt.ylabel(y)
    savefig(folder/f'scatter_{safe(x)}_vs_{safe(y)}_{suffix}.png')

def line_group(df, x, y, folder, suffix, title=None):
    if x not in df.columns or y not in df.columns: return
    tmp=df.groupby(x)[y].mean().reset_index().sort_values(x)
    if tmp.empty: return
    plt.figure(figsize=(10,5)); plt.plot(tmp[x].astype(str), tmp[y], marker='o')
    plt.title(title or f'Line chart - avg {y} by {x} ({suffix})'); plt.xlabel(x); plt.ylabel(f'Average {y}')
    plt.xticks(rotation=45)
    savefig(folder/f'line_avg_{safe(y)}_by_{safe(x)}_{suffix}.png')

def heatmap_corr(df, folder, suffix):
    nums=df.select_dtypes(include=np.number)
    if nums.shape[1]<2: return
    corr=nums.corr(numeric_only=True)
    fig, ax=plt.subplots(figsize=(10,8))
    im=ax.imshow(corr.values, aspect='auto')
    ax.set_xticks(range(len(corr.columns))); ax.set_xticklabels(corr.columns, rotation=90)
    ax.set_yticks(range(len(corr.index))); ax.set_yticklabels(corr.index)
    for i in range(len(corr.index)):
        for j in range(len(corr.columns)):
            ax.text(j,i,f'{corr.iloc[i,j]:.2f}',ha='center',va='center',fontsize=7)
    plt.title(f'Heatmap correlation ({suffix})')
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    savefig(folder/f'heatmap_correlation_{suffix}.png')

# Read
df=pd.read_csv(INPUT)
raw=df.copy()

# Add text length features even before preprocessing for richer plots (NaN stays NaN -> length NaN)
for frame in (raw,):
    frame['Review Text Length'] = frame['Review Text'].apply(lambda x: len(str(x)) if pd.notna(x) else np.nan)
    frame['Title Length'] = frame['Title'].apply(lambda x: len(str(x)) if pd.notna(x) else np.nan)
    frame['Sentiment Label'] = frame['Rating'].map(lambda r: 'Positive' if r>=4 else ('Neutral' if r==3 else 'Negative'))

# Classify variables
rows=[]
for col in df.columns:
    dtype=str(df[col].dtype)
    nunique=int(df[col].nunique(dropna=True))
    missing=int(df[col].isna().sum())
    if col in ['Unnamed: 0']:
        nature='Cột chỉ mục dư thừa'; role='Loại bỏ khi tiền xử lý'
    elif col in ['Clothing ID']:
        nature='Định danh sản phẩm, định tính mã hóa số'; role='Biến giải thích / nhóm sản phẩm'
    elif col in ['Age']:
        nature='Định lượng rời rạc'; role='Tuổi khách hàng'
    elif col in ['Rating']:
        nature='Định lượng thứ bậc/rời rạc'; role='Nhãn mức hài lòng 1-5'
    elif col in ['Recommended IND']:
        nature='Định tính nhị phân'; role='Biến mục tiêu khuyến nghị / phân loại'
    elif col in ['Positive Feedback Count']:
        nature='Định lượng rời rạc'; role='Số lượt phản hồi tích cực'
    elif col in ['Title','Review Text']:
        nature='Dữ liệu văn bản phi cấu trúc'; role='Đầu vào NLP / phân tích cảm xúc'
    else:
        nature='Định tính danh nghĩa'; role='Phân nhóm sản phẩm'
    rows.append({'bien':col,'kieu_du_lieu':dtype,'so_gia_tri_phan_biet':nunique,'so_gia_tri_thieu':missing,'ban_chat':nature,'vai_tro_phan_tich':role})
pd.DataFrame(rows).to_csv(OUT/'reports'/'01_phan_loai_bien.csv', index=False, encoding='utf-8-sig')

# Missing report raw
missing=pd.DataFrame({'bien':raw.columns,'missing_before':[raw[c].isna().sum() for c in raw.columns], 'missing_rate_before':[raw[c].isna().mean() for c in raw.columns]})
missing.to_csv(OUT/'reports'/'00_missing_values_before.csv', index=False, encoding='utf-8-sig')

# Reports before
num_before=raw.select_dtypes(include=np.number)
num_before.describe().T.to_csv(OUT/'reports'/'02_mo_ta_bien_dinh_luong_before.csv', encoding='utf-8-sig')

# Outlier IQR before
out=[]
for c in num_before.columns:
    s=num_before[c].dropna();
    if len(s)==0: continue
    q1=s.quantile(.25); q3=s.quantile(.75); iqr=q3-q1; lo=q1-1.5*iqr; hi=q3+1.5*iqr
    out.append({'bien':c,'Q1':q1,'Q3':q3,'IQR':iqr,'lower_bound':lo,'upper_bound':hi,'so_outlier':int(((s<lo)|(s>hi)).sum()),'ty_le_outlier':float(((s<lo)|(s>hi)).mean())})
pd.DataFrame(out).to_csv(OUT/'reports'/'03_outlier_iqr_report_before.csv', index=False, encoding='utf-8-sig')

# Categorical before
cat_cols=[c for c in raw.columns if raw[c].dtype=='object'] + ['Recommended IND','Rating','Clothing ID']
cat_rows=[]
for c in cat_cols:
    if c not in raw.columns: continue
    vc=raw[c].astype(str).replace('nan','Missing').value_counts().head(20)
    for val,count in vc.items():
        cat_rows.append({'bien':c,'gia_tri':val,'tan_suat':int(count),'ty_le':float(count/len(raw))})
pd.DataFrame(cat_rows).to_csv(OUT/'reports'/'06_mo_ta_bien_dinh_tinh_before.csv', index=False, encoding='utf-8-sig')

# Preprocessing
clean=df.copy()
if 'Unnamed: 0' in clean.columns: clean=clean.drop(columns=['Unnamed: 0'])
# Fill missing
clean['Title']=clean['Title'].fillna('No Title')
clean['Review Text']=clean['Review Text'].fillna('No Review')
for c in ['Division Name','Department Name','Class Name']:
    clean[c]=clean[c].fillna('Unknown')
# Strip string columns
for c in clean.select_dtypes(include='object').columns:
    clean[c]=clean[c].astype(str).str.strip()
# Remove duplicate rows if any
before_dup=len(clean)
clean=clean.drop_duplicates()
removed_dup=before_dup-len(clean)
# Feature engineering
clean['Review Text Length']=clean['Review Text'].str.len()
clean['Title Length']=clean['Title'].str.len()
clean['Word Count']=clean['Review Text'].str.split().str.len()
clean['Has Title']=np.where(clean['Title'].eq('No Title'),0,1)
clean['Has Review Text']=np.where(clean['Review Text'].eq('No Review'),0,1)
clean['Recommended Label']=clean['Recommended IND'].map({1:'Recommended',0:'Not Recommended'})
clean['Sentiment Label']=clean['Rating'].map(lambda r: 'Positive' if r>=4 else ('Neutral' if r==3 else 'Negative'))
clean['Age Group']=pd.cut(clean['Age'], bins=[0,25,35,45,55,65,120], labels=['<=25','26-35','36-45','46-55','56-65','>65'], right=True)
# Visual capping version after for skewed numeric variables, keep originals too
for c in ['Age','Positive Feedback Count','Review Text Length','Title Length','Word Count']:
    lo=clean[c].quantile(0.01); hi=clean[c].quantile(0.99)
    clean[c+'_capped_1_99']=clean[c].clip(lo,hi)

# after reports
clean.to_csv(OUT/'womens_clothing_reviews_cleaned.csv', index=False, encoding='utf-8-sig')
num_after=clean.select_dtypes(include=np.number)
num_after.describe().T.to_csv(OUT/'reports'/'04_mo_ta_bien_dinh_luong_after.csv', encoding='utf-8-sig')
# categorical after
cat_cols_after=list(clean.select_dtypes(include=['object','category']).columns)+['Recommended IND','Rating','Clothing ID','Has Title','Has Review Text']
cat_rows=[]
for c in cat_cols_after:
    if c not in clean.columns: continue
    vc=clean[c].astype(str).replace('nan','Missing').value_counts().head(20)
    for val,count in vc.items():
        cat_rows.append({'bien':c,'gia_tri':val,'tan_suat':int(count),'ty_le':float(count/len(clean))})
pd.DataFrame(cat_rows).to_csv(OUT/'reports'/'06_mo_ta_bien_dinh_tinh_after.csv', index=False, encoding='utf-8-sig')
# missing after
missing_after=pd.DataFrame({'bien':clean.columns,'missing_after':[clean[c].isna().sum() for c in clean.columns], 'missing_rate_after':[clean[c].isna().mean() for c in clean.columns]})
missing_after.to_csv(OUT/'reports'/'00_missing_values_after.csv', index=False, encoding='utf-8-sig')

# compare report
compare=[]
for c in sorted(set(list(raw.columns)+list(clean.columns))):
    compare.append({
        'bien':c,
        'co_truoc': c in raw.columns,
        'co_sau': c in clean.columns,
        'missing_truoc': int(raw[c].isna().sum()) if c in raw.columns else '',
        'missing_sau': int(clean[c].isna().sum()) if c in clean.columns else '',
        'so_gia_tri_phan_biet_truoc': int(raw[c].nunique(dropna=True)) if c in raw.columns else '',
        'so_gia_tri_phan_biet_sau': int(clean[c].nunique(dropna=True)) if c in clean.columns else '',
        'ghi_chu': 'Tạo mới khi tiền xử lý/feature engineering' if c not in raw.columns else ('Đã loại bỏ vì là chỉ mục dư thừa' if c=='Unnamed: 0' else 'Giữ lại và/hoặc làm sạch')
    })
pd.DataFrame(compare).to_csv(OUT/'reports'/'05_so_sanh_truoc_sau_tien_xu_ly.csv', index=False, encoding='utf-8-sig')

# summary csv
summary_before=pd.DataFrame({
    'chi_tieu':['so_dong','so_cot','tong_missing','so_dong_trung_lap','so_bien_dinh_luong','so_bien_object'],
    'gia_tri':[len(raw), raw.shape[1], int(raw.isna().sum().sum()), int(raw.duplicated().sum()), raw.select_dtypes(include=np.number).shape[1], raw.select_dtypes(include='object').shape[1]]
})
summary_after=pd.DataFrame({
    'chi_tieu':['so_dong','so_cot','tong_missing','so_dong_trung_lap','so_dong_trung_lap_da_xoa','so_bien_dinh_luong','so_bien_object_category'],
    'gia_tri':[len(clean), clean.shape[1], int(clean.isna().sum().sum()), int(clean.duplicated().sum()), removed_dup, clean.select_dtypes(include=np.number).shape[1], clean.select_dtypes(include=['object','category']).shape[1]]
})
summary_before.to_csv(OUT/'reports'/'summary_before.csv', index=False, encoding='utf-8-sig')
summary_after.to_csv(OUT/'reports'/'summary_after.csv', index=False, encoding='utf-8-sig')

# Draw charts before
bf=OUT/'charts'/'before_preprocessing'; af=OUT/'charts'/'after_preprocessing'
for c in ['Age','Rating','Recommended IND','Positive Feedback Count','Review Text Length','Title Length']:
    if c in raw.columns:
        hist(raw,c,bf,'before'); box(raw,c,bf,'before')
for c in ['Division Name','Department Name','Class Name','Recommended IND','Rating','Sentiment Label']:
    bar_counts(raw,c,bf,'before')
for x,y in [('Age','Rating'),('Age','Positive Feedback Count'),('Review Text Length','Rating'),('Review Text Length','Positive Feedback Count'),('Rating','Positive Feedback Count')]:
    scatter(raw,x,y,bf,'before')
line_group(raw,'Age','Rating',bf,'before','Average rating by age (before)')
line_group(raw,'Age','Positive Feedback Count',bf,'before','Average positive feedback by age (before)')
line_group(raw,'Rating','Recommended IND',bf,'before','Recommendation rate by rating (before)')
heatmap_corr(raw,bf,'before')
# Draw charts after using capped features for skewed visual
for c in ['Age','Rating','Recommended IND','Positive Feedback Count','Review Text Length','Title Length','Word Count','Has Title','Has Review Text',
          'Positive Feedback Count_capped_1_99','Review Text Length_capped_1_99','Word Count_capped_1_99']:
    if c in clean.columns:
        hist(clean,c,af,'after'); box(clean,c,af,'after')
for c in ['Division Name','Department Name','Class Name','Recommended Label','Sentiment Label','Age Group','Rating','Has Title','Has Review Text']:
    bar_counts(clean,c,af,'after')
for x,y in [('Age','Rating'),('Age','Positive Feedback Count'),('Review Text Length','Rating'),('Review Text Length','Positive Feedback Count'),('Word Count','Rating'),('Rating','Recommended IND')]:
    scatter(clean,x,y,af,'after')
line_group(clean,'Age','Rating',af,'after','Average rating by age (after)')
line_group(clean,'Age','Positive Feedback Count',af,'after','Average positive feedback by age (after)')
line_group(clean,'Rating','Recommended IND',af,'after','Recommendation rate by rating (after)')
# Rating/recommend by age group with categorical line via ordered labels
ag=clean.groupby('Age Group', observed=False).agg(avg_rating=('Rating','mean'), recommend_rate=('Recommended IND','mean'), count=('Rating','size')).reset_index()
ag.to_csv(OUT/'reports'/'07_age_group_summary.csv', index=False, encoding='utf-8-sig')
for y in ['avg_rating','recommend_rate','count']:
    plt.figure(figsize=(9,5)); plt.plot(ag['Age Group'].astype(str), ag[y], marker='o')
    plt.title(f'Line chart - {y} by Age Group'); plt.xlabel('Age Group'); plt.ylabel(y); plt.xticks(rotation=45)
    savefig(af/f'line_{y}_by_age_group_after.png')
heatmap_corr(clean,af,'after')

# analytical comments CSV
comments = [
    ['Phân loại biến','Dữ liệu gồm định lượng (Age, Rating, Positive Feedback Count), nhị phân (Recommended IND), định tính (Division/Department/Class), và văn bản phi cấu trúc (Title, Review Text).'],
    ['Định lượng','Rating tập trung cao ở mức 4-5; Positive Feedback Count lệch phải mạnh, nhiều dòng bằng 0 nhưng có một số bài review nhận nhiều phản hồi.'],
    ['Định tính','Tops, Dresses, Bottoms là các nhóm sản phẩm quan trọng; cần phân tích theo Department/Class để hiểu sản phẩm nào nhận đánh giá cao/thấp.'],
    ['Văn bản','Title và Review Text có thiếu dữ liệu; sau tiền xử lý được thay bằng No Title/No Review, đồng thời tạo Review Text Length, Word Count để trực quan hóa.'],
    ['Tiền xử lý','Loại bỏ Unnamed: 0, điền missing, chuẩn hóa chuỗi, tạo Sentiment Label từ Rating, Recommended Label từ Recommended IND, Age Group và cắt ngoại lệ 1%-99% phục vụ biểu đồ.'],
    ['Mô hình hóa','Có thể dùng Rating hoặc Sentiment Label làm nhãn cảm xúc; Recommended IND làm nhãn phân loại khuyến nghị như bài Agarap 2018.'],
]
pd.DataFrame(comments, columns=['muc','nhan_xet']).to_csv(OUT/'reports'/'08_nhan_xet_tong_hop.csv', index=False, encoding='utf-8-sig')

# Excel guide and README
readme = f"""
PHÂN TÍCH WOMEN'S E-COMMERCE CLOTHING REVIEWS - THEO MẪU OUTPUT

Nguồn dữ liệu: Kaggle Women's E-Commerce Clothing Reviews.
Bài báo tham khảo: Agarap, A. F. (2018), Statistical Analysis on E-Commerce Reviews, with Sentiment Classification using Bidirectional RNN, arXiv:1805.03687.

1. CẤU TRÚC THƯ MỤC
- charts/before_preprocessing: biểu đồ trước tiền xử lý
- charts/after_preprocessing: biểu đồ sau tiền xử lý
- reports: bảng thống kê CSV
- womens_clothing_reviews_cleaned.csv: dữ liệu đã tiền xử lý
- phan_tich_womens_ecommerce_reviews.py: code Python tái chạy toàn bộ phân tích

2. CÁC LOẠI BIỂU ĐỒ ĐÃ XUẤT
- Histogram: Age, Rating, Recommended IND, Positive Feedback Count, Review Text Length, Word Count...
- Boxplot: Age, Rating, Positive Feedback Count, Review/Text Length...
- Scatter Plot: Age vs Rating, Review Text Length vs Rating, Rating vs Positive Feedback Count...
- Bar Chart: Division Name, Department Name, Class Name, Recommended Label, Sentiment Label, Age Group...
- Line Chart: Rating trung bình theo tuổi, tỷ lệ recommended theo rating, thống kê theo Age Group...
- Heatmap: tương quan các biến định lượng

3. TIỀN XỬ LÝ ĐÃ THỰC HIỆN
- Xóa cột Unnamed: 0 vì chỉ là chỉ mục dư thừa.
- Điền Title thiếu bằng 'No Title'.
- Điền Review Text thiếu bằng 'No Review'.
- Điền Division/Department/Class thiếu bằng 'Unknown'.
- Tạo Review Text Length, Title Length, Word Count.
- Tạo Sentiment Label: Rating >= 4 là Positive, Rating = 3 là Neutral, Rating <= 2 là Negative.
- Tạo Recommended Label từ Recommended IND.
- Tạo Age Group.
- Tạo bản capping 1%-99% cho một số biến lệch phải để biểu đồ dễ đọc hơn.

4. HƯỚNG DẪN EXCEL STEP BY STEP
Bước 1: Data -> Get Data -> From Text/CSV -> chọn Womens Clothing E-Commerce Reviews.csv.
Bước 2: Kiểm tra missing bằng COUNTBLANK, ví dụ =COUNTBLANK(D:D) cho Title.
Bước 3: Tạo cột độ dài Review Text: =LEN(E2). Tạo Word Count: =LEN(TRIM(E2))-LEN(SUBSTITUTE(TRIM(E2)," ",""))+1.
Bước 4: Tạo Sentiment Label: =IF(F2>=4,"Positive",IF(F2=3,"Neutral","Negative")).
Bước 5: Tạo Recommended Label: =IF(G2=1,"Recommended","Not Recommended").
Bước 6: Insert -> PivotTable để đếm Rating, Department Name, Class Name, Sentiment Label.
Bước 7: Insert -> Histogram cho Age, Rating, Positive Feedback Count, Review Text Length.
Bước 8: Insert -> Box & Whisker cho Age, Positive Feedback Count.
Bước 9: Insert -> Scatter cho Review Text Length và Rating.
Bước 10: Insert -> Line chart cho Rating trung bình theo Age hoặc Age Group.
Bước 11: Tạo ma trận CORREL giữa các biến số rồi dùng Conditional Formatting -> Color Scales để tạo heatmap.

5. ƯU ĐIỂM / KHUYẾT ĐIỂM
Python:
Ưu điểm: tự động hóa cao, xử lý lặp lại nhanh, xuất nhiều biểu đồ, dễ tái lập kết quả, phù hợp NLP.
Khuyết điểm: cần biết code, lỗi môi trường/thư viện có thể gây khó cho người mới.
Excel:
Ưu điểm: dễ thao tác, PivotTable trực quan, phù hợp báo cáo nhanh và trình bày với giáo viên.
Khuyết điểm: khó tự động hóa nhiều biểu đồ, xử lý văn bản/NLP hạn chế, dễ sai khi thao tác thủ công.

6. NHẬN XÉT TỔNG HỢP
Dữ liệu phù hợp cho phân tích hành vi đánh giá khách hàng và bài toán phân loại cảm xúc/khuyến nghị. Rating và Recommended IND là hai biến quan trọng nhất. Review Text và Title là nguồn dữ liệu văn bản quan trọng cho mô hình NLP như BiLSTM trong bài Agarap 2018. Sau tiền xử lý, dữ liệu đầy đủ hơn, có thêm biến đặc trưng độ dài văn bản, nhóm tuổi và nhãn cảm xúc, giúp trực quan hóa và mô hình hóa tốt hơn.
""".strip()
(OUT/'README.txt').write_text(readme, encoding='utf-8')

# Python code copy (this script as output runnable)
script_out = OUT/'phan_tich_womens_ecommerce_reviews.py'
shutil.copyfile(Path(__file__), script_out)

# Zip
if ZIP.exists(): ZIP.unlink()
with zipfile.ZipFile(ZIP,'w',zipfile.ZIP_DEFLATED) as z:
    for p in OUT.rglob('*'):
        z.write(p, p.relative_to(OUT.parent))
print('CREATED', ZIP, ZIP.stat().st_size)
print('OUT', OUT)
print('charts before', len(list((OUT/'charts'/'before_preprocessing').glob('*.png'))),'after', len(list((OUT/'charts'/'after_preprocessing').glob('*.png'))))
