from pathlib import Path
import shutil, zipfile
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

base = Path('/mnt/data')
out = base/'store_sales_outputs_with_train_like_mau'
charts = out/'charts'
before_dir = charts/'before_preprocessing'
after_dir = charts/'after_preprocessing'
reports = out/'reports'
if out.exists(): shutil.rmtree(out)
for p in [before_dir, after_dir, reports]: p.mkdir(parents=True, exist_ok=True)

# Input
train = pd.read_csv(base/'train.csv (1).zip', compression='zip', parse_dates=['date'])
stores = pd.read_csv(base/'stores(1).csv')
test = pd.read_csv(base/'test(1).csv', parse_dates=['date'])
transactions = pd.read_csv(base/'transactions(1).csv', parse_dates=['date'])
oil = pd.read_csv(base/'oil(1).csv', parse_dates=['date'])
holidays = pd.read_csv(base/'holidays_events(1).csv', parse_dates=['date'])
submission = pd.read_csv(base/'sample_submission(1).csv')

# Classification
file_map = {'train.csv':train,'stores.csv':stores,'test.csv':test,'transactions.csv':transactions,'oil.csv':oil,'holidays_events.csv':holidays,'sample_submission.csv':submission}
def nature(col, ser):
    dtype=str(ser.dtype)
    if col=='id': return 'Biến định danh'
    if col=='date' or dtype.startswith('datetime'): return 'Biến thời gian'
    if col=='store_nbr': return 'Biến định danh / định tính mã hóa số'
    if col=='cluster': return 'Biến định tính nhóm/cụm mã hóa số'
    if col=='family': return 'Biến định tính danh nghĩa - nhóm sản phẩm'
    if col=='sales': return 'Biến mục tiêu định lượng liên tục cần dự báo'
    if col in ['onpromotion','transactions']: return 'Biến định lượng rời rạc'
    if col=='dcoilwtico': return 'Biến định lượng liên tục'
    if col=='transferred': return 'Biến nhị phân Boolean'
    if dtype in ['object','bool'] or dtype=='boolean': return 'Biến định tính'
    if np.issubdtype(ser.dtype, np.number): return 'Biến định lượng'
    return 'Khác'
rows=[]
for fname, df in file_map.items():
    for c in df.columns:
        rows.append({'file':fname,'ten_bien':c,'kieu_du_lieu_python':str(df[c].dtype),'ban_chat_bien':nature(c,df[c]),'so_gia_tri_khuyet':int(df[c].isna().sum()),'ty_le_khuyet_%':round(float(df[c].isna().mean()*100),4),'so_gia_tri_khac_nhau':int(df[c].nunique(dropna=True))})
pd.DataFrame(rows).to_csv(reports/'01_phan_loai_bien.csv', index=False, encoding='utf-8-sig')

# Before summaries: based on original files
summary_before = pd.DataFrame([
    {'ten_bo_du_lieu':'train.csv','so_dong':train.shape[0],'so_cot':train.shape[1],'so_du_lieu_khuyet':int(train.isna().sum().sum()),'so_dong_trung_lap':int(train.duplicated().sum()),'ngay_min':train.date.min(),'ngay_max':train.date.max(),'tong_sales':train.sales.sum(),'sales_trung_binh':train.sales.mean()},
    {'ten_bo_du_lieu':'oil.csv','so_dong':oil.shape[0],'so_cot':oil.shape[1],'so_du_lieu_khuyet':int(oil.isna().sum().sum()),'so_dong_trung_lap':int(oil.duplicated().sum()),'ngay_min':oil.date.min(),'ngay_max':oil.date.max(),'tong_sales':np.nan,'sales_trung_binh':np.nan},
    {'ten_bo_du_lieu':'transactions.csv','so_dong':transactions.shape[0],'so_cot':transactions.shape[1],'so_du_lieu_khuyet':int(transactions.isna().sum().sum()),'so_dong_trung_lap':int(transactions.duplicated().sum()),'ngay_min':transactions.date.min(),'ngay_max':transactions.date.max(),'tong_sales':np.nan,'sales_trung_binh':np.nan},
])
summary_before.to_csv(reports/'summary_before.csv', index=False, encoding='utf-8-sig')

# Numeric before combined descriptions
num_desc_parts=[]
for name, df, cols in [('train',train,['sales','onpromotion']),('transactions',transactions,['transactions']),('oil',oil,['dcoilwtico']),('stores',stores,['cluster'])]:
    desc=df[cols].describe().T
    desc.insert(0,'source',name)
    num_desc_parts.append(desc)
pd.concat(num_desc_parts).to_csv(reports/'02_mo_ta_bien_dinh_luong_before.csv', encoding='utf-8-sig')

# Categorical before
cat_items=[('train','family',train),('train','store_nbr',train),('stores','city',stores),('stores','state',stores),('stores','type',stores),('stores','cluster',stores),('holidays_events','type',holidays),('holidays_events','locale',holidays),('holidays_events','transferred',holidays)]
cat_rows=[]
for source,c,df in cat_items:
    top=df[c].astype(str).value_counts(dropna=False).head(15)
    cat_rows.append({'source':source,'bien':c,'so_gia_tri_khac_nhau':int(df[c].nunique(dropna=True)),'so_khuyet':int(df[c].isna().sum()),'top_15_gia_tri':' | '.join([f'{idx}: {val}' for idx,val in top.items()])})
pd.DataFrame(cat_rows).to_csv(reports/'06_mo_ta_bien_dinh_tinh.csv', index=False, encoding='utf-8-sig')

# Preprocess support tables
oil_clean=oil.sort_values('date').copy(); oil_clean['dcoilwtico']=oil_clean['dcoilwtico'].ffill().bfill()
holiday_day=holidays.copy(); holiday_day['is_holiday']=1
holiday_day=holiday_day.sort_values('date').drop_duplicates('date')[['date','type','locale','locale_name','description','transferred','is_holiday']].rename(columns={'type':'holiday_type','locale':'holiday_locale','locale_name':'holiday_locale_name','description':'holiday_description'})

# Sample cleaned merged for Excel/demo
sample_n=min(100000,len(train))
train_sample=train.sample(sample_n, random_state=42).copy()
clean_sample=train_sample.merge(stores,on='store_nbr',how='left').merge(oil_clean,on='date',how='left').merge(holiday_day,on='date',how='left').merge(transactions,on=['date','store_nbr'],how='left')
clean_sample['transactions']=clean_sample['transactions'].fillna(0)
clean_sample['is_holiday']=clean_sample['is_holiday'].fillna(0).astype(int)
for c in ['holiday_type','holiday_locale','holiday_locale_name','holiday_description']:
    clean_sample[c]=clean_sample[c].fillna('No holiday')
clean_sample['transferred']=clean_sample['transferred'].fillna(False)
clean_sample['year']=clean_sample['date'].dt.year; clean_sample['month']=clean_sample['date'].dt.month; clean_sample['day']=clean_sample['date'].dt.day
clean_sample['dayofweek']=clean_sample['date'].dt.dayofweek; clean_sample['weekofyear']=clean_sample['date'].dt.isocalendar().week.astype(int); clean_sample['is_weekend']=clean_sample['dayofweek'].isin([5,6]).astype(int)
clean_sample.to_csv(out/'store_sales_train_merged_cleaned_sample_100k.csv', index=False, encoding='utf-8-sig')

# Aggregations from full train + support
train_time=train.copy()
train_time['year']=train_time['date'].dt.year; train_time['month']=train_time['date'].dt.month; train_time['dayofweek']=train_time['date'].dt.dayofweek; train_time['is_weekend']=train_time['dayofweek'].isin([5,6]).astype(int)
daily=train_time.groupby('date', as_index=False).agg(total_sales=('sales','sum'),avg_sales=('sales','mean'),total_onpromotion=('onpromotion','sum'))
daily=daily.merge(oil_clean,on='date',how='left').merge(holiday_day[['date','is_holiday']],on='date',how='left')
daily['is_holiday']=daily['is_holiday'].fillna(0).astype(int)
trans_daily=transactions.groupby('date',as_index=False)['transactions'].sum().rename(columns={'transactions':'total_transactions'})
daily=daily.merge(trans_daily,on='date',how='left'); daily['total_transactions']=daily['total_transactions'].fillna(0)
daily['year']=daily['date'].dt.year; daily['month']=daily['date'].dt.month; daily['dayofweek']=daily['date'].dt.dayofweek; daily['is_weekend']=daily['dayofweek'].isin([5,6]).astype(int)
monthly=daily.groupby(['year','month'],as_index=False).agg(total_sales=('total_sales','sum'),avg_sales=('avg_sales','mean'),total_onpromotion=('total_onpromotion','sum'),avg_oil=('dcoilwtico','mean'),total_transactions=('total_transactions','sum'))
sales_by_family=train.groupby('family',as_index=False).agg(total_sales=('sales','sum'),avg_sales=('sales','mean'),total_onpromotion=('onpromotion','sum'),rows=('id','count')).sort_values('total_sales',ascending=False)
sales_by_store=train.groupby('store_nbr',as_index=False).agg(total_sales=('sales','sum'),avg_sales=('sales','mean'),total_onpromotion=('onpromotion','sum'),rows=('id','count')).merge(stores,on='store_nbr',how='left').sort_values('total_sales',ascending=False)
daily.to_csv(out/'store_sales_daily_aggregated_cleaned.csv',index=False,encoding='utf-8-sig')
monthly.to_csv(reports/'10_doanh_thu_theo_thang.csv',index=False,encoding='utf-8-sig')
sales_by_family.to_csv(reports/'07_doanh_thu_theo_family.csv',index=False,encoding='utf-8-sig')
sales_by_store.to_csv(reports/'08_doanh_thu_theo_store.csv',index=False,encoding='utf-8-sig')
daily.to_csv(reports/'09_doanh_thu_theo_ngay.csv',index=False,encoding='utf-8-sig')

# Outlier report and after desc using clean sample + daily
outlier_rows=[]
for c in ['sales','onpromotion','dcoilwtico','transactions']:
    s=clean_sample[c].dropna()
    q1,q3=s.quantile(.25),s.quantile(.75); iqr=q3-q1; lower,upper=q1-1.5*iqr,q3+1.5*iqr
    if iqr==0: lower,upper=s.min(),s.max()
    outlier_rows.append({'bien':c,'q1':q1,'q3':q3,'iqr':iqr,'can_duoi':lower,'can_tren':upper,'so_ngoai_le_truoc_clip':int(((s<lower)|(s>upper)).sum()),'so_ngoai_le_sau_clip':0})
pd.DataFrame(outlier_rows).to_csv(reports/'03_outlier_iqr_report.csv', index=False, encoding='utf-8-sig')
chart_after=clean_sample.copy()
for r in outlier_rows:
    c=r['bien']; chart_after[c]=chart_after[c].clip(r['can_duoi'],r['can_tren'])
chart_after[['sales','onpromotion','dcoilwtico','transactions','cluster','year','month','dayofweek','is_weekend','is_holiday']].describe().T.to_csv(reports/'04_mo_ta_bien_dinh_luong_after.csv', encoding='utf-8-sig')
pd.DataFrame([{'ten_bo_du_lieu':'clean_sample_and_aggregations','so_dong_sample':clean_sample.shape[0],'so_cot_sample':clean_sample.shape[1],'so_du_lieu_khuyet_sample':int(clean_sample.isna().sum().sum()),'so_dong_daily':daily.shape[0],'tong_sales':train.sales.sum(),'sales_trung_binh':train.sales.mean()}]).to_csv(reports/'summary_after.csv', index=False, encoding='utf-8-sig')
compare=[]
for c in ['sales','onpromotion']:
    compare.append({'bien':c,'mean_before_full_train':train[c].mean(),'mean_after_sample_clipped':chart_after[c].mean(),'median_before_full_train':train[c].median(),'median_after_sample_clipped':chart_after[c].median(),'std_before_full_train':train[c].std(),'std_after_sample_clipped':chart_after[c].std(),'min_before_full_train':train[c].min(),'min_after_sample_clipped':chart_after[c].min(),'max_before_full_train':train[c].max(),'max_after_sample_clipped':chart_after[c].max(),'missing_before':int(train[c].isna().sum()),'missing_after_sample':int(clean_sample[c].isna().sum())})
for c,df in [('dcoilwtico',oil),('transactions',transactions)]:
    after_s=chart_after[c]
    before_s=df[c]
    compare.append({'bien':c,'mean_before_full_train':before_s.mean(),'mean_after_sample_clipped':after_s.mean(),'median_before_full_train':before_s.median(),'median_after_sample_clipped':after_s.median(),'std_before_full_train':before_s.std(),'std_after_sample_clipped':after_s.std(),'min_before_full_train':before_s.min(),'min_after_sample_clipped':after_s.min(),'max_before_full_train':before_s.max(),'max_after_sample_clipped':after_s.max(),'missing_before':int(before_s.isna().sum()),'missing_after_sample':int(after_s.isna().sum())})
pd.DataFrame(compare).to_csv(reports/'05_so_sanh_truoc_sau_tien_xu_ly.csv', index=False, encoding='utf-8-sig')

# Charts
before_sample=train.sample(min(100000,len(train)), random_state=1).merge(stores,on='store_nbr',how='left')
before_sample=before_sample.merge(oil,on='date',how='left').merge(transactions,on=['date','store_nbr'],how='left')

def safe(x): return ''.join(ch if ch.isalnum() or ch in ['_','-'] else '_' for ch in str(x))
def hist(df, folder, label):
    for c in ['sales','onpromotion','dcoilwtico','transactions']:
        if c not in df: continue
        s=df[c].dropna()
        plt.figure(figsize=(8,5)); plt.hist(s,bins=40); plt.title(f'Histogram - {c} ({label})'); plt.xlabel(c); plt.ylabel('Tần suất'); plt.tight_layout(); plt.savefig(folder/f'histogram_{safe(c)}_{label}.png',dpi=150); plt.close()
def box(df, folder, label):
    for c in ['sales','onpromotion','dcoilwtico','transactions']:
        if c not in df: continue
        s=df[c].dropna()
        plt.figure(figsize=(7,5)); plt.boxplot(s,vert=True); plt.title(f'Boxplot - {c} ({label})'); plt.ylabel(c); plt.tight_layout(); plt.savefig(folder/f'boxplot_{safe(c)}_{label}.png',dpi=150); plt.close()
def bar(df, folder, label):
    for c in ['family','city','state','type','cluster','store_nbr']:
        if c not in df: continue
        counts=df[c].astype(str).value_counts().head(20)
        plt.figure(figsize=(10,5)); counts.plot(kind='bar'); plt.title(f'Bar Chart - {c} ({label})'); plt.xlabel(c); plt.ylabel('Số lượng'); plt.xticks(rotation=45,ha='right'); plt.tight_layout(); plt.savefig(folder/f'bar_{safe(c)}_{label}.png',dpi=150); plt.close()
    top=sales_by_family.head(20).set_index('family')['total_sales']
    plt.figure(figsize=(11,5)); top.plot(kind='bar'); plt.title(f'Top 20 family theo sales ({label})'); plt.xlabel('family'); plt.ylabel('sales'); plt.xticks(rotation=45,ha='right'); plt.tight_layout(); plt.savefig(folder/f'bar_top_family_sales_{label}.png',dpi=150); plt.close()
    topst=sales_by_store.head(20).set_index('store_nbr')['total_sales']
    plt.figure(figsize=(11,5)); topst.plot(kind='bar'); plt.title(f'Top 20 store theo sales ({label})'); plt.xlabel('store_nbr'); plt.ylabel('sales'); plt.tight_layout(); plt.savefig(folder/f'bar_top_store_sales_{label}.png',dpi=150); plt.close()
def scatter(df, folder, label):
    for x,y in [('onpromotion','sales'),('transactions','sales'),('dcoilwtico','sales'),('onpromotion','transactions')]:
        if x in df and y in df:
            tmp=df[[x,y]].dropna()
            plt.figure(figsize=(8,5)); plt.scatter(tmp[x],tmp[y],alpha=.25,s=8); plt.title(f'Scatter Plot - {x} vs {y} ({label})'); plt.xlabel(x); plt.ylabel(y); plt.tight_layout(); plt.savefig(folder/f'scatter_{safe(x)}_vs_{safe(y)}_{label}.png',dpi=150); plt.close()
def line(folder,label):
    for y in ['total_sales','total_transactions','dcoilwtico','total_onpromotion']:
        plt.figure(figsize=(12,5)); plt.plot(daily['date'],daily[y],linewidth=1); plt.title(f'Line Chart - {y} theo ngày ({label})'); plt.xlabel('date'); plt.ylabel(y); plt.tight_layout(); plt.savefig(folder/f'line_{safe(y)}_by_date_{label}.png',dpi=150); plt.close()
    m=monthly.copy(); m['ym']=pd.to_datetime(m['year'].astype(str)+'-'+m['month'].astype(str)+'-01')
    plt.figure(figsize=(12,5)); plt.plot(m['ym'],m['total_sales'],marker='o'); plt.title(f'Line Chart - total_sales theo tháng ({label})'); plt.xlabel('month'); plt.ylabel('sales'); plt.tight_layout(); plt.savefig(folder/f'line_total_sales_by_month_{label}.png',dpi=150); plt.close()
def heat(df, folder,label):
    cols=[c for c in ['sales','onpromotion','dcoilwtico','transactions','cluster'] if c in df]
    corr=df[cols].corr(numeric_only=True)
    fig,ax=plt.subplots(figsize=(8,6)); im=ax.imshow(corr.values,aspect='auto'); ax.set_xticks(range(len(corr.columns))); ax.set_yticks(range(len(corr.index))); ax.set_xticklabels(corr.columns,rotation=45,ha='right'); ax.set_yticklabels(corr.index)
    for i in range(len(corr.index)):
        for j in range(len(corr.columns)): ax.text(j,i,f'{corr.iloc[i,j]:.2f}',ha='center',va='center',fontsize=9)
    plt.title(f'Heatmap tương quan ({label})'); plt.colorbar(im,ax=ax,fraction=0.046,pad=0.04); plt.tight_layout(); plt.savefig(folder/f'heatmap_correlation_{label}.png',dpi=150); plt.close()
for df,folder,label in [(before_sample,before_dir,'before'),(chart_after,after_dir,'after')]:
    hist(df,folder,label); box(df,folder,label); bar(df,folder,label); scatter(df,folder,label); line(folder,label); heat(df,folder,label)

# README and script copy
readme=f'''PHÂN TÍCH STORE SALES TIME SERIES FORECASTING - ĐÃ BỔ SUNG TRAIN.CSV

File này phân tích theo cấu trúc giống mau_outputs.zip, gồm before_preprocessing, after_preprocessing, reports và code Python.

Điểm khác file trước:
- Đã có train.csv nên phân tích trực tiếp biến mục tiêu sales.
- Có thống kê doanh thu theo ngày, tháng, family, store.
- Có biểu đồ sales: histogram, boxplot, scatter với promotion/transactions/oil, line chart theo ngày/tháng.

Cấu trúc:
charts/before_preprocessing/  : Biểu đồ trước tiền xử lý
charts/after_preprocessing/   : Biểu đồ sau tiền xử lý và clip outlier để dễ quan sát
reports/                      : Các bảng CSV phân tích
store_sales_train_merged_cleaned_sample_100k.csv : Mẫu 100.000 dòng đã merge + tiền xử lý để mở Excel nhẹ
store_sales_daily_aggregated_cleaned.csv         : Tổng hợp ngày từ toàn bộ train
phan_tich_store_sales_with_train.py              : Code Python tạo lại output

Thông tin chính:
- train.csv: {train.shape[0]:,} dòng, {train.shape[1]} cột
- Giai đoạn: {train.date.min().date()} đến {train.date.max().date()}
- Tổng sales: {train.sales.sum():,.2f}
- Sales trung bình mỗi dòng: {train.sales.mean():,.4f}
- Số family: {train.family.nunique()}
- Số cửa hàng: {train.store_nbr.nunique()}

Lưu ý:
Để file ZIP không quá nặng, dữ liệu cleaned chi tiết chỉ lưu mẫu 100.000 dòng. Các báo cáo tổng hợp và biểu đồ vẫn được tính từ toàn bộ train khi cần.
'''
(out/'README.txt').write_text(readme,encoding='utf-8')
shutil.copy(base/'create_store_sales_with_train_like_mau_fast.py', out/'phan_tich_store_sales_with_train.py')
zip_path=base/'store_sales_outputs_with_train_like_mau.zip'
if zip_path.exists(): zip_path.unlink()
with zipfile.ZipFile(zip_path,'w',zipfile.ZIP_DEFLATED) as z:
    for path in out.rglob('*'):
        if path.is_file(): z.write(path, arcname=str(path.relative_to(out.parent)))
print(zip_path)
