import csv, re, math, statistics, random, os
from collections import Counter, defaultdict
from artifact_tool import Workbook, SpreadsheetFile

CSV_PATH = '/mnt/data/Womens Clothing E-Commerce Reviews(2).csv'
OUT_PATH = '/mnt/data/womens_ecommerce_excel_auto_charts.xlsx'

# ---------- Data loading & preprocessing ----------
def to_int(x):
    try:
        if x is None or str(x).strip() == '': return None
        return int(float(str(x).strip()))
    except Exception:
        return None

def text_len(s):
    return len(s or '')

def word_count(s):
    return len(re.findall(r"\b\w+\b", (s or '').lower()))

def sentiment_label(rating):
    if rating is None: return 'Missing'
    if rating <= 2: return 'Negative'
    if rating == 3: return 'Neutral'
    return 'Positive'

def recommended_label(v):
    if v is None: return 'Missing'
    return 'Recommended' if v == 1 else 'Not recommended'

def age_group(age):
    if age is None: return 'Missing'
    if age < 25: return '<25'
    if age <= 34: return '25-34'
    if age <= 44: return '35-44'
    if age <= 54: return '45-54'
    if age <= 64: return '55-64'
    return '65+'

def build_record(row):
    title = row.get('Title') or ''
    review = row.get('Review Text') or ''
    age = to_int(row.get('Age'))
    rating = to_int(row.get('Rating'))
    rec = to_int(row.get('Recommended IND'))
    feedback = to_int(row.get('Positive Feedback Count'))
    return {
        'Clothing ID': to_int(row.get('Clothing ID')),
        'Age': age,
        'Title': title,
        'Review Text': review,
        'Rating': rating,
        'Recommended IND': rec,
        'Positive Feedback Count': feedback,
        'Division Name': (row.get('Division Name') or '').strip() or 'Missing',
        'Department Name': (row.get('Department Name') or '').strip() or 'Missing',
        'Class Name': (row.get('Class Name') or '').strip() or 'Missing',
        'Review Text Length': text_len(review),
        'Review Word Count': word_count(review),
        'Title Length': text_len(title),
        'Recommended Label': recommended_label(rec),
        'Sentiment Label': sentiment_label(rating),
        'Age Group': age_group(age),
        'Has Title': 'Has title' if title.strip() else 'No title'
    }

raw = []
with open(CSV_PATH, newline='', encoding='utf-8') as f:
    for row in csv.DictReader(f):
        raw.append(build_record(row))

num_cols = ['Age','Rating','Recommended IND','Positive Feedback Count','Review Text Length','Review Word Count','Title Length']
cat_cols = ['Rating','Recommended IND','Recommended Label','Sentiment Label','Division Name','Department Name','Class Name','Age Group','Has Title']

# after preprocessing: remove missing key text/categorical/numeric rows, clip continuous outliers
clean = []
for r in raw:
    if not r['Review Text'].strip():
        continue
    if r['Division Name'] == 'Missing' or r['Department Name'] == 'Missing' or r['Class Name'] == 'Missing':
        continue
    if r['Age'] is None or r['Rating'] is None or r['Recommended IND'] is None or r['Positive Feedback Count'] is None:
        continue
    clean.append(dict(r))

def quantile(vals, q):
    vals = sorted([v for v in vals if v is not None])
    if not vals: return None
    pos = (len(vals)-1)*q
    lo = math.floor(pos); hi = math.ceil(pos)
    if lo == hi: return vals[lo]
    return vals[lo] + (vals[hi]-vals[lo])*(pos-lo)

# clip outliers for skewed continuous variables only
for col in ['Age','Positive Feedback Count','Review Text Length','Review Word Count','Title Length']:
    vals = [r[col] for r in clean if r[col] is not None]
    q1, q3 = quantile(vals, .25), quantile(vals, .75)
    iqr = q3 - q1
    low, high = q1 - 1.5*iqr, q3 + 1.5*iqr
    for r in clean:
        if r[col] is not None:
            r[col] = max(low, min(high, r[col]))
    if col == 'Age':
        for r in clean:
            r['Age Group'] = age_group(r['Age'])

datasets = {'before': raw, 'after': clean}

# ---------- Aggregations ----------
def numeric_values(data, col):
    return [r[col] for r in data if isinstance(r.get(col), (int,float)) and r.get(col) is not None and not math.isnan(r[col])]

def top_counts(data, col, max_items=12):
    c = Counter(str(r.get(col, 'Missing')) for r in data)
    items = c.most_common(max_items)
    return [k for k,v in items], [v for k,v in items]

def hist_bins(vals, bins=10):
    vals = [float(v) for v in vals if v is not None]
    if not vals: return [], []
    mn, mx = min(vals), max(vals)
    if mn == mx:
        return [str(mn)], [len(vals)]
    step = (mx-mn)/bins
    edges = [mn+i*step for i in range(bins+1)]
    counts=[0]*bins
    for v in vals:
        idx = min(int((v-mn)/step), bins-1)
        counts[idx]+=1
    labels=[]
    for i in range(bins):
        labels.append(f"{edges[i]:.0f}-{edges[i+1]:.0f}")
    return labels, counts

def group_count_by(data, col):
    buckets=defaultdict(int)
    for r in data:
        if r.get(col) is not None and r.get(col)!='Missing': buckets[r[col]] += 1
    items=sorted(buckets.items(), key=lambda kv: kv[0])
    return [str(k) for k,v in items], [v for k,v in items]

def group_avg_by(data, key, val):
    d=defaultdict(list)
    for r in data:
        if r.get(key) is not None and r.get(key)!='Missing' and r.get(val) is not None:
            d[r[key]].append(r[val])
    items=sorted(d.items(), key=lambda kv: kv[0])
    return [str(k) for k,v in items], [round(sum(v)/len(v), 3) for k,v in items]

def rec_rate_by(data, key):
    d=defaultdict(list)
    for r in data:
        if r.get(key) is not None and r.get(key)!='Missing' and r.get('Recommended IND') is not None:
            d[r[key]].append(r['Recommended IND'])
    # age group order
    order={'<25':1,'25-34':2,'35-44':3,'45-54':4,'55-64':5,'65+':6}
    items=sorted(d.items(), key=lambda kv: order.get(kv[0], kv[0]))
    return [str(k) for k,v in items], [round(sum(v)/len(v)*100, 2) for k,v in items]

def avg_feedback_by_rating(data):
    d=defaultdict(list)
    for r in data:
        if r.get('Rating') is not None and r.get('Positive Feedback Count') is not None:
            d[r['Rating']].append(r['Positive Feedback Count'])
    items=sorted(d.items())
    return [str(k) for k,v in items], [round(sum(v)/len(v), 3) for k,v in items]

def grouped_rating_by_department(data):
    depts = [k for k,v in Counter(r['Department Name'] for r in data).most_common(6)]
    rating_levels = [1,2,3,4,5]
    result = []
    for rating in rating_levels:
        vals=[]
        for dept in depts:
            vals.append(sum(1 for r in data if r['Department Name']==dept and r['Rating']==rating))
        result.append((str(rating), vals))
    return depts, [str(x) for x in rating_levels], result

def top_class_rec_rate(data):
    d=defaultdict(list)
    for r in data:
        if r['Class Name'] != 'Missing' and r.get('Recommended IND') is not None:
            d[r['Class Name']].append(r['Recommended IND'])
    items=[]
    for k, vals in d.items():
        if len(vals)>=30:
            items.append((k, sum(vals)/len(vals)*100, len(vals)))
    items=sorted(items, key=lambda x: x[1], reverse=True)[:12]
    return [k for k,rate,n in items], [round(rate,2) for k,rate,n in items]

def corr_matrix(data, cols):
    def corr(a,b):
        pairs=[(r[a],r[b]) for r in data if r.get(a) is not None and r.get(b) is not None]
        if len(pairs)<2: return 0
        xs, ys = zip(*pairs)
        mx, my = sum(xs)/len(xs), sum(ys)/len(ys)
        sx = math.sqrt(sum((x-mx)**2 for x in xs))
        sy = math.sqrt(sum((y-my)**2 for y in ys))
        if sx==0 or sy==0: return 0
        return sum((x-mx)*(y-my) for x,y in pairs)/(sx*sy)
    return [[round(corr(a,b),3) for b in cols] for a in cols]

def sample_pairs(data, xcol, ycol, maxn=500):
    pairs=[(r[xcol], r[ycol]) for r in data if r.get(xcol) is not None and r.get(ycol) is not None]
    if len(pairs)>maxn:
        random.seed(42); pairs=random.sample(pairs,maxn)
    return [p[0] for p in pairs], [p[1] for p in pairs]

# ---------- Workbook helpers ----------
wb = Workbook.create()
HEADER_FMT = {"fill":"#1F4E79","font":{"bold":True,"color":"#FFFFFF"},"horizontal_alignment":"center","vertical_alignment":"center","wrap_text":True}
SUB_FMT = {"fill":"#D9EAF7","font":{"bold":True},"wrap_text":True}

def write_table(sheet, start_row, start_col, rows):
    if not rows: return
    rng = sheet.get_range_by_indexes(start_row, start_col, len(rows), len(rows[0]))
    rng.values = rows
    return rng

def style_used(sheet, range_a1=None):
    try:
        if range_a1:
            sheet.get_range(range_a1).format.autofit_columns()
        else:
            sheet.get_range("A1:Z200").format.autofit_columns()
    except Exception:
        pass

def add_chart(sheet, chart_type, title, categories=None, values=None, xvalues=None, series_list=None, row=1, col=0, width=430, height=260, y_suffix=None):
    chart = sheet.charts.add(chart_type, {'from': {'row': row, 'col': col}, 'extent': {'widthPx': width, 'heightPx': height}})
    try: chart.title_text = title
    except Exception:
        try: chart.title = title
        except Exception: pass
    if series_list:
        if categories is not None:
            try: chart.categories = categories
            except Exception: pass
        for name, vals in series_list:
            s = chart.series.add(name)
            if categories is not None:
                try: s.categories = categories
                except Exception: pass
            s.values = vals
    else:
        s = chart.series.add(title[:25])
        if chart_type == 'scatter':
            s.x_values = xvalues or categories or []
            s.values = values or []
        else:
            try: chart.categories = categories or []
            except Exception: pass
            try: s.categories = categories or []
            except Exception: pass
            s.values = values or []
    try:
        chart.has_legend = False
    except Exception:
        pass
    return chart

# ---------- Overview and instructions ----------
readme = wb.worksheets.add('README')
readme.get_range('A1:H1').merge()
readme.get_range('A1').values = [["Women's E-Commerce Clothing Reviews - Excel Auto Charts"]]
readme.get_range('A1').format = {"fill":"#0F766E","font":{"bold":True,"color":"#FFFFFF","size":16},"horizontal_alignment":"center"}
readme.get_range('A3:B13').values = [
    ['Nguồn dữ liệu', 'Kaggle: nicapotato/womens-ecommerce-clothing-reviews'],
    ['Bài báo tham chiếu', 'Statistical Analysis on E-Commerce Reviews, with Sentiment Classification using Bidirectional RNN (Agarap, 2018)'],
    ['arXiv', 'https://arxiv.org/abs/1805.03687'],
    ['Số dòng trước tiền xử lý', len(raw)],
    ['Số dòng sau tiền xử lý', len(clean)],
    ['Biểu đồ trước tiền xử lý', 39],
    ['Biểu đồ sau tiền xử lý', 41],
    ['Cách dùng', 'Mở các sheet Charts_* để xem biểu đồ đã vẽ sẵn. Xem Excel_Steps để tự vẽ thủ công từng nhóm biểu đồ.'],
    ['Ghi chú', 'Boxplot và heatmap được tạo bằng công cụ biểu đồ/định dạng trong Excel. Heatmap dùng ma trận tương quan + Conditional Formatting.'],
    ['Tác vụ tự động', 'Workbook này đã dựng sẵn chart objects bằng code; không cần chạy macro để xem biểu đồ.']
]
readme.get_range('A3:A13').format = SUB_FMT
readme.get_range('A:B').format.column_width = 34
readme.get_range('B:B').format.column_width = 85
readme.get_range('B:B').format.wrap_text = True

# variable classification
var_sheet = wb.worksheets.add('Variable_Classification')
var_rows = [['Biến','Bản chất','Kiểu dữ liệu','Vai trò phân tích','Ghi chú']]
var_meta = [
    ('Clothing ID','Định tính định danh','Numeric ID','Mã sản phẩm','Không nên xem là biến định lượng liên tục'),
    ('Age','Định lượng rời rạc','Integer','Tuổi khách hàng','Dùng histogram, boxplot, line theo tuổi'),
    ('Title','Định tính văn bản','Text','Tiêu đề review','Tạo Title Length, Has Title'),
    ('Review Text','Định tính văn bản','Text','Nội dung review','Tạo Review Text Length, Word Count'),
    ('Rating','Định lượng thứ bậc / định tính thứ bậc','Integer 1-5','Điểm đánh giá','Có thể vẽ histogram/bar'),
    ('Recommended IND','Định tính nhị phân','0/1','Có khuyến nghị hay không','Chuyển thành Recommended Label'),
    ('Positive Feedback Count','Định lượng rời rạc','Integer','Số phản hồi tích cực','Thường lệch phải, cần kiểm tra outlier'),
    ('Division Name','Định tính danh nghĩa','Category','Nhóm sản phẩm cấp Division','Bar chart'),
    ('Department Name','Định tính danh nghĩa','Category','Nhóm sản phẩm cấp Department','Bar chart, grouped bar'),
    ('Class Name','Định tính danh nghĩa','Category','Lớp sản phẩm','Bar chart'),
    ('Review Text Length','Định lượng rời rạc','Integer','Độ dài nội dung','Biến tạo thêm'),
    ('Review Word Count','Định lượng rời rạc','Integer','Số từ trong review','Biến tạo thêm'),
    ('Title Length','Định lượng rời rạc','Integer','Độ dài tiêu đề','Biến tạo thêm'),
    ('Sentiment Label','Định tính thứ bậc','Category','Nhóm cảm xúc theo rating','Negative/Neutral/Positive'),
    ('Age Group','Định tính thứ bậc','Category','Nhóm tuổi','Phục vụ bar/line chart')
]
var_rows += [list(x) for x in var_meta]
write_table(var_sheet,0,0,var_rows)
var_sheet.get_range('A1:E1').format = HEADER_FMT
var_sheet.get_range('A:E').format.column_width = 24
var_sheet.get_range('E:E').format.column_width = 42
var_sheet.get_range('A:E').format.wrap_text = True

# Excel steps
steps = wb.worksheets.add('Excel_Steps')
steps.get_range('A1:H1').merge()
steps.get_range('A1').values = [['Hướng dẫn step-by-step vẽ biểu đồ trong Excel']]
steps.get_range('A1').format = {"fill":"#7F6000","font":{"bold":True,"color":"#FFFFFF","size":15},"horizontal_alignment":"center"}
step_blocks = [
    ['Loại biểu đồ','Step-by-step trên Excel'],
    ['Histogram','1) Chọn cột định lượng hoặc bảng bin. 2) Insert > Statistic Chart > Histogram hoặc Insert > Column. 3) Đặt tiêu đề Histogram_<biến>. 4) Chỉnh trục X là khoảng giá trị, trục Y là tần suất.'],
    ['Boxplot','1) Chọn cột định lượng. 2) Insert > Statistic Chart > Box & Whisker. 3) Đặt tiêu đề Boxplot_<biến>. 4) Bật outlier/mean marker nếu cần.'],
    ['Scatter Plot','1) Chọn 2 cột định lượng X và Y. 2) Insert > Scatter. 3) Đặt tiêu đề X_vs_Y. 4) Có thể thêm Trendline để xem xu hướng.'],
    ['Bar Chart','1) Tạo bảng tần suất bằng PivotTable hoặc COUNTIF. 2) Chọn bảng tần suất. 3) Insert > Bar/Column Chart. 4) Sắp xếp giảm dần để dễ đọc.'],
    ['Line Chart','1) Tạo bảng tổng hợp theo Age/Rating/Age Group. 2) Insert > Line. 3) Dùng khi trục X có thứ tự. 4) Gắn nhãn % nếu là tỷ lệ.'],
    ['Heatmap','1) Tạo ma trận tương quan bằng Data Analysis hoặc hàm CORREL. 2) Chọn ma trận. 3) Home > Conditional Formatting > Color Scales. 4) Màu càng đậm biểu thị tương quan mạnh/yếu.']
]
write_table(steps,2,0,step_blocks)
steps.get_range('A3:B3').format = HEADER_FMT
steps.get_range('A:B').format.column_width = 38
steps.get_range('B:B').format.column_width = 110
steps.get_range('A:B').format.wrap_text = True

# Chart index rows
chart_index_rows = [['STT','Giai đoạn','Nhóm','Tên biểu đồ','Loại Excel','Sheet chứa biểu đồ','Cách vẽ nhanh']]
chart_names_before = []
chart_names_after = []
for stage in ['before','after']:
    suffix = 'trước' if stage=='before' else 'sau'
    for col in num_cols:
        chart_index_rows.append([len(chart_index_rows), suffix, 'Histogram', f'histogram_{col}_{stage}', 'Histogram/Column', f'Charts_{stage}_HistBox', 'Chọn cột '+col+' > Insert > Histogram'])
    for col in num_cols:
        chart_index_rows.append([len(chart_index_rows), suffix, 'Boxplot', f'boxplot_{col}_{stage}', 'Box & Whisker', f'Charts_{stage}_HistBox', 'Chọn cột '+col+' > Insert > Box & Whisker'])
    for col in cat_cols:
        chart_index_rows.append([len(chart_index_rows), suffix, 'Bar Chart', f'bar_{col}_{stage}', 'Column/Bar', f'Charts_{stage}_Bar', 'Tạo bảng tần suất > Insert > Bar/Column'])
    scatters = [('Age','Rating'),('Age','Positive Feedback Count'),('Age','Review Word Count'),('Rating','Positive Feedback Count'),('Rating','Review Word Count'),('Review Text Length','Positive Feedback Count'),('Review Word Count','Positive Feedback Count'),('Review Text Length','Rating'),('Review Word Count','Rating'),('Title Length','Rating')]
    for x,y in scatters:
        chart_index_rows.append([len(chart_index_rows), suffix, 'Scatter Plot', f'scatter_{x}_vs_{y}_{stage}', 'Scatter', f'Charts_{stage}_Scatter', 'Chọn 2 cột X/Y > Insert > Scatter'])
    for name in ['line_review_count_by_age','line_avg_rating_by_age','line_recommended_rate_by_rating','line_avg_feedback_by_rating','line_recommended_rate_by_age_group']:
        chart_index_rows.append([len(chart_index_rows), suffix, 'Line Chart', f'{name}_{stage}', 'Line', f'Charts_{stage}_LineHeat', 'Tạo bảng tổng hợp theo trục X > Insert > Line'])
    chart_index_rows.append([len(chart_index_rows), suffix, 'Heatmap', f'heatmap_correlation_{stage}', 'Conditional Formatting', f'Charts_{stage}_LineHeat', 'Tạo ma trận CORREL > Conditional Formatting > Color Scales'])
    if stage == 'after':
        chart_index_rows.append([len(chart_index_rows), suffix, 'Grouped Bar', 'grouped_bar_rating_by_department_after', 'Clustered Column', 'Charts_after_Bar', 'PivotTable Department x Rating > Insert > Clustered Column'])
        chart_index_rows.append([len(chart_index_rows), suffix, 'Bar Chart', 'bar_top_class_by_recommended_rate_after', 'Column/Bar', 'Charts_after_Bar', 'Tính tỷ lệ Recommended theo Class > Insert > Bar'])
write_table(steps,12,0,chart_index_rows)
steps.get_range('A13:G13').format = HEADER_FMT
steps.get_range('A:G').format.wrap_text = True
steps.get_range('D:D').format.column_width = 42
steps.get_range('G:G').format.column_width = 55

# VBA code sheet (reference code)
vba = wb.worksheets.add('VBA_Code_Optional')
vba_code = ''''Lưu ý: file .xlsx này đã có sẵn biểu đồ. Nếu muốn dùng macro, hãy Save As .xlsm trước.
'Ý tưởng macro: tự tạo lại biểu đồ từ các sheet dữ liệu/tổng hợp.
Sub TaoBieuDoMau()
    MsgBox "Workbook này đã được dựng sẵn 39 biểu đồ trước tiền xử lý và 41 biểu đồ sau tiền xử lý." & vbCrLf & _
           "Muốn tự vẽ thủ công: xem sheet Excel_Steps." & vbCrLf & _
           "Muốn tự động hóa nâng cao: dùng Office Script/VBA dựa trên Chart_Index."
End Sub
'''
vba.get_range('A1').values = [['VBA tham khảo - không bắt buộc chạy']]
vba.get_range('A1').format = HEADER_FMT
# split code lines into rows
write_table(vba,2,0,[[line] for line in vba_code.splitlines()])
vba.get_range('A:A').format.column_width = 120
vba.get_range('A:A').format.wrap_text = True

# Summary sheets with basic descriptive stats
for stage, data in datasets.items():
    sh = wb.worksheets.add(f'Summary_{stage}')
    rows = [['Biến','Count','Missing','Mean','Median','Min','Q1','Q3','Max','Std Dev']]
    for col in num_cols:
        vals = numeric_values(data, col)
        missing = len(data)-len(vals)
        if vals:
            mean = round(sum(vals)/len(vals),3); med = round(quantile(vals,.5),3)
            q1 = round(quantile(vals,.25),3); q3 = round(quantile(vals,.75),3)
            std = round(statistics.pstdev(vals),3) if len(vals)>1 else 0
            rows.append([col,len(vals),missing,mean,med,min(vals),q1,q3,max(vals),std])
        else:
            rows.append([col,0,missing,None,None,None,None,None,None,None])
    write_table(sh,0,0,rows)
    sh.get_range('A1:J1').format = HEADER_FMT
    sh.get_range('A:J').format.column_width = 18

# ---------- Charts ----------
def layout_pos(i):
    per_row = 2
    row = 1 + (i//per_row)*16
    col = (i%per_row)*8
    return row, col

def create_hist_box(stage, data):
    sheet = wb.worksheets.add(f'Charts_{stage}_HistBox')
    sheet.get_range('A1').values = [[f'Histogram + Boxplot ({stage})']]
    sheet.get_range('A1').format = HEADER_FMT
    i = 0
    for col in num_cols:
        vals = numeric_values(data, col)
        cats, counts = hist_bins(vals, 10)
        row, c = layout_pos(i)
        add_chart(sheet,'bar',f'Histogram {col} ({stage})',cats,counts,row=row,col=c,width=430,height=250)
        i += 1
    # Excel file uses a five-number-summary line chart for boxplot-like display,
    # because not every Excel renderer exports native Box & Whisker charts reliably.
    # Manual native Box & Whisker steps are still documented in Excel_Steps.
    for col in num_cols:
        vals = numeric_values(data, col)
        row, c = layout_pos(i)
        qvals = [round(min(vals),3), round(quantile(vals,.25),3), round(quantile(vals,.5),3), round(quantile(vals,.75),3), round(max(vals),3)] if vals else []
        add_chart(sheet,'line',f'Boxplot summary {col} ({stage})',['Min','Q1','Median','Q3','Max'],qvals,row=row,col=c,width=430,height=250)
        i += 1

def create_bar(stage, data):
    sheet = wb.worksheets.add(f'Charts_{stage}_Bar')
    sheet.get_range('A1').values = [[f'Bar charts ({stage})']]
    sheet.get_range('A1').format = HEADER_FMT
    i=0
    for col in cat_cols:
        cats, vals = top_counts(data, col, 12)
        row,c=layout_pos(i)
        add_chart(sheet,'bar',f'Bar {col} ({stage})',cats,vals,row=row,col=c,width=430,height=250)
        i+=1
    if stage=='after':
        # grouped bar rating by department
        depts, ratings, result = grouped_rating_by_department(data)
        row,c=layout_pos(i)
        chart = sheet.charts.add('bar', {'from': {'row': row, 'col': c}, 'extent': {'widthPx': 560, 'heightPx': 280}})
        try: chart.title_text = 'Grouped bar: Rating by Department (after)'
        except Exception: chart.title = 'Grouped bar: Rating by Department (after)'
        try: chart.categories = depts
        except Exception: pass
        for rating, vals in result:
            s=chart.series.add('Rating '+rating)
            try: s.categories = depts
            except Exception: pass
            s.values = vals
        i += 1
        cats, vals = top_class_rec_rate(data)
        row,c=layout_pos(i)
        add_chart(sheet,'bar','Top Class by Recommended Rate (after)',cats,vals,row=row,col=c,width=560,height=280)

def create_scatter(stage, data):
    sheet = wb.worksheets.add(f'Charts_{stage}_Scatter')
    sheet.get_range('A1').values = [[f'Scatter plots ({stage})']]
    sheet.get_range('A1').format = HEADER_FMT
    pairs = [('Age','Rating'),('Age','Positive Feedback Count'),('Age','Review Word Count'),('Rating','Positive Feedback Count'),('Rating','Review Word Count'),('Review Text Length','Positive Feedback Count'),('Review Word Count','Positive Feedback Count'),('Review Text Length','Rating'),('Review Word Count','Rating'),('Title Length','Rating')]
    for i,(x,y) in enumerate(pairs):
        xs, ys = sample_pairs(data,x,y,500)
        row,c = layout_pos(i)
        add_chart(sheet,'scatter',f'{x} vs {y} ({stage})',xvalues=xs,values=ys,row=row,col=c,width=430,height=250)

def create_line_heat(stage, data):
    sheet = wb.worksheets.add(f'Charts_{stage}_LineHeat')
    sheet.get_range('A1').values = [[f'Line charts + Heatmap ({stage})']]
    sheet.get_range('A1').format = HEADER_FMT
    line_specs = []
    cats, vals = group_count_by(data,'Age'); line_specs.append(('Review count by Age', cats, vals))
    cats, vals = group_avg_by(data,'Age','Rating'); line_specs.append(('Average Rating by Age', cats, vals))
    cats, vals = rec_rate_by(data,'Rating'); line_specs.append(('Recommended Rate by Rating (%)', cats, vals))
    cats, vals = avg_feedback_by_rating(data); line_specs.append(('Average Feedback by Rating', cats, vals))
    cats, vals = rec_rate_by(data,'Age Group'); line_specs.append(('Recommended Rate by Age Group (%)', cats, vals))
    for i,(title,cats,vals) in enumerate(line_specs):
        row,c=layout_pos(i)
        add_chart(sheet,'line',f'{title} ({stage})',cats,vals,row=row,col=c,width=430,height=250)
    # Heatmap table below/right
    hm_start_row = 1 + (len(line_specs)//2 + 1)*16
    cols = ['Age','Rating','Recommended IND','Positive Feedback Count','Review Text Length','Review Word Count','Title Length']
    mat = corr_matrix(data, cols)
    hm = [['Correlation Heatmap'] + cols] + [[cols[i]] + mat[i] for i in range(len(cols))]
    write_table(sheet, hm_start_row, 0, hm)
    end_row = hm_start_row + len(hm) - 1
    end_col = len(cols)
    sheet.get_range_by_indexes(hm_start_row,0,1,len(cols)+1).format = HEADER_FMT
    sheet.get_range_by_indexes(hm_start_row+1,0,len(cols),1).format = SUB_FMT
    rng = sheet.get_range_by_indexes(hm_start_row+1,1,len(cols),len(cols))
    try:
        rng.conditional_formats.add_color_scale({"minColor":"#F4CCCC","midColor":"#FFFFFF","maxColor":"#B6D7A8"})
    except Exception:
        pass
    sheet.get_range_by_indexes(hm_start_row,0,len(cols)+1,len(cols)+1).format.column_width = 16

for st, data in datasets.items():
    create_hist_box(st, data)
    create_bar(st, data)
    create_scatter(st, data)
    create_line_heat(st, data)

# Final formatting: try autofit selected sheets
for shname in ['README','Variable_Classification','Excel_Steps','VBA_Code_Optional','Summary_before','Summary_after']:
    try:
        wb.worksheets.get_item(shname).get_range('A:Z').format.autofit_columns()
    except Exception:
        pass

# Inspect formula errors not much formulas, export
SpreadsheetFile.export_xlsx(wb).save(OUT_PATH)
print(OUT_PATH)
