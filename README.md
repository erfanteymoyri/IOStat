# IOStat — پیش‌بینی بهره‌وری واقعی دستگاه‌های ذخیره‌سازی با یادگیری ماشین

<p align="center">
  <strong>Python · scikit-learn · matplotlib</strong>
</p>

---

## فهرست مطالب

- [درباره پروژه](#درباره-پروژه)
- [مشکلی که حل می‌کنیم](#مشکلی-که-حل-میکنیم)
- [ساختار پروژه](#ساختار-پروژه)
- [پیش‌نیازها](#پیشنیازها)
- [نصب و راه‌اندازی](#نصب-و-راهاندازی)
- [مدل‌های یادگیری ماشین](#مدلهای-یادگیری-ماشین)
- [نحوه استفاده](#نحوه-استفاده)
  - [۱. آموزش مدل‌ها](#۱-آموزش-مدلها)
  - [۲. ارزیابی مدل‌ها](#۲-ارزیابی-مدلها)
  - [۳. نمودارهای انگیزشی](#۳-نمودارهای-انگیزشی)
- [تنظیمات پروژه](#تنظیمات-پروژه)
- [ویژگی‌های ورودی مدل](#ویژگیهای-ورودی-مدل)
- [خروجی‌ها](#خروجیها)
- [وابستگی‌ها](#وابستگیها)

---

## درباره پروژه

**IOStat** یک پروژه یادگیری ماشین مبتنی بر پایتون است که برای **پیش‌بینی حداکثر IOPS دستگاه‌های ذخیره‌سازی** و محاسبه دقیق **بهره‌وری واقعی (Utilization)** طراحی شده است.

این پروژه از **۴ مدل رگرسیون** برای پیش‌بینی حداکثر IOPS قابل دستیابی یک دستگاه با توجه به پارامترهای بار کاری استفاده می‌کند. بعد از پیش‌بینی حداکثر IOPS، بهره‌وری واقعی به‌سادگی محاسبه می‌شود:

```
بهره‌وری واقعی = (IOPS فعلی / حداکثر IOPS پیش‌بینی‌شده) × ۱۰۰
```

---

## مشکلی که حل می‌کنیم

ابزار `iostat` در لینوکس، بهره‌وری دستگاه را بر اساس **یک درخواست I/O همزمان** محاسبه می‌کند. این روش زمانی که چندین درخواست I/O به‌صورت همزمان اجرا می‌شوند، **عدد اشتباه و خیلی کمتر از واقعیت** گزارش می‌دهد.

| سناریو | IOPS | بهره‌وری واقعی | گزارش iostat |
|--------|------|----------------|--------------|
| تک درخواست — ۵ IOPS | ۵ | ۹٪ | ۹٪ ✅ |
| تک درخواست — ۲۰ IOPS | ۲۰ | ۳۵٪ | ۳۰٪ ≈ |
| چند درخواست — ۱۰۰ IOPS | ۱۰۰ | ۱۸٪ | ۸۰٪ ❌ |
| چند درخواست — ۲۰۰ IOPS | ۲۰۰ | ۳۵٪ | ۹۳٪ ❌ |

**راه‌حل ما:** با استفاده از مدل‌های یادگیری ماشین، حداکثر IOPS دستگاه را پیش‌بینی کرده و بهره‌وری واقعی را دقیق محاسبه می‌کنیم.

---

## ساختار پروژه

```
IOStat/
├── config.py                    # تنظیمات مرکزی (هایپرپارامترها، رنگ‌ها، فونت‌ها)
├── utils.py                     # توابع مشترک (بارگذاری داده، مهندسی ویژگی، ساخت مدل)
├── requirements.txt             # وابستگی‌های پایتون
├── README.md                    # این فایل
│
└── scripts/
    ├── motivational/            # اسکریپت‌های نمودار انگیزشی
    │   ├── 01_summarized_bar_chart.py
    │   ├── 02_motivational_curves.py
    │   ├── 03_max_iops_per_device.py
    │   └── 04_iops_and_bandwidth.py
    │
    ├── training/                # اسکریپت‌های آموزش مدل‌ها
    │   ├── 05_deterministic_training.py
    │   ├── 06_blocksize_generalization.py
    │   ├── 07_smart_hybrid_model.py
    │   ├── 08_random_split_analysis.py
    │   ├── 09_hybrid_regime_random.py
    │   ├── 10_iops_bound_focused.py
    │   ├── 11_deterministic_mentor_viz.py
    │   └── 12_mentor_generalization_viz.py
    │
    └── evaluation/              # اسکریپت‌های ارزیابی و مقایسه مدل‌ها
        ├── 13_execute_training.py
        ├── 14_evaluation_line_plot.py
        ├── 15_evaluation_smooth_curves.py
        ├── 16_evaluation_time_series.py
        ├── 17_benchmark_evaluation.py
        ├── 18_evaluation_interpolation.py
        ├── 19_evaluation_max_error.py
        ├── 20–30_eval_workload_*.py     # ارزیابی ورک‌لودهای مختلف
        └── evaluation_utils.py          # توابع مشترک ارزیابی
```

---

## پیش‌نیازها

- **Python 3.8** یا بالاتر
- **pip** (مدیر بسته پایتون)

---

## نصب و راه‌اندازی

### ۱. کلون کردن مخزن

```bash
git clone https://github.com/erfanteymoyri/IOStat.git
cd IOStat
```

### ۲. ساخت محیط مجازی (پیشنهادی)

```bash
python -m venv venv
source venv/bin/activate        # لینوکس / مک
# venv\Scripts\activate         # ویندوز
```

### ۳. نصب وابستگی‌ها

```bash
pip install -r requirements.txt
```

همین! پروژه آماده استفاده است. نیازی به build یا کامپایل نیست.

---

## مدل‌های یادگیری ماشین

این پروژه از **۴ مدل رگرسیون** استفاده می‌کند که هر کدام یک پایپ‌لاین scikit-learn هستند:

| مدل | توضیح | کاربرد |
|-----|--------|--------|
| **LINEAR** | رگرسیون خطی ساده با نرمال‌سازی | مدل پایه برای مقایسه |
| **LASSO_POLY** | رگرسیون Lasso با ویژگی‌های چندجمله‌ای (درجه ۳) | گرفتن روابط غیرخطی |
| **RANDOM_FOREST** | جنگل تصادفی (۱۰۰ درخت) | عملکرد قوی روی داده‌های متنوع |
| **SVR** | رگرسیون بردار پشتیبان (کرنل RBF) | بهترین عملکرد برای بلاک‌سایزهای کوچک |

علاوه بر این‌ها، مدل **SMART_HYBRID** هم وجود دارد که بر اساس اندازه بلاک، بین مدل‌ها مسیریابی می‌کند:
- بلاک‌سایز ≤ ۸ کیلوبایت → **SVR**
- بلاک‌سایز ۸ تا ۱۶ کیلوبایت → **Random Forest**
- بلاک‌سایز > ۱۶ کیلوبایت → **Random Forest**

---

## نحوه استفاده

> **نکته:** تمام اسکریپت‌ها باید از **دایرکتوری ریشه پروژه** اجرا شوند تا import مشترک `config.py` و `utils.py` به‌درستی کار کند.

### ۱. آموزش مدل‌ها

#### آموزش و ذخیره مدل‌ها

اولین قدم، آموزش مدل‌ها روی داده‌های آموزشی و ذخیره آن‌هاست:

```bash
python scripts/evaluation/13_execute_training.py data/AllDevices.xlsx --save models.joblib
```

این اسکریپت:
- داده‌های آموزشی را بارگذاری و پیش‌پردازش می‌کند
- برای هر دستگاه ذخیره‌سازی، ۴ مدل آموزش می‌دهد
- مدل‌ها را در فایل `models.joblib` ذخیره می‌کند

#### آموزش قطعی (Deterministic)

آموزش روی یک مجموعه داده و تست روی مجموعه دیگر:

```bash
python scripts/training/05_deterministic_training.py data/AllDevices.xlsx data/limit100.xlsx
```

خروجی شامل نمودارهای میله‌ای مقایسه IOPS واقعی با پیش‌بینی‌شده و فایل اکسل نتایج است.

#### تعمیم‌پذیری بلاک‌سایز

آموزش بر اساس رژیم بلاک‌سایز (کوچک و بزرگ) و تست روی بلاک‌سایزهای غیراستاندارد:

```bash
python scripts/training/06_blocksize_generalization.py data/AllDevices.xlsx
```

#### مدل هیبرید هوشمند

آموزش مدل ترکیبی که بر اساس بلاک‌سایز بین SVR و Random Forest مسیریابی می‌کند:

```bash
python scripts/training/07_smart_hybrid_model.py data/AllDevices.xlsx
```

#### تحلیل تقسیم تصادفی

تحلیل عملکرد مدل‌ها با تقسیم تصادفی داده‌ها:

```bash
python scripts/training/08_random_split_analysis.py data/AllDevices.xlsx
```

#### سایر اسکریپت‌های آموزش

| اسکریپت | توضیح |
|---------|--------|
| `09_hybrid_regime_random.py` | مدل هیبرید رژیمی با تقسیم تصادفی |
| `10_iops_bound_focused.py` | آموزش متمرکز بر محدوده IOPS |
| `11_deterministic_mentor_viz.py` | ویژوالایز سبک منتور (فونت بزرگ) |
| `12_mentor_generalization_viz.py` | تحلیل تعمیم‌پذیری سبک منتور |

---

### ۲. ارزیابی مدل‌ها

پس از آموزش مدل‌ها، می‌توانید با اسکریپت‌های ارزیابی، عملکرد آن‌ها را بررسی کنید:

#### نمودار خطی ساده

مقایسه سریع بهره‌وری IOstat در مقابل مدل ما:

```bash
python scripts/evaluation/14_evaluation_line_plot.py data/evaluation.xlsx
```

#### نمودار منحنی هموار

منحنی‌های هموارشده با فلش‌های خطا:

```bash
python scripts/evaluation/15_evaluation_smooth_curves.py data/evaluation.xlsx
```

#### نمودار سری زمانی

مقایسه بهره‌وری در طول زمان (IOstat در مقابل واقعی در مقابل پیش‌بینی ما):

```bash
python scripts/evaluation/16_evaluation_time_series.py data/evaluation.xlsx
```

#### ارزیابی بنچمارک

```bash
python scripts/evaluation/17_benchmark_evaluation.py
```

#### ارزیابی درون‌یابی

تست پیش‌بینی برای بلاک‌سایزهای غیراستاندارد با درون‌یابی خطی:

```bash
python scripts/evaluation/18_evaluation_interpolation.py data/evaluation.xlsx
```

#### تحلیل حداکثر خطا

پیدا کردن نقاط داده با بیشترین خطای پیش‌بینی:

```bash
python scripts/evaluation/19_evaluation_max_error.py data/evaluation.xlsx
```

#### ارزیابی ورک‌لودهای خاص

هر ورک‌لود دو اسکریپت دارد: `basic` (ارزیابی پایه) و `comparison` (مقایسه مدل‌ها):

| اسکریپت | ورک‌لود |
|---------|---------|
| `20/21_eval_workload_42_*.py` | Workload 42 |
| `22/23_eval_fiumail_qd1_*.py` | FIUMail QD1 |
| `24/25_eval_fiumail_qd2_*.py` | FIUMail QD2 |
| `26/27_eval_workload_81_*.py` | Workload 81 |
| `28/29_eval_workload_669_*.py` | Workload 669 |
| `30_eval_workload_17_basic.py` | Workload 17 |

---

### ۳. نمودارهای انگیزشی

این اسکریپت‌ها نمودارهایی تولید می‌کنند که **مشکل iostat** را به‌صورت بصری نشان می‌دهند:

```bash
# نمودار میله‌ای: بهره‌وری واقعی در مقابل گزارش iostat
python scripts/motivational/01_summarized_bar_chart.py

# منحنی‌های انگیزشی
python scripts/motivational/02_motivational_curves.py

# حداکثر IOPS هر دستگاه
python scripts/motivational/03_max_iops_per_device.py

# تحلیل IOPS و پهنای باند
python scripts/motivational/04_iops_and_bandwidth.py
```

---

## تنظیمات پروژه

تمام تنظیمات مرکزی در فایل `config.py` قرار دارند:

### هایپرپارامترها

```python
HYPERPARAMS = {
    "lasso_degree": 3,          # درجه چندجمله‌ای Lasso
    "lasso_alpha": 0.1,         # ضریب نظم‌دهی Lasso
    "rf_n_estimators": 100,     # تعداد درخت‌های جنگل تصادفی
    "rf_max_depth": None,       # حداکثر عمق درخت (بدون محدودیت)
    "rf_min_samples_leaf": 1,   # حداقل نمونه در هر برگ
}
```

### بلاک‌سایزهای استاندارد و لنگرگاه‌ها

```python
STANDARD_BLOCK_SIZES = [2, 4, 8, 16, 32, 64, 128]              # کیلوبایت
ANCHOR_BLOCK_SIZES = [4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
```

### رنگ‌بندی مدل‌ها

هر مدل رنگ و الگوی مشخصی در نمودارها دارد:

| مدل | رنگ | الگو |
|-----|------|------|
| Actual | بنفش `#d896ff` | `///` |
| LINEAR | نارنجی `#ffcb85` | `xxx` |
| LASSO_POLY | سبز `#c2ffb4` | `+++` |
| RANDOM_FOREST | آبی `#99ccff` | `\\` |
| SVR | صورتی `#ffd4e5` | `---` |
| SMART_HYBRID | طلایی `#ffd700` | `**` |

---

## ویژگی‌های ورودی مدل

مدل‌ها فقط از **۲ ویژگی** استفاده می‌کنند:

| ویژگی | توضیح | محدوده |
|-------|--------|--------|
| `read_percent` | درصد عملیات خواندن | ۰ تا ۱۰۰ |
| `block_size_kb` | اندازه بلاک به کیلوبایت | ۲ تا ۴۰۹۶ |

### درون‌یابی بلاک‌سایز

برای بلاک‌سایزهایی که مستقیماً در داده‌های آموزشی نیستند (مثلاً ۱۲ کیلوبایت)، سیستم به‌صورت خودکار **درون‌یابی خطی** بین نزدیک‌ترین بلاک‌سایزهای لنگرگاه انجام می‌دهد.

---

## خروجی‌ها

اسکریپت‌ها خروجی‌های زیر را تولید می‌کنند:

| نوع | فرمت | توضیح |
|-----|-------|--------|
| نمودارها | PDF | نمودارهای مقایسه‌ای و ارزیابی |
| نتایج | Excel (`.xlsx`) | جداول عددی نتایج پیش‌بینی |
| مدل‌ها | joblib (`.joblib`) | مدل‌های آموزش‌دیده ذخیره‌شده |

---

## وابستگی‌ها

| بسته | نسخه | کاربرد |
|------|-------|--------|
| numpy | ≥ 1.21.0 | محاسبات عددی |
| pandas | ≥ 1.3.0 | مدیریت داده‌ها |
| matplotlib | ≥ 3.4.0 | رسم نمودار |
| seaborn | ≥ 0.11.0 | نمودارهای آماری |
| scipy | ≥ 1.7.0 | درون‌یابی منحنی |
| scikit-learn | ≥ 1.0.0 | مدل‌های یادگیری ماشین |
| joblib | ≥ 1.1.0 | ذخیره و بارگذاری مدل‌ها |
| openpyxl | ≥ 3.0.0 | خواندن و نوشتن فایل اکسل |

---

## توابع کلیدی

### بارگذاری و آماده‌سازی داده (`utils.py`)

```python
from utils import load_data, read_and_prep_data, engineer_features

# بارگذاری ساده فایل اکسل یا CSV
df = load_data("data/AllDevices.xlsx")

# بارگذاری و پیش‌پردازش کامل برای آموزش
train_df = read_and_prep_data("data/AllDevices.xlsx", is_test_file=False)
train_df = engineer_features(train_df)

# بارگذاری و پیش‌پردازش کامل برای تست
test_df = read_and_prep_data("data/limit100.xlsx", is_test_file=True)
test_df = engineer_features(test_df)
```

### ساخت و آموزش مدل‌ها

```python
from utils import get_base_pipelines
from config import FEATURES

# ساخت ۴ پایپ‌لاین مدل
pipelines = get_base_pipelines()

# آموزش هر مدل روی داده‌های یک دستگاه
for name, pipeline in pipelines.items():
    pipeline.fit(X_train[FEATURES], y_train)
    predictions = pipeline.predict(X_test[FEATURES])
    print(f"{name}: {predictions[:5]}")
```

### پیش‌بینی با درون‌یابی

```python
from utils import predict_with_interpolation
import numpy as np

# پیش‌بینی برای بلاک‌سایزهای غیراستاندارد
X_new = np.array([[50, 12], [100, 24]])  # [read_percent, block_size_kb]
predictions = predict_with_interpolation(trained_model, X_new)
```

---

## مجوز

این پروژه برای اهداف تحقیقاتی و آکادمیک توسعه داده شده است.