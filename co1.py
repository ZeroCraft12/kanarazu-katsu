import pandas as pd
import numpy as np
from sklearn.linear_model import BayesianRidge, HuberRegressor
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import warnings
warnings.filterwarnings('ignore')

print("Membaca Data Klaim...")
df_klaim = pd.read_csv('Data_Klaim.csv')

# ==========================================
# 1. PONDASI DATA & PEMOTONGAN OUTLIER
# ==========================================
date_col = 'Tanggal Pasien Masuk RS'
df_klaim[date_col] = pd.to_datetime(df_klaim[date_col], errors='coerce')
df_klaim = df_klaim.dropna(subset=[date_col])

df_klaim_paid = df_klaim[df_klaim['Status Klaim'] == 'PAID'].copy()

# Capping di 98.5% (Sweet-spot Golden Bullet)
limit = df_klaim_paid['Nominal Klaim Yang Disetujui'].quantile(0.985)
df_klaim_paid['Nominal_Klaim_Capped'] = np.clip(df_klaim_paid['Nominal Klaim Yang Disetujui'], a_min=0, a_max=limit)

df_klaim_paid['YearMonth'] = df_klaim_paid[date_col].dt.to_period('M').astype(str)

monthly_data = df_klaim_paid.groupby('YearMonth').agg(
    Claim_Frequency=('Claim ID', 'count'),
    Total_Claim=('Nominal_Klaim_Capped', 'sum') 
).reset_index()

monthly_data['Date'] = pd.to_datetime(monthly_data['YearMonth'] + '-01')
monthly_data = monthly_data.sort_values('Date').reset_index(drop=True)

# ---------------------------------------------------------
# THE HARD IBNR DROP (KUNCI KEMURNIAN DATA)
# ---------------------------------------------------------
# Alih-alih menebak pengali 1.4x, kita buang bulan-bulan terakhir yang belum direkap sempurna oleh RS!
while len(monthly_data) > 2 and monthly_data.iloc[-1]['Claim_Frequency'] < 0.8 * monthly_data.iloc[-2]['Claim_Frequency']:
    print(f"[IBNR DETECTED] Membuang bulan yang belum lengkap: {monthly_data.iloc[-1]['YearMonth']}")
    monthly_data = monthly_data.iloc[:-1]

# Gunakan era pasca-Covid yang stabil
monthly_data = monthly_data[monthly_data['Date'] >= '2022-01-01'].reset_index(drop=True)

# ---------------------------------------------------------
# TARGET LOG-TRANSFORM (KUNCI STABILITAS AI)
# ---------------------------------------------------------
monthly_data['Log_Freq'] = np.log1p(monthly_data['Claim_Frequency'])
monthly_data['Log_Total'] = np.log1p(monthly_data['Total_Claim'])

# ==========================================
# 2. CYCLICAL FEATURE ENGINEERING (ANTI-LAG ERROR)
# ==========================================
# Fitur ini mengubah urutan waktu menjadi Garis Tren dan Gelombang Musiman
monthly_data['Time_Index'] = np.arange(len(monthly_data))
monthly_data['Month'] = monthly_data['Date'].dt.month

# Transformasi Sinus & Cosinus (Kaggle Grandmaster Trick untuk Seasonality)
monthly_data['sin_M'] = np.sin(2 * np.pi * monthly_data['Month'] / 12)
monthly_data['cos_M'] = np.cos(2 * np.pi * monthly_data['Month'] / 12)

features = ['Time_Index', 'sin_M', 'cos_M']

# ==========================================
# 3. DIRECT FUTURE MAPPING
# ==========================================
# Buat kerangka waktu untuk prediksi dari bulan setelah data terakhir sampai Desember 2025
last_date = monthly_data['Date'].max()
all_future_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), end='2025-12-01', freq='MS')

all_future_df = pd.DataFrame({'Date': all_future_dates})
all_future_df['Time_Index'] = np.arange(len(monthly_data), len(monthly_data) + len(all_future_df))
all_future_df['Month'] = all_future_df['Date'].dt.month
all_future_df['sin_M'] = np.sin(2 * np.pi * all_future_df['Month'] / 12)
all_future_df['cos_M'] = np.cos(2 * np.pi * all_future_df['Month'] / 12)

# ==========================================
# 4. EXTRAPOLATION ENSEMBLE TRAINING
# ==========================================
print("\nMelatih Model Extrapolasi Murni (Bayesian + Huber + Holt-Winters)...")

for target in ['Log_Freq', 'Log_Total']:
    X_train = monthly_data[features]
    y_train = monthly_data[target]
    X_test = all_future_df[features]

    # Model 1: Bayesian Ridge (Menarik garis tren yang sangat logis ke masa depan)
    bayes = BayesianRidge()
    bayes.fit(X_train, y_train)

    # Model 2: Huber Regressor (Sangat kebal terhadap data bulan anomali)
    huber = HuberRegressor(epsilon=1.35)
    huber.fit(X_train, y_train)

    # Model 3: Holt-Winters (Raja Time-Series, langsung melihat pola asli tanpa fitur tambahan)
    try:
        hw_model = ExponentialSmoothing(monthly_data[target].values, trend='add', seasonal='add', seasonal_periods=12, damped_trend=True)
        hw_fit = hw_model.fit(optimized=True)
        hw_preds = hw_fit.forecast(steps=len(all_future_df))
    except:
        hw_model = ExponentialSmoothing(monthly_data[target].values, trend='add', damped_trend=True)
        hw_fit = hw_model.fit(optimized=True)
        hw_preds = hw_fit.forecast(steps=len(all_future_df))

    # ENSEMBLE BLEND: 40% HW (Musiman Kuat) + 30% Bayes (Tren Halus) + 30% Huber (Anti-Outlier)
    pred_log_future = (0.4 * hw_preds) + (0.3 * bayes.predict(X_test)) + (0.3 * huber.predict(X_test))
    
    all_future_df[target] = pred_log_future

# ==========================================
# 5. FILTER TARGET MONTHS & EXPORT
# ==========================================
submission_rows = []
print("\n--- HASIL PREDIKSI (PLATINUM EXTRAPOLATOR) ---")

# Ambil hanya 5 bulan target yang diminta kompetisi
target_months = ['2025-08-01', '2025-09-01', '2025-10-01', '2025-11-01', '2025-12-01']
final_output = all_future_df[all_future_df['Date'].isin(pd.to_datetime(target_months))].copy()

for _, row in final_output.iterrows():
    month_key = row['Date'].strftime('%Y_%m')
    
    # Kembalikan skala Logaritma ke asli (Miliaran)
    freq = np.expm1(row['Log_Freq'])
    total = np.expm1(row['Log_Total'])
    
    # Hitung Severity sebagai turunan agar tidak ada halusinasi
    sev = total / freq if freq > 0 else 0
    
    print(f"{month_key} -> Freq: {freq:.1f} | Sev: {sev:,.0f} | Total: {total:,.0f}")
    
    submission_rows.append({'id': f'{month_key}_Claim_Frequency', 'value': freq})
    submission_rows.append({'id': f'{month_key}_Claim_Severity', 'value': sev})
    submission_rows.append({'id': f'{month_key}_Total_Claim', 'value': total})

submission_df = pd.DataFrame(submission_rows)
submission_df.to_csv('submission_platinum_extrapolator.csv', index=False)
print("\n[LOCKED] File 'submission_platinum_extrapolator.csv' siap!")
print("Ini adalah batas maksimal dari kemurnian matematika statistik tanpa cacat Machine Learning.")