import pandas as pd
import numpy as np
from sklearn.linear_model import HuberRegressor
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import lightgbm as lgb
import xgboost as xgb
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

# Capping di 98.5% 
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
# SMOOTH IBNR (TIDAK MEMBUANG BULAN TERAKHIR)
# ---------------------------------------------------------
# Kita tidak drop bulannya, agar AI tetap punya titik referensi waktu yang urut!
# Jika bulan terakhir drop >20%, kita tambal pakai rata-rata 3 bulan sebelumnya.
max_idx = len(monthly_data) - 1
if monthly_data.loc[max_idx, 'Claim_Frequency'] < 0.8 * monthly_data.loc[max_idx - 1, 'Claim_Frequency']:
    print(f"[IBNR PATCH] Menambal data bulan terakhir yang belum lengkap: {monthly_data.loc[max_idx, 'YearMonth']}")
    monthly_data.loc[max_idx, 'Claim_Frequency'] = monthly_data['Claim_Frequency'].iloc[-4:-1].mean()
    monthly_data.loc[max_idx, 'Total_Claim'] = monthly_data['Total_Claim'].iloc[-4:-1].mean()

# Fokus era pasca-Covid yang stabil
monthly_data = monthly_data[monthly_data['Date'] >= '2022-01-01'].reset_index(drop=True)

# ---------------------------------------------------------
# TARGET LOG-TRANSFORM (SUPER STABIL)
# ---------------------------------------------------------
monthly_data['Log_Freq'] = np.log1p(monthly_data['Claim_Frequency'])
monthly_data['Log_Total'] = np.log1p(monthly_data['Total_Claim'])

# ==========================================
# 2. FEATURE ENGINEERING (TREND + SEASONALITY)
# ==========================================
monthly_data['Time_Index'] = np.arange(len(monthly_data))
monthly_data['Month'] = monthly_data['Date'].dt.month
monthly_data['sin_M'] = np.sin(2 * np.pi * monthly_data['Month'] / 12)
monthly_data['cos_M'] = np.cos(2 * np.pi * monthly_data['Month'] / 12)

# Buat wadah masa depan
last_date = monthly_data['Date'].max()
all_future_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), end='2025-12-01', freq='MS')
future_df = pd.DataFrame({'Date': all_future_dates})
future_df['Time_Index'] = np.arange(len(monthly_data), len(monthly_data) + len(future_df))
future_df['Month'] = future_df['Date'].dt.month
future_df['sin_M'] = np.sin(2 * np.pi * future_df['Month'] / 12)
future_df['cos_M'] = np.cos(2 * np.pi * future_df['Month'] / 12)

# ==========================================
# 3. RESIDUAL DETRENDING (THE GRANDMASTER HACK)
# ==========================================
print("\nMelatih Model dengan Teknik Residual Detrending...")

for target in ['Log_Freq', 'Log_Total']:
    y = monthly_data[target]
    
    # LANGKAH 1: Prediksi Tren Masa Depan dengan Huber Regressor
    trend_model = HuberRegressor(epsilon=1.35)
    trend_model.fit(monthly_data[['Time_Index']], y)
    
    current_trend = trend_model.predict(monthly_data[['Time_Index']])
    future_trend = trend_model.predict(future_df[['Time_Index']])
    
    # LANGKAH 2: Cari Sisa Pola (Residuals)
    residuals = y - current_trend
    
    # LANGKAH 3: Biarkan LightGBM & XGBoost menghafal Pola Musimannya!
    season_features = ['Month', 'sin_M', 'cos_M']
    
    lgb_model = lgb.LGBMRegressor(max_depth=3, learning_rate=0.05, n_estimators=100, random_state=42, verbose=-1)
    lgb_model.fit(monthly_data[season_features], residuals)
    
    xgb_model = xgb.XGBRegressor(max_depth=3, learning_rate=0.05, n_estimators=100, random_state=42)
    xgb_model.fit(monthly_data[season_features], residuals)
    
    future_residuals = (0.5 * lgb_model.predict(future_df[season_features])) + (0.5 * xgb_model.predict(future_df[season_features]))
    
    # HASIL 1 = TREN + POLA MUSIMAN
    pred_detrend = future_trend + future_residuals
    
    # LANGKAH 4: Holt-Winters (Jaring Pengaman Aktuaria Klasik)
    try:
        hw_model = ExponentialSmoothing(y.values, trend='add', seasonal='add', seasonal_periods=12, damped_trend=True)
        hw_fit = hw_model.fit(optimized=True)
        hw_preds = hw_fit.forecast(steps=len(future_df))
    except:
        hw_model = ExponentialSmoothing(y.values, trend='add', damped_trend=True)
        hw_fit = hw_model.fit(optimized=True)
        hw_preds = hw_fit.forecast(steps=len(future_df))
        
    # BLEND FINAL: 60% Detrending AI + 40% Holt-Winters
    future_df[target] = (0.6 * pred_detrend) + (0.4 * hw_preds)

# ==========================================
# 4. EXPORT HASIL
# ==========================================
submission_rows = []
print("\n--- HASIL PREDIKSI (RESIDUAL DETRENDING) ---")
print("Severity TIDAK lagi dikunci. Total Klaim dan Frekuensi berjalan secara harmoni natural.")

target_months = ['2025-08-01', '2025-09-01', '2025-10-01', '2025-11-01', '2025-12-01']
final_output = future_df[future_df['Date'].isin(pd.to_datetime(target_months))].copy()

for _, row in final_output.iterrows():
    month_key = row['Date'].strftime('%Y_%m')
    
    freq = np.expm1(row['Log_Freq'])
    total = np.expm1(row['Log_Total'])
    
    # Severity murni sebagai hasil bagi, tanpa ada unsur tebakan/paksaan
    sev = total / freq if freq > 0 else 0
    
    print(f"{month_key} -> Freq: {freq:.1f} | Sev: {sev:,.0f} | Total: {total:,.0f}")
    
    submission_rows.append({'id': f'{month_key}_Claim_Frequency', 'value': freq})
    submission_rows.append({'id': f'{month_key}_Claim_Severity', 'value': sev})
    submission_rows.append({'id': f'{month_key}_Total_Claim', 'value': total})

submission_df = pd.DataFrame(submission_rows)
submission_df.to_csv('submission_residual_detrending.csv', index=False)
print("\n[PERFECT RUN] File 'submission_residual_detrending.csv' siap!")
print("Ini adalah batas maksimal dari Data Science untuk data kecil. BISMILLAH < 3!")