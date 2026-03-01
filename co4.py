import pandas as pd
import numpy as np
from sklearn.linear_model import HuberRegressor
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

# Capping di 98.5% (Terbukti paling optimal di semua skor 6.x)
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
# SMOOTH IBNR (TAMBALAN DATA TERAKHIR)
# ---------------------------------------------------------
max_idx = len(monthly_data) - 1
if monthly_data.loc[max_idx, 'Claim_Frequency'] < 0.8 * monthly_data.loc[max_idx - 1, 'Claim_Frequency']:
    print(f"[IBNR PATCH] Menambal data bulan terakhir yang belum lengkap: {monthly_data.loc[max_idx, 'YearMonth']}")
    monthly_data.loc[max_idx, 'Claim_Frequency'] = monthly_data['Claim_Frequency'].iloc[-4:-1].mean()
    monthly_data.loc[max_idx, 'Total_Claim'] = monthly_data['Total_Claim'].iloc[-4:-1].mean()

# Fokus era pasca-Covid yang stabil
monthly_data = monthly_data[monthly_data['Date'] >= '2022-01-01'].reset_index(drop=True)

# THE ACTUARIAL DECOUPLING: Hitung Severity (Biaya per Orang) secara historis
monthly_data['Claim_Severity'] = monthly_data['Total_Claim'] / monthly_data['Claim_Frequency']

# ---------------------------------------------------------
# TARGET LOG-TRANSFORM (KUNCI STABILITAS AI)
# ---------------------------------------------------------
# Kita prediksi Log Frekuensi dan Log Severity (BUKAN TOTAL CLAIM)
monthly_data['Log_Freq'] = np.log1p(monthly_data['Claim_Frequency'])
monthly_data['Log_Sev'] = np.log1p(monthly_data['Claim_Severity'])

# ==========================================
# 2. CYCLICAL FEATURE ENGINEERING
# ==========================================
monthly_data['Time_Index'] = np.arange(len(monthly_data))
monthly_data['Month'] = monthly_data['Date'].dt.month
monthly_data['sin_M'] = np.sin(2 * np.pi * monthly_data['Month'] / 12)
monthly_data['cos_M'] = np.cos(2 * np.pi * monthly_data['Month'] / 12)

# Wadah masa depan
last_date = monthly_data['Date'].max()
all_future_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), end='2025-12-01', freq='MS')
future_df = pd.DataFrame({'Date': all_future_dates})
future_df['Time_Index'] = np.arange(len(monthly_data), len(monthly_data) + len(future_df))
future_df['Month'] = future_df['Date'].dt.month
future_df['sin_M'] = np.sin(2 * np.pi * future_df['Month'] / 12)
future_df['cos_M'] = np.cos(2 * np.pi * future_df['Month'] / 12)

# ==========================================
# 3. DYNAMIC FORECASTING (HOLT-WINTERS + HUBER)
# ==========================================
print("\nMelatih Model Dynamic Actuarial (Frequency & Severity dipisah)...")

# --- A. PREDIKSI FREKUENSI (Punya Siklus/Musim) ---
features_freq = ['Time_Index', 'sin_M', 'cos_M']
huber_freq = HuberRegressor(epsilon=1.35).fit(monthly_data[features_freq], monthly_data['Log_Freq'])
pred_log_freq_huber = huber_freq.predict(future_df[features_freq])

# Holt-Winters dengan Seasonality
try:
    hw_freq_model = ExponentialSmoothing(monthly_data['Log_Freq'].values, trend='add', seasonal='add', seasonal_periods=12, damped_trend=True)
    hw_freq = hw_freq_model.fit(optimized=True)
    pred_log_freq_hw = hw_freq.forecast(steps=len(future_df))
except Exception as e:
    print(f"[INFO] Data historis < 24 bulan. Mengalihkan Holt-Winters ke mode Non-Seasonal...")
    hw_freq_model = ExponentialSmoothing(monthly_data['Log_Freq'].values, trend='add', damped_trend=True)
    hw_freq = hw_freq_model.fit(optimized=True)
    pred_log_freq_hw = hw_freq.forecast(steps=len(future_df))

# Blend Frekuensi: 60% HW (Musiman Kuat/Tren) + 40% Huber (Tren Stabil)
future_df['Log_Freq'] = (0.6 * pred_log_freq_hw) + (0.4 * pred_log_freq_huber)


# --- B. PREDIKSI SEVERITY (Hanya Inflasi, Tidak Ada Musim) ---
features_sev = ['Time_Index'] # Severity tidak peduli bulan, hanya terus naik karena inflasi
huber_sev = HuberRegressor(epsilon=1.35).fit(monthly_data[features_sev], monthly_data['Log_Sev'])
pred_log_sev_huber = huber_sev.predict(future_df[features_sev])

# Holt-Winters TANPA Seasonality (Hanya Tren Inflasi)
try:
    hw_sev_model = ExponentialSmoothing(monthly_data['Log_Sev'].values, trend='add', seasonal=None, damped_trend=True)
    hw_sev = hw_sev_model.fit(optimized=True)
    pred_log_sev_hw = hw_sev.forecast(steps=len(future_df))
except Exception as e:
    hw_sev_model = ExponentialSmoothing(monthly_data['Log_Sev'].values, trend=None, seasonal=None)
    hw_sev = hw_sev_model.fit(optimized=True)
    pred_log_sev_hw = hw_sev.forecast(steps=len(future_df))

# Blend Severity: 50% HW + 50% Huber
future_df['Log_Sev'] = (0.5 * pred_log_sev_hw) + (0.5 * pred_log_sev_huber)

# ==========================================
# 4. EXPORT & KALKULASI TOTAL KLAIM
# ==========================================
submission_rows = []
print("\n--- HASIL PREDIKSI (DYNAMIC ACTUARIAL DECOUPLER) ---")
print("Total Klaim dihitung matematis dari Frekuensi x Severity hasil prediksi ML.")

target_months = ['2025-08-01', '2025-09-01', '2025-10-01', '2025-11-01', '2025-12-01']
final_output = future_df[future_df['Date'].isin(pd.to_datetime(target_months))].copy()

for _, row in final_output.iterrows():
    month_key = row['Date'].strftime('%Y_%m')
    
    # Inverse Log
    freq = np.expm1(row['Log_Freq'])
    sev = np.expm1(row['Log_Sev'])
    
    # RUMUS MUTLAK AKTUARIA
    total = freq * sev
    
    print(f"{month_key} -> Freq: {freq:.1f} | Sev: {sev:,.0f} | Total: {total:,.0f}")
    
    submission_rows.append({'id': f'{month_key}_Claim_Frequency', 'value': freq})
    submission_rows.append({'id': f'{month_key}_Claim_Severity', 'value': sev})
    submission_rows.append({'id': f'{month_key}_Total_Claim', 'value': total})

submission_df = pd.DataFrame(submission_rows)
submission_df.to_csv('submission_dynamic_actuarial.csv', index=False)
print("\n[FINAL PUSH] File 'submission_dynamic_actuarial.csv' siap!")
print("Ini adalah algoritma yang paling kebal error. BISMILLAH TEMBUS < 3.0!")