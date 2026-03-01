import pandas as pd
import numpy as np
from sklearn.linear_model import HuberRegressor, BayesianRidge
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

# KEMBALI KE CAPPING 98.0% (Pahlawan Skor 4.8)
limit = df_klaim_paid['Nominal Klaim Yang Disetujui'].quantile(0.980)
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

# THE ACTUARIAL DECOUPLING
monthly_data['Claim_Severity'] = monthly_data['Total_Claim'] / monthly_data['Claim_Frequency']

# ---------------------------------------------------------
# TARGET TRANSFORMATION (KUNCI ANTI-LEDAKAN)
# ---------------------------------------------------------
# Frekuensi tetap di-LOG karena bentuknya siklus/gelombang
monthly_data['Log_Freq'] = np.log1p(monthly_data['Claim_Frequency'])
# SEVERITY TIDAK DI-LOG! Agar tebakan inflasi masa depannya berupa garis lurus, bukan eksponensial!

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
# 3. GRANDMASTER FORECASTING (BAYESIAN + HW)
# ==========================================
print("\nMelatih Model Grandmaster Stabilizer (Linear Severity)...")

# --- A. PREDIKSI FREKUENSI (Menggunakan Log & Siklus Musiman) ---
features_freq = ['Time_Index', 'sin_M', 'cos_M']

# Panggil kembali Bayesian Ridge (Raja stabilitas micro-data)
bayes_freq = BayesianRidge().fit(monthly_data[features_freq], monthly_data['Log_Freq'])
pred_log_freq_bayes = bayes_freq.predict(future_df[features_freq])

huber_freq = HuberRegressor(epsilon=1.35).fit(monthly_data[features_freq], monthly_data['Log_Freq'])
pred_log_freq_huber = huber_freq.predict(future_df[features_freq])

# Holt-Winters dengan Seasonality
try:
    hw_freq_model = ExponentialSmoothing(monthly_data['Log_Freq'].values, trend='add', seasonal='add', seasonal_periods=12, damped_trend=True)
    hw_freq = hw_freq_model.fit(optimized=True)
    pred_log_freq_hw = hw_freq.forecast(steps=len(future_df))
except Exception as e:
    hw_freq_model = ExponentialSmoothing(monthly_data['Log_Freq'].values, trend='add', damped_trend=True)
    hw_freq = hw_freq_model.fit(optimized=True)
    pred_log_freq_hw = hw_freq.forecast(steps=len(future_df))

# Blend Frekuensi: 50% HW + 30% Bayes + 20% Huber
future_df['Log_Freq'] = (0.50 * pred_log_freq_hw) + (0.30 * pred_log_freq_bayes) + (0.20 * pred_log_freq_huber)


# --- B. PREDIKSI SEVERITY (Nilai Nyata, Hanya Tren Inflasi) ---
features_sev = ['Time_Index'] 

bayes_sev = BayesianRidge().fit(monthly_data[features_sev], monthly_data['Claim_Severity'])
pred_sev_bayes = bayes_sev.predict(future_df[features_sev])

# Holt-Winters TANPA Seasonality
try:
    hw_sev_model = ExponentialSmoothing(monthly_data['Claim_Severity'].values, trend='add', seasonal=None, damped_trend=True)
    hw_sev = hw_sev_model.fit(optimized=True)
    pred_sev_hw = hw_sev.forecast(steps=len(future_df))
except Exception as e:
    hw_sev_model = ExponentialSmoothing(monthly_data['Claim_Severity'].values, trend=None, seasonal=None)
    hw_sev = hw_sev_model.fit(optimized=True)
    pred_sev_hw = hw_sev.forecast(steps=len(future_df))

# Blend Severity: 50% HW + 50% Bayes (Kenaikan inflasi berupa garis lurus sempurna)
future_df['Final_Severity'] = (0.50 * pred_sev_hw) + (0.50 * pred_sev_bayes)

# ==========================================
# 4. EXPORT & KALKULASI TOTAL KLAIM
# ==========================================
submission_rows = []
print("\n--- HASIL PREDIKSI (GRANDMASTER STABILIZER) ---")
print("Severity diprediksi secara linear (Garis Lurus) agar tidak meledak di bulan Desember.")

target_months = ['2025-08-01', '2025-09-01', '2025-10-01', '2025-11-01', '2025-12-01']
final_output = future_df[future_df['Date'].isin(pd.to_datetime(target_months))].copy()

for _, row in final_output.iterrows():
    month_key = row['Date'].strftime('%Y_%m')
    
    # Hanya Frekuensi yang di-Inverse Log. Severity sudah nilai asli!
    freq = np.expm1(row['Log_Freq'])
    sev = row['Final_Severity']
    
    # RUMUS MUTLAK AKTUARIA
    total = freq * sev
    
    print(f"{month_key} -> Freq: {freq:.1f} | Sev: {sev:,.0f} | Total: {total:,.0f}")
    
    submission_rows.append({'id': f'{month_key}_Claim_Frequency', 'value': freq})
    submission_rows.append({'id': f'{month_key}_Claim_Severity', 'value': sev})
    submission_rows.append({'id': f'{month_key}_Total_Claim', 'value': total})

submission_df = pd.DataFrame(submission_rows)
submission_df.to_csv('submission_grandmaster_stabilizer.csv', index=False)
print("\n[PERFECT RUN] File 'submission_grandmaster_stabilizer.csv' siap!")
print("Ini adalah model paling matematis dan stabil sejauh ini. BISMILLAH < 3.0!")