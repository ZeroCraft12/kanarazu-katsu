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

# Capping di 98.5% (Sweet-spot anti-outlier)
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
# Buang bulan terakhir jika datanya belum direkap sempurna oleh RS
while len(monthly_data) > 2 and monthly_data.iloc[-1]['Claim_Frequency'] < 0.8 * monthly_data.iloc[-2]['Claim_Frequency']:
    print(f"[IBNR DETECTED] Membuang bulan yang belum lengkap: {monthly_data.iloc[-1]['YearMonth']}")
    monthly_data = monthly_data.iloc[:-1]

# Fokus era pasca-Covid
monthly_data = monthly_data[monthly_data['Date'] >= '2022-01-01'].reset_index(drop=True)

# Hitung Severity aktual historis
monthly_data['Claim_Severity'] = monthly_data['Total_Claim'] / monthly_data['Claim_Frequency']

# KUNCI STABILITAS BIAYA: Kunci Median dari 9 bulan terakhir (Anti-halusinasi)
base_severity = monthly_data['Claim_Severity'].tail(9).median()
print(f"\n[ANCHOR] Base Severity (Biaya Per Orang) dikunci di: Rp {base_severity:,.2f}")

# Target untuk Machine Learning HANYA Frekuensi!
monthly_data['Log_Freq'] = np.log1p(monthly_data['Claim_Frequency'])

# ==========================================
# 2. CYCLICAL FEATURE ENGINEERING
# ==========================================
monthly_data['Time_Index'] = np.arange(len(monthly_data))
monthly_data['Month'] = monthly_data['Date'].dt.month
monthly_data['sin_M'] = np.sin(2 * np.pi * monthly_data['Month'] / 12)
monthly_data['cos_M'] = np.cos(2 * np.pi * monthly_data['Month'] / 12)
features = ['Time_Index', 'sin_M', 'cos_M']

# ==========================================
# 3. DIRECT FUTURE MAPPING
# ==========================================
last_date = monthly_data['Date'].max()
all_future_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), end='2025-12-01', freq='MS')

all_future_df = pd.DataFrame({'Date': all_future_dates})
all_future_df['Time_Index'] = np.arange(len(monthly_data), len(monthly_data) + len(all_future_df))
all_future_df['Month'] = all_future_df['Date'].dt.month
all_future_df['sin_M'] = np.sin(2 * np.pi * all_future_df['Month'] / 12)
all_future_df['cos_M'] = np.cos(2 * np.pi * all_future_df['Month'] / 12)

# ==========================================
# 4. FREQUENCY ENSEMBLE EXTRAPOLATION
# ==========================================
print("Melatih Model Ekstrapolasi Gelombang (Hanya untuk Frekuensi)...")

X_train = monthly_data[features]
y_train_log = monthly_data['Log_Freq']
X_test = all_future_df[features]

# Model 1 & 2: Cyclical Linear (Bagus di pola gelombang)
bayes = BayesianRidge()
bayes.fit(X_train, y_train_log)

huber = HuberRegressor(epsilon=1.35)
huber.fit(X_train, y_train_log)

pred_log_linear = (0.6 * bayes.predict(X_test)) + (0.4 * huber.predict(X_test))
pred_freq_linear = np.expm1(pred_log_linear) # Kembalikan ke angka nyata

# Model 3: Holt-Winters (Raja Time-Series murni)
try:
    hw_model = ExponentialSmoothing(monthly_data['Claim_Frequency'].values, trend='add', seasonal='add', seasonal_periods=12, damped_trend=True)
    hw_fit = hw_model.fit(optimized=True)
    pred_freq_hw = hw_fit.forecast(steps=len(all_future_df))
except:
    hw_model = ExponentialSmoothing(monthly_data['Claim_Frequency'].values, trend='add', damped_trend=True)
    hw_fit = hw_model.fit(optimized=True)
    pred_freq_hw = hw_fit.forecast(steps=len(all_future_df))

# BLEND FREKUENSI: 50% Time-Series + 50% Cyclical Linear
all_future_df['Final_Freq'] = (0.5 * pred_freq_hw) + (0.5 * pred_freq_linear)

# ==========================================
# 5. FILTER TARGET MONTHS & ACTUARIAL DECOUPLING
# ==========================================
submission_rows = []
print("\n--- HASIL PREDIKSI (HARMONIC DECOUPLER) ---")

target_months = ['2025-08-01', '2025-09-01', '2025-10-01', '2025-11-01', '2025-12-01']
final_output = all_future_df[all_future_df['Date'].isin(pd.to_datetime(target_months))].copy()
final_output = final_output.reset_index(drop=True)

# Asumsi inflasi medis rumah sakit: 0.2% per bulan (Standar Aktuaria)
medical_inflation = 1.002 

for i, row in final_output.iterrows():
    month_key = row['Date'].strftime('%Y_%m')
    
    freq = row['Final_Freq']
    
    # Severity TIDAK ditebak oleh AI, tapi dikunci matematis dari Median masa lalu + inflasi
    sev = base_severity * (medical_inflation ** (i + 1))
    
    # Total Klaim murni dari hitungan kalkulator (Frekuensi x Biaya Per Orang)
    total = freq * sev
    
    print(f"{month_key} -> Freq: {freq:.1f} | Sev: {sev:,.0f} | Total: {total:,.0f}")
    
    submission_rows.append({'id': f'{month_key}_Claim_Frequency', 'value': freq})
    submission_rows.append({'id': f'{month_key}_Claim_Severity', 'value': sev})
    submission_rows.append({'id': f'{month_key}_Total_Claim', 'value': total})

submission_df = pd.DataFrame(submission_rows)
submission_df.to_csv('submission_harmonic_decoupler.csv', index=False)
print("\n[LOCKED & PERFECTED] File 'submission_harmonic_decoupler.csv' siap!")
print("Skor 6.0 tadi adalah langkah yang tepat, sekarang kita hilangkan sisa noise-nya!")