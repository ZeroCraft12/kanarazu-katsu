import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
from sklearn.linear_model import Ridge
import warnings
warnings.filterwarnings('ignore')

print("Membaca Data Klaim...")
df_klaim = pd.read_csv('Data_Klaim.csv')

# ==========================================
# 1. PONDASI DATA (PREPROCESSING & CAPPING)
# ==========================================
date_col = 'Tanggal Pasien Masuk RS'
df_klaim[date_col] = pd.to_datetime(df_klaim[date_col], errors='coerce')
df_klaim = df_klaim.dropna(subset=[date_col])

df_klaim_paid = df_klaim[df_klaim['Status Klaim'] == 'PAID'].copy()

# Capping di 98.0% (Paling stabil untuk semua skor kita)
upper_limit = df_klaim_paid['Nominal Klaim Yang Disetujui'].quantile(0.98)
df_klaim_paid['Nominal_Klaim_Capped'] = np.clip(df_klaim_paid['Nominal Klaim Yang Disetujui'], a_min=0, a_max=upper_limit)

df_klaim_paid['YearMonth'] = df_klaim_paid[date_col].dt.to_period('M').astype(str)

monthly_data = df_klaim_paid.groupby('YearMonth').agg(
    Claim_Frequency=('Claim ID', 'count'),
    Total_Claim=('Nominal_Klaim_Capped', 'sum') 
).reset_index()

monthly_data['Date'] = pd.to_datetime(monthly_data['YearMonth'] + '-01')
monthly_data = monthly_data.sort_values('Date').reset_index(drop=True)
monthly_data['Claim_Severity'] = monthly_data['Total_Claim'] / monthly_data['Claim_Frequency']

# ---------------------------------------------------------
# IBNR COMPLETION (KUNCI STABILITAS SKOR 4.8)
# ---------------------------------------------------------
max_idx = len(monthly_data) - 1
last_freq = monthly_data.loc[max_idx, 'Claim_Frequency']
prev_freq = monthly_data.loc[max_idx - 1, 'Claim_Frequency']

if last_freq < 0.7 * prev_freq:
    print(f"\n[IBNR COMPLETION AKTIF] Menggenapkan data bulan terakhir yang anjlok x1.4")
    monthly_data.loc[max_idx, 'Claim_Frequency'] *= 1.4
    monthly_data.loc[max_idx, 'Total_Claim'] *= 1.4
    monthly_data.loc[max_idx, 'Claim_Severity'] = monthly_data.loc[max_idx, 'Total_Claim'] / monthly_data.loc[max_idx, 'Claim_Frequency']

# Ubah Target Menjadi Log-Transform agar AI sangat kebal fluktuasi
monthly_data['Log_Freq'] = np.log1p(monthly_data['Claim_Frequency'])
monthly_data['Log_Sev'] = np.log1p(monthly_data['Claim_Severity'])

# ==========================================
# 2. DIRECT FORECASTING (ANTI-ERROR ACCUMULATION)
# ==========================================
# Buat wadah waktu masa depan (Agustus - Desember 2025)
future_dates = pd.to_datetime(['2025-08-01', '2025-09-01', '2025-10-01', '2025-11-01', '2025-12-01'])
future_df = pd.DataFrame({'Date': future_dates})

# Gabungkan data masa lalu dan wadah masa depan
all_data = pd.concat([monthly_data, future_df], ignore_index=True)
all_data['Time_Index'] = np.arange(len(all_data))
all_data['Month'] = all_data['Date'].dt.month

# THE MASTER TRICK: HANYA GUNAKAN LAG 6 DAN LAG 12
# Karena kita menebak maksimal 5 bulan ke depan, Lag 6 untuk bulan Desember adalah bulan Juni (Data Asli!)
# Ini menjamin model TIDAK PERNAH memakan tebakannya sendiri. Tidak akan ada NaN atau meledak!
for col in ['Log_Freq', 'Log_Sev']:
    all_data[f'{col}_lag_6'] = all_data[col].shift(6)
    all_data[f'{col}_lag_12'] = all_data[col].shift(12)

# Pisahkan kembali data latih (Train) dan data uji (Test)
# Kita mulai belajar dari 2022 agar trennya relevan
train_data = all_data[(all_data['Date'] >= '2022-01-01') & (all_data['Date'] <= monthly_data['Date'].max())].copy()
test_data = all_data[all_data['Date'] > monthly_data['Date'].max()].copy()

# ==========================================
# 3. ROBUST ENSEMBLE TRAINING
# ==========================================
print("\nMelatih Model Direct Multi-Step Forecasting (Aman dari ledakan Error)...")

for target in ['Log_Freq', 'Log_Sev']:
    features = ['Time_Index', 'Month', f'{target}_lag_6', f'{target}_lag_12']
    
    X_train = train_data[features].fillna(0)
    y_train = train_data[target]
    X_test = test_data[features].fillna(0)
    
    # 1. Ridge Regression (Penarik Garis Lurus yang sangat stabil)
    ridge = Ridge(alpha=5.0)
    ridge.fit(X_train, y_train)
    
    # 2. LightGBM (Pembaca pola siklus bulan-ke-bulan)
    lgbm = lgb.LGBMRegressor(objective='mae', learning_rate=0.05, max_depth=3, n_estimators=100, random_state=42, verbose=-1)
    lgbm.fit(X_train, y_train)
    
    # 3. XGBoost
    xgb_model = xgb.XGBRegressor(objective='reg:absoluteerror', learning_rate=0.05, max_depth=3, n_estimators=100, random_state=42)
    xgb_model.fit(X_train, y_train)
    
    # BLEND: Kombinasi kestabilan Linear (40%) dan Kejelian Pohon (60%)
    preds_log = (0.4 * ridge.predict(X_test)) + (0.3 * lgbm.predict(X_test)) + (0.3 * xgb_model.predict(X_test))
    
    # Simpan hasil tebakan log ke test_data
    test_data.loc[:, target] = preds_log

# ==========================================
# 4. EXPORT HASIL AKTUARIA MURNI
# ==========================================
submission_rows = []
print("\n--- HASIL PREDIKSI (STABIL & AMAN) ---")
print("Prediksi Frekuensi dan Severity (Biaya Per Orang) dilakukan terpisah.")

# Cari Median Severity historis sebagai jaring pengaman
hist_sev_median = monthly_data['Claim_Severity'].median()

for _, row in test_data.iterrows():
    month_key = row['Date'].strftime('%Y_%m')
    
    # Kembalikan skala Logaritma ke angka asli Miliaran/Ratusan
    freq = np.expm1(row['Log_Freq'])
    sev = np.expm1(row['Log_Sev'])
    
    # SAFETY NET: Pastikan AI tidak berhalusinasi biaya RS yang terlalu ekstrem
    if sev < 0.6 * hist_sev_median or sev > 1.8 * hist_sev_median:
        sev = hist_sev_median
        
    # Total Claim murni dari hasil Freq x Sev
    total = freq * sev
    
    print(f"{month_key} -> Freq: {freq:.1f} | Sev: {sev:,.0f} | Total: {total:,.0f}")
    
    submission_rows.append({'id': f'{month_key}_Claim_Frequency', 'value': freq})
    submission_rows.append({'id': f'{month_key}_Claim_Severity', 'value': sev})
    submission_rows.append({'id': f'{month_key}_Total_Claim', 'value': total})

submission_df = pd.DataFrame(submission_rows)
submission_df.to_csv('submission_direct_forecast_safe.csv', index=False)
print("\nFile 'submission_direct_forecast_safe.csv' berhasil dibuat! Silakan Submit!")