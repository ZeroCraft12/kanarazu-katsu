import pandas as pd
import numpy as np
from sklearn.linear_model import BayesianRidge, HuberRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import warnings
warnings.filterwarnings('ignore')

print("Membaca Data Klaim...")
df_klaim = pd.read_csv('Data_Klaim.csv')

# ==========================================
# 1. PEMBERSIHAN & PONDASI DATA (2022+)
# ==========================================
date_col = 'Tanggal Pasien Masuk RS'
df_klaim[date_col] = pd.to_datetime(df_klaim[date_col], errors='coerce')
df_klaim = df_klaim.dropna(subset=[date_col])

df_klaim_paid = df_klaim[df_klaim['Status Klaim'] == 'PAID'].copy()

# Capping di 98.5% (Memberikan sedikit ruang untuk tagihan mahal agar Mean tidak under-predict)
limit = df_klaim_paid['Nominal Klaim Yang Disetujui'].quantile(0.985)
df_klaim_paid['Nominal_Klaim_Capped'] = np.clip(df_klaim_paid['Nominal Klaim Yang Disetujui'], a_min=0, a_max=limit)

df_klaim_paid['YearMonth'] = df_klaim_paid[date_col].dt.to_period('M').astype(str)

monthly_data = df_klaim_paid.groupby('YearMonth').agg(
    Claim_Frequency=('Claim ID', 'count'),
    Total_Claim=('Nominal_Klaim_Capped', 'sum') 
).reset_index()

monthly_data['Date'] = pd.to_datetime(monthly_data['YearMonth'] + '-01')
monthly_data = monthly_data.sort_values('Date').reset_index(drop=True)

# HANYA gunakan data 2022 ke atas (Menghindari anomali Covid 2021)
monthly_data = monthly_data[monthly_data['Date'] >= '2022-01-01'].reset_index(drop=True)

# ==========================================
# 2. THE SAFE IMPUTATION (Penambal Rantai Waktu)
# ==========================================
# Jika bulan terakhir (Juli) datanya anjlok karena RS belum selesai rekap,
# kita tambal menggunakan rata-rata 2 bulan sebelumnya agar Lag_1 Agustus tidak hancur.
max_idx = len(monthly_data) - 1
if monthly_data.loc[max_idx, 'Claim_Frequency'] < 0.8 * monthly_data.loc[max_idx-1, 'Claim_Frequency']:
    print("\n[SAFE IMPUTATION] Menambal data bulan terakhir yang belum lengkap...")
    mean_freq = monthly_data['Claim_Frequency'].iloc[-3:-1].mean()
    mean_total = monthly_data['Total_Claim'].iloc[-3:-1].mean()
    monthly_data.loc[max_idx, 'Claim_Frequency'] = mean_freq
    monthly_data.loc[max_idx, 'Total_Claim'] = mean_total

# ==========================================
# 3. TARGET AKTUARIA (Pemisahan Logika)
# ==========================================
# Kita tebak "Biaya per Orang" (Severity dalam Jutaan), BUKAN Total Klaim langsung!
monthly_data['Severity_M'] = (monthly_data['Total_Claim'] / monthly_data['Claim_Frequency']) / 1e6

# ==========================================
# 4. POLYNOMIAL & KINEMATIC FEATURES
# ==========================================
def create_features(df):
    df = df.copy()
    
    # KUNCI UTAMA: Waktu Kuadratik (Membantu AI menggambar kurva inflasi yang melengkung naik)
    df['Time_Index'] = np.arange(1, len(df) + 1)
    df['Time_Sq'] = df['Time_Index'] ** 2
    
    # Trigonometri Musim (Menghindari "Desember = 12x lipat lebih besar dari Januari")
    months = df['Date'].dt.month
    df['sin_M'] = np.sin(2 * np.pi * months / 12)
    df['cos_M'] = np.cos(2 * np.pi * months / 12)
    
    for col in ['Claim_Frequency', 'Severity_M']:
        # Lag pendek untuk momentum instan
        for i in [1, 2]:
            df[f'{col}_lag_{i}'] = df[col].shift(i)
        
        # Exponential Moving Average (Jauh lebih responsif dari Rolling Mean biasa)
        df[f'{col}_ema_3'] = df[col].shift(1).ewm(span=3, adjust=False).mean()
        
    return df

ts_data = create_features(monthly_data)
months_to_predict = ['2025-08-01', '2025-09-01', '2025-10-01', '2025-11-01', '2025-12-01']
targets = ['Claim_Frequency', 'Severity_M']

# ==========================================
# 5. PURE SCIKIT-LEARN ENSEMBLE (BEBAS Kutukan HW/ARIMA)
# ==========================================
print("\nMelatih Model Polynomial Actuary...")
current_ts_data = ts_data.copy()
final_preds = {'2025_08': {}, '2025_09': {}, '2025_10': {}, '2025_11': {}, '2025_12': {}}
        
for i, pred_date in enumerate(months_to_predict):
    
    if not (current_ts_data['Date'] == pred_date).any():
        new_row = pd.DataFrame({'Date': [pd.to_datetime(pred_date)]})
        current_ts_data = pd.concat([current_ts_data, new_row], ignore_index=True)
        
    temp_df = create_features(current_ts_data)
    
    exclude_cols = ['YearMonth', 'Date', 'Claim_Frequency', 'Total_Claim', 'Severity_M']
    features = [c for c in temp_df.columns if c not in exclude_cols]
    
    for target in targets:
        train_df = temp_df[temp_df['Date'] < pred_date].dropna()
        X_train = train_df[features]
        y_train = train_df[target]
        
        X_test = temp_df[temp_df['Date'] == pred_date][features].fillna(0)
        
        # MODEL 1: Bayesian Ridge (Standardisasi + Super Stabil)
        bayes = make_pipeline(StandardScaler(), BayesianRidge()).fit(X_train, y_train)
        
        # MODEL 2: Huber Regressor (Sangat tangguh melawan data aneh/outlier)
        huber = make_pipeline(StandardScaler(), HuberRegressor(epsilon=1.35)).fit(X_train, y_train)
        
        pred_bayes = bayes.predict(X_test)[0]
        pred_huber = huber.predict(X_test)[0]
        
        # BLENDING 50-50: Stabil + Tangguh
        final_val = (0.50 * pred_bayes) + (0.50 * pred_huber)
        
        current_ts_data.loc[current_ts_data['Date'] == pred_date, target] = final_val
        
        month_key = pred_date[:7].replace('-', '_')
        final_preds[month_key][target] = final_val

# ==========================================
# 6. REKONSTRUKSI & EXPORT HASIL
# ==========================================
submission_rows = []
print("\n--- HASIL PREDIKSI (THE POLYNOMIAL ACTUARY) ---")

for month_key in final_preds.keys():
    freq = final_preds[month_key]['Claim_Frequency']
    sev_m = final_preds[month_key]['Severity_M']
    
    # Konversi kembali dari Jutaan ke nilai asli
    sev = sev_m * 1e6
    # Total Klaim adalah hasil kali Frekuensi x Biaya per Orang
    total = freq * sev
    
    print(f"{month_key} -> Freq: {freq:.1f} | Sev: {sev:,.0f} | Total: {total:,.0f}")
    
    submission_rows.append({'id': f'{month_key}_Claim_Frequency', 'value': freq})
    submission_rows.append({'id': f'{month_key}_Claim_Severity', 'value': sev})
    submission_rows.append({'id': f'{month_key}_Total_Claim', 'value': total})

submission_df = pd.DataFrame(submission_rows)
submission_df.to_csv('submission_polynomial_actuary.csv', index=False)
print("\n[LOCKED] File 'submission_polynomial_actuary.csv' siap!")
print("Hantu skor kembar telah dibasmi. Ini adalah hitungan murni yang akan membelah tembok 3.0!")