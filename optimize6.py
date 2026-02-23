import pandas as pd #SKOR 11
import numpy as np
import lightgbm as lgb
from sklearn.linear_model import Ridge
import warnings
warnings.filterwarnings('ignore')

# ==========================================
# 1. PERSIAPAN DATA & PRECISE CAPPING
# ==========================================
print("Membaca data...")
df_klaim = pd.read_csv('Data_Klaim.csv')

df_klaim['Tanggal Pasien Masuk RS'] = pd.to_datetime(df_klaim['Tanggal Pasien Masuk RS'])
date_col = 'Tanggal Pasien Masuk RS' 
df_klaim = df_klaim.dropna(subset=[date_col])

df_klaim_paid = df_klaim[df_klaim['Status Klaim'] == 'PAID'].copy()

# KEMBALI KE CAPPING YANG BENAR: 98.5%
# Kita hanya memotong 1.5% klaim paling ekstrem (Outlier sejati)
upper_limit = df_klaim_paid['Nominal Klaim Yang Disetujui'].quantile(0.985)
df_klaim_paid['Nominal_Klaim_Capped'] = np.clip(df_klaim_paid['Nominal Klaim Yang Disetujui'], a_min=0, a_max=upper_limit)

df_klaim_paid['YearMonth'] = df_klaim_paid[date_col].dt.to_period('M').astype(str)

# ==========================================
# 2. AGREGASI BULANAN & FILTER BULAN BOCOR
# ==========================================
monthly_data = df_klaim_paid.groupby('YearMonth').agg(
    Claim_Frequency=('Claim ID', 'count'),
    Total_Claim=('Nominal_Klaim_Capped', 'sum') 
).reset_index()

monthly_data['Date'] = pd.to_datetime(monthly_data['YearMonth'] + '-01')
monthly_data = monthly_data.sort_values('Date').reset_index(drop=True)
monthly_data['Claim_Severity'] = monthly_data['Total_Claim'] / monthly_data['Claim_Frequency']

# --- KAGGLE GRANDMASTER TRICK: INCOMPLETE MONTH DROP ---
# Jika bulan terakhir di data historis jumlah klaimnya drop >40% dari bulan sebelumnya,
# itu artinya datanya terpotong (misal baru rekap tanggal 15). AI akan tertipu kalau ini dipakai!
last_idx = len(monthly_data) - 1
if monthly_data.loc[last_idx, 'Claim_Frequency'] < 0.6 * monthly_data.loc[last_idx - 1, 'Claim_Frequency']:
    print(f"\n[WARNING] Bulan {monthly_data.loc[last_idx, 'YearMonth']} terdeteksi tidak lengkap (bocor). Dihapus dari data latih!")
    monthly_data = monthly_data.iloc[:-1].reset_index(drop=True)

# Batasi mulai dari 2022 agar AI tidak bingung dengan era awal pandemi
monthly_data = monthly_data[monthly_data['Date'] >= '2022-01-01'].reset_index(drop=True)

# ==========================================
# 3. FEATURE ENGINEERING
# ==========================================
def create_features(df):
    df = df.copy()
    df['Month'] = df['Date'].dt.month
    
    for col in ['Claim_Frequency', 'Claim_Severity']:
        for i in [1, 2, 3, 12]: # Lag 1,2,3 untuk tren pendek, Lag 12 untuk seasonality tahunan
            df[f'{col}_lag_{i}'] = df[col].shift(i)
            
        # PERBAIKAN BUG NaN: Gunakan shift(1) agar tidak memasukkan nilai NaN bulan ini ke hitungan rata-rata
        df[f'{col}_roll_mean_3'] = df[col].shift(1).rolling(window=3).mean()
        
    return df

ts_data = create_features(monthly_data)

# ==========================================
# 4. ENSEMBLE MODELING (LGBM + RIDGE) & PREDIKSI
# ==========================================
targets = ['Claim_Frequency', 'Claim_Severity']
months_to_predict = ['2025-08-01', '2025-09-01', '2025-10-01', '2025-11-01', '2025-12-01']
predictions = {'2025_08': {}, '2025_09': {}, '2025_10': {}, '2025_11': {}, '2025_12': {}}

current_ts_data = ts_data.copy()
exclude_cols = ['YearMonth', 'Date', 'Claim_Frequency', 'Total_Claim', 'Claim_Severity']
features = [c for c in current_ts_data.columns if c not in exclude_cols]

print("\nMelatih Model Presisi Tinggi (LGBM + Ridge)...")

for target in targets:
    for pred_date in months_to_predict:
        train_df = current_ts_data[current_ts_data['Date'] < pred_date].dropna()
        
        # PERBAIKAN BUG NaN: fillna(0) memastikan Ridge sama sekali tidak memakan nilai kosong
        X_train = train_df[features].fillna(0)
        y_train = train_df[target]
        
        # 1. Model LightGBM (Membaca pola non-linear & musiman)
        lgb_model = lgb.LGBMRegressor(
            objective='regression',
            metric='rmse',
            learning_rate=0.03,
            max_depth=4,
            n_estimators=120,
            random_state=42,
            verbose=-1
        )
        lgb_model.fit(X_train, y_train)
        
        # 2. Model Ridge (Membaca tren linear dengan sangat stabil)
        ridge_model = Ridge(alpha=10.0)
        ridge_model.fit(X_train, y_train)
        
        # Tambahkan baris untuk bulan yang diprediksi jika belum ada
        if not (current_ts_data['Date'] == pred_date).any():
            new_row = pd.DataFrame({'Date': [pd.to_datetime(pred_date)]})
            current_ts_data = pd.concat([current_ts_data, new_row], ignore_index=True)
            
        current_ts_data = create_features(current_ts_data)
        
        # PERBAIKAN BUG NaN: fillna(0) untuk pengujian juga
        X_test = current_ts_data[current_ts_data['Date'] == pred_date][features].fillna(0)
        
        # ENSEMBLE: 60% LGBM (Musiman) + 40% Ridge (Stabilitas)
        pred_lgb = lgb_model.predict(X_test)[0]
        pred_ridge = ridge_model.predict(X_test)[0]
        final_pred = (0.6 * pred_lgb) + (0.4 * pred_ridge)
        
        # Update dataset agar bisa jadi 'Lag' untuk prediksi bulan berikutnya
        current_ts_data.loc[current_ts_data['Date'] == pred_date, target] = final_pred
        
        month_key = pred_date[:7].replace('-', '_')
        predictions[month_key][target] = final_pred

# ==========================================
# 5. HIGH-PRECISION FORMATTING (SESUAI INSTINGMU)
# ==========================================
submission_rows = []
print("\nHasil Prediksi Akhir (Full Desimal):")

for month_key, preds in predictions.items():
    # KITA BIARKAN ANGKA MENGGUNAKAN FLOAT64 PENUH TANPA DIBULATKAN
    freq = float(preds['Claim_Frequency'])
    sev = float(preds['Claim_Severity'])
    total = float(freq * sev)
    
    print(f"{month_key} -> Freq: {freq:.4f} | Sev: {sev:.4f} | Total: {total:.4f}")
    
    submission_rows.append({'id': f'{month_key}_Claim_Frequency', 'value': freq})
    submission_rows.append({'id': f'{month_key}_Claim_Severity', 'value': sev})
    submission_rows.append({'id': f'{month_key}_Total_Claim', 'value': total})

submission_df = pd.DataFrame(submission_rows)
submission_df.to_csv('submission_precision_ensemble.csv', index=False)
print("\nFile 'submission_precision_ensemble.csv' siap! Bismillah < 3.")