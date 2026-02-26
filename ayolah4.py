import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import HuberRegressor
import warnings
warnings.filterwarnings('ignore')

print("Membaca data...")
df_klaim = pd.read_csv('Data_Klaim.csv')

# ==========================================
# 1. PREPROCESSING SAKTI (KEMBALI KE AKAR 5.125)
# ==========================================
date_col = 'Tanggal Pasien Masuk RS'
df_klaim[date_col] = pd.to_datetime(df_klaim[date_col], errors='coerce')
df_klaim = df_klaim.dropna(subset=[date_col])

df_klaim_paid = df_klaim[df_klaim['Status Klaim'] == 'PAID'].copy()

# Capping 98.0% (Terbukti paling ampuh)
upper_limit = df_klaim_paid['Nominal Klaim Yang Disetujui'].quantile(0.98)
df_klaim_paid['Nominal_Klaim_Capped'] = np.clip(df_klaim_paid['Nominal Klaim Yang Disetujui'], a_min=0, a_max=upper_limit)

df_klaim_paid['YearMonth'] = df_klaim_paid[date_col].dt.to_period('M').astype(str)

monthly_data = df_klaim_paid.groupby('YearMonth').agg(
    Claim_Frequency=('Claim ID', 'count'),
    Total_Claim=('Nominal_Klaim_Capped', 'sum') 
).reset_index()

monthly_data['Date'] = pd.to_datetime(monthly_data['YearMonth'] + '-01')
monthly_data = monthly_data.sort_values('Date').reset_index(drop=True)

# ---------------------------------------------------------
# IBNR COMPLETION (KUNCI SKOR 5.125)
# ---------------------------------------------------------
max_idx = len(monthly_data) - 1
last_freq = monthly_data.loc[max_idx, 'Claim_Frequency']
prev_freq = monthly_data.loc[max_idx - 1, 'Claim_Frequency']

print(f"Bulan terakhir di data historis: {monthly_data.loc[max_idx, 'YearMonth']}")

if last_freq < 0.7 * prev_freq:
    print("\n[IBNR COMPLETION AKTIF]")
    print(f"Mengisi data bulan terakhir yang anjlok (x1.4)...")
    monthly_data.loc[max_idx, 'Claim_Frequency'] = last_freq * 1.4
    monthly_data.loc[max_idx, 'Total_Claim'] = monthly_data.loc[max_idx, 'Total_Claim'] * 1.4

monthly_data = monthly_data[monthly_data['Date'] >= '2022-01-01'].reset_index(drop=True)

# ==========================================
# 2. HEAVY FEATURE ENGINEERING (MAXIMAL LAGS)
# ==========================================
def create_features(df):
    df = df.copy()
    df['Month'] = df['Date'].dt.month
    df['Quarter'] = df['Date'].dt.quarter
    
    for col in ['Claim_Frequency', 'Total_Claim']:
        # Tambah lebih banyak lag agar model bisa overfit ke pola historis yang lebih dalam
        for i in [1, 2, 3, 4, 5, 6, 12]:
            df[f'{col}_lag_{i}'] = df[col].shift(i)
            
        # Variasi rolling
        df[f'{col}_roll_mean_3'] = df[col].shift(1).rolling(window=3).mean()
        df[f'{col}_roll_mean_6'] = df[col].shift(1).rolling(window=6).mean()
        df[f'{col}_roll_median_3'] = df[col].shift(1).rolling(window=3).median()
    return df

ts_data = create_features(monthly_data)
months_to_predict = ['2025-08-01', '2025-09-01', '2025-10-01', '2025-11-01', '2025-12-01']
targets = ['Claim_Frequency', 'Total_Claim']

# ==========================================
# 3. MASSIVE HETEROGENEOUS ENSEMBLE (5 ALGORITMA DEWA)
# ==========================================
print("\nMelatih MASSIVE ENSEMBLE (LGBM + XGB + RF + GBM + HUBER)...")
print("Proses ini akan sedikit memakan waktu. Harap bersabar...")

ensemble_preds = {'2025_08': {}, '2025_09': {}, '2025_10': {}, '2025_11': {}, '2025_12': {}}
current_ts_data = ts_data.copy()
features = [c for c in current_ts_data.columns if c not in ['YearMonth', 'Date', 'Claim_Frequency', 'Total_Claim']]

for target in targets:
    for pred_date in months_to_predict:
        train_df = current_ts_data[current_ts_data['Date'] < pred_date].dropna()
        X_train = train_df[features].fillna(0) # Fillna untuk model linear & sklearn
        y_train = train_df[target]
        
        # 1. LIGHTGBM (Best Performer Sebelumnya)
        model_lgb = lgb.LGBMRegressor(objective='mae', learning_rate=0.03, max_depth=4, n_estimators=150, random_state=42, verbose=-1)
        model_lgb.fit(X_train, y_train)
        
        # 2. XGBOOST (Raja Kompetisi Kaggle)
        model_xgb = xgb.XGBRegressor(objective='reg:absoluteerror', learning_rate=0.03, max_depth=3, n_estimators=150, random_state=42)
        model_xgb.fit(X_train, y_train)
        
        # 3. RANDOM FOREST (Sangat kebal overfit pada data kecil)
        model_rf = RandomForestRegressor(criterion='absolute_error', n_estimators=100, max_depth=4, random_state=42)
        model_rf.fit(X_train, y_train)
        
        # 4. GRADIENT BOOSTING (Sklearn fallback)
        model_gbm = GradientBoostingRegressor(loss='absolute_error', learning_rate=0.03, n_estimators=150, max_depth=3, random_state=42)
        model_gbm.fit(X_train, y_train)
        
        # 5. HUBER REGRESSOR (Linear Model Kuat Hadapi Outlier)
        model_huber = HuberRegressor(epsilon=1.35, max_iter=200)
        model_huber.fit(X_train, y_train)
        
        if not (current_ts_data['Date'] == pred_date).any():
            new_row = pd.DataFrame({'Date': [pd.to_datetime(pred_date)]})
            current_ts_data = pd.concat([current_ts_data, new_row], ignore_index=True)
            
        temp_ts_data = create_features(current_ts_data)
        X_test = temp_ts_data[temp_ts_data['Date'] == pred_date][features].fillna(0)
        
        # Prediksi dari kelima model!
        p_lgb = model_lgb.predict(X_test)[0]
        p_xgb = model_xgb.predict(X_test)[0]
        p_rf = model_rf.predict(X_test)[0]
        p_gbm = model_gbm.predict(X_test)[0]
        p_huber = model_huber.predict(X_test)[0]
        
        # THE ULTIMATE BLEND (Bobot diatur berdasarkan rekam jejak)
        # 30% LGBM | 30% XGB | 15% RF | 15% GBM | 10% Huber
        final_pred_val = (0.30 * p_lgb) + (0.30 * p_xgb) + (0.15 * p_rf) + (0.15 * p_gbm) + (0.10 * p_huber)
        
        current_ts_data.loc[current_ts_data['Date'] == pred_date, target] = final_pred_val
        month_key = pred_date[:7].replace('-', '_')
        ensemble_preds[month_key][target] = final_pred_val

# ==========================================
# 4. FINAL EXPORT (KEMBALI KE DECAY TERBAIK)
# ==========================================
submission_rows = []
print("\n--- HASIL PREDIKSI (OVERKILL ENSEMBLE + DECAY 0.98) ---")

# Kembali murni menggunakan Decay Rate 0.98 yang membuktikan skor 5.125
decay_rate = 0.98 

for i, month_key in enumerate(ensemble_preds.keys()):
    freq = ensemble_preds[month_key]['Claim_Frequency'] * (decay_rate ** i)
    total = ensemble_preds[month_key]['Total_Claim'] * (decay_rate ** i)
    
    # Severity otomatis konstan
    sev = total / freq if freq > 0 else 0
    
    print(f"{month_key} -> Freq: {freq:.1f} | Sev: {sev:,.0f} | Total: {total:,.0f}")
    
    submission_rows.append({'id': f'{month_key}_Claim_Frequency', 'value': freq})
    submission_rows.append({'id': f'{month_key}_Claim_Severity', 'value': sev})
    submission_rows.append({'id': f'{month_key}_Total_Claim', 'value': total})

submission_df = pd.DataFrame(submission_rows)
submission_df.to_csv('submission_massive_ensemble.csv', index=False)
print("\nFile 'submission_massive_ensemble.csv' berhasil dibuat! Siap dieksekusi.")