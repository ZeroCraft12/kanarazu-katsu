import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

print("Membaca data Klaim...")
# Kita tinggalkan data Polis, kembali ke fondasi murni Klaim yang terbukti dapat 5.125
df_klaim = pd.read_csv('Data_Klaim.csv')

# ==========================================
# 1. PREPROCESSING MURNI (KUNCI SKOR 5.125)
# ==========================================
date_col = 'Tanggal Pasien Masuk RS'
df_klaim[date_col] = pd.to_datetime(df_klaim[date_col], errors='coerce')
df_klaim = df_klaim.dropna(subset=[date_col])

df_klaim_paid = df_klaim[df_klaim['Status Klaim'] == 'PAID'].copy()

# Capping Sakti 98.0%
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

if last_freq < 0.7 * prev_freq:
    print("\n[IBNR COMPLETION AKTIF] Menggenapkan data bulan terakhir x1.4")
    monthly_data.loc[max_idx, 'Claim_Frequency'] = last_freq * 1.4
    monthly_data.loc[max_idx, 'Total_Claim'] = monthly_data.loc[max_idx, 'Total_Claim'] * 1.4

monthly_data = monthly_data[monthly_data['Date'] >= '2022-01-01'].reset_index(drop=True)

# ==========================================
# 2. FEATURE ENGINEERING SEDERHANA & KUAT
# ==========================================
def create_features(df):
    df = df.copy()
    df['Month'] = df['Date'].dt.month
    df['Quarter'] = df['Date'].dt.quarter
    
    for col in ['Claim_Frequency', 'Total_Claim']:
        for i in [1, 2, 3, 6, 12]:
            df[f'{col}_lag_{i}'] = df[col].shift(i)
        df[f'{col}_roll_mean_3'] = df[col].shift(1).rolling(window=3).mean()
    return df

ts_data = create_features(monthly_data)
months_to_predict = ['2025-08-01', '2025-09-01', '2025-10-01', '2025-11-01', '2025-12-01']
targets = ['Claim_Frequency', 'Total_Claim']

# ==========================================
# 3. PURE MACHINE LEARNING (TANPA PAKSAAN/HACK)
# ==========================================
print("\nMelatih Model Pure ML (LGBM + XGBoost)...")
adv_preds = {'2025_08': {}, '2025_09': {}, '2025_10': {}, '2025_11': {}, '2025_12': {}}
current_ts_data = ts_data.copy()

feat_cols = [c for c in current_ts_data.columns if c not in ['YearMonth', 'Date', 'Claim_Frequency', 'Total_Claim']]

for target in targets:
    for pred_date in months_to_predict:
        train_df = current_ts_data[current_ts_data['Date'] < pred_date].dropna()
        X_train = train_df[feat_cols].fillna(0)
        y_train = train_df[target]
        
        # Kembali menggunakan paramter murni skor 5.125
        m_lgb = lgb.LGBMRegressor(objective='mae', learning_rate=0.05, max_depth=4, n_estimators=120, random_state=42, verbose=-1)
        m_xgb = xgb.XGBRegressor(objective='reg:absoluteerror', learning_rate=0.05, max_depth=4, n_estimators=120, random_state=42)
        
        m_lgb.fit(X_train, y_train)
        m_xgb.fit(X_train, y_train)
        
        if not (current_ts_data['Date'] == pred_date).any():
            new_row = pd.DataFrame({'Date': [pd.to_datetime(pred_date)]})
            current_ts_data = pd.concat([current_ts_data, new_row], ignore_index=True)
            
        current_ts_data = create_features(current_ts_data)
        X_test = current_ts_data[current_ts_data['Date'] == pred_date][feat_cols].fillna(0)
        
        # Ensemble 50:50 yang natural
        pred_val = (0.5 * m_lgb.predict(X_test)[0]) + (0.5 * m_xgb.predict(X_test)[0])
        current_ts_data.loc[current_ts_data['Date'] == pred_date, target] = pred_val
        
        month_key = pred_date[:7].replace('-', '_')
        adv_preds[month_key][target] = pred_val

# ==========================================
# 4. EXPORT MURNI (TANPA DECAY BUATAN)
# ==========================================
submission_rows = []
print("\n--- HASIL PREDIKSI (PURE ML - NO ARTIFICIAL DECAY) ---")
print("Kita buang semua diskon paksa. Ini adalah prediksi murni AI!")

for month_key in adv_preds.keys():
    # TIDAK ADA LAGI DECAY RATE. KITA BIARKAN ANGKA ASLI DARI AI.
    final_freq = adv_preds[month_key]['Claim_Frequency']
    final_total = adv_preds[month_key]['Total_Claim']
    
    final_sev = final_total / final_freq if final_freq > 0 else 0
    
    print(f"{month_key} -> Freq: {final_freq:.1f} | Sev: {final_sev:,.0f} | Total: {final_total:,.0f}")
    
    submission_rows.append({'id': f'{month_key}_Claim_Frequency', 'value': final_freq})
    submission_rows.append({'id': f'{month_key}_Claim_Severity', 'value': final_sev})
    submission_rows.append({'id': f'{month_key}_Total_Claim', 'value': final_total})

submission_df = pd.DataFrame(submission_rows)
submission_df.to_csv('submission_pure_ml_recovery.csv', index=False)
print("\nFile 'submission_pure_ml_recovery.csv' siap!")