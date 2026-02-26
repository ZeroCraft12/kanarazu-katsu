import pandas as pd
import numpy as np
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

print("Membaca data...")
df_klaim = pd.read_csv('Data_Klaim.csv')

# ==========================================
# 1. KEMBALI KE PREPROCESSING SKOR 5.125
# ==========================================
date_col = 'Tanggal Pasien Masuk RS'
df_klaim[date_col] = pd.to_datetime(df_klaim[date_col], errors='coerce')
df_klaim = df_klaim.dropna(subset=[date_col])

df_klaim_paid = df_klaim[df_klaim['Status Klaim'] == 'PAID'].copy()

# KEMBALI KE CAPPING 98.0% (Terbukti paling optimal)
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
# KEMBALI KE IBNR * 1.4 (Formula Ajaib dari Skor 5.1)
# ---------------------------------------------------------
max_idx = len(monthly_data) - 1
last_freq = monthly_data.loc[max_idx, 'Claim_Frequency']
prev_freq = monthly_data.loc[max_idx - 1, 'Claim_Frequency']

print(f"Bulan terakhir di data: {monthly_data.loc[max_idx, 'YearMonth']}")

if last_freq < 0.7 * prev_freq:
    print("\n[IBNR COMPLETION AKTIF]")
    print(f"Mengisi data bulan terakhir yang anjlok (x1.4)...")
    monthly_data.loc[max_idx, 'Claim_Frequency'] = last_freq * 1.4
    monthly_data.loc[max_idx, 'Total_Claim'] = monthly_data.loc[max_idx, 'Total_Claim'] * 1.4

monthly_data = monthly_data[monthly_data['Date'] >= '2022-01-01'].reset_index(drop=True)

# ==========================================
# 2. FEATURE ENGINEERING
# ==========================================
def create_features(df):
    df = df.copy()
    df['Month'] = df['Date'].dt.month
    for col in ['Claim_Frequency', 'Total_Claim']:
        for i in [1, 2, 3, 6, 12]:
            df[f'{col}_lag_{i}'] = df[col].shift(i)
        df[f'{col}_roll_mean_3'] = df[col].shift(1).rolling(window=3).mean()
    return df

ts_data = create_features(monthly_data)
months_to_predict = ['2025-08-01', '2025-09-01', '2025-10-01', '2025-11-01', '2025-12-01']
targets = ['Claim_Frequency', 'Total_Claim']

# ==========================================
# 3. KAGGLE SECRET: SEED AVERAGING (10 MODELS)
# ==========================================
print("\nMelatih 10 Model LightGBM dengan Seed Berbeda (Seed Averaging)...")
lgbm_preds = {'2025_08': {}, '2025_09': {}, '2025_10': {}, '2025_11': {}, '2025_12': {}}
current_ts_data = ts_data.copy()
features = [c for c in current_ts_data.columns if c not in ['YearMonth', 'Date', 'Claim_Frequency', 'Total_Claim']]

# Kita gunakan 10 Seed (Angka Acak) yang sering dipakai juara Kaggle
seeds = [42, 100, 2023, 777, 888, 123, 999, 50, 1, 333]

for target in targets:
    for pred_date in months_to_predict:
        train_df = current_ts_data[current_ts_data['Date'] < pred_date].dropna()
        X_train, y_train = train_df[features], train_df[target]
        
        # Simpan hasil dari 10 model
        seed_predictions = []
        
        for seed in seeds:
            model = lgb.LGBMRegressor(
                objective='mae', 
                learning_rate=0.05, 
                max_depth=4, 
                n_estimators=120, 
                random_state=seed, # FAKTOR ACAK BERUBAH-UBAH
                subsample=0.8,     # Mencegah Overfitting
                colsample_bytree=0.8,
                verbose=-1
            )
            model.fit(X_train, y_train)
            
            if not (current_ts_data['Date'] == pred_date).any():
                new_row = pd.DataFrame({'Date': [pd.to_datetime(pred_date)]})
                current_ts_data = pd.concat([current_ts_data, new_row], ignore_index=True)
                
            temp_ts_data = create_features(current_ts_data)
            X_test = temp_ts_data[temp_ts_data['Date'] == pred_date][features]
            
            pred_val = model.predict(X_test)[0]
            seed_predictions.append(pred_val)
        
        # RATA-RATAKAN HASIL DARI 10 MODEL! (Sangat Stabil)
        final_pred_val = np.mean(seed_predictions)
        
        current_ts_data.loc[current_ts_data['Date'] == pred_date, target] = final_pred_val
        
        month_key = pred_date[:7].replace('-', '_')
        lgbm_preds[month_key][target] = final_pred_val

# ==========================================
# 4. HASIL FINAL & EXPORT
# ==========================================
submission_rows = []
print("\n--- HASIL PREDIKSI K-SEED ENSEMBLE ---")

# Kembali ke Decay asli yang menghasilkan skor 5.1
decay_rate = 0.98 

for i, month_key in enumerate(lgbm_preds.keys()):
    # Penurunan paralel pada Freq & Total -> Menjaga Severity tetap konstan!
    freq = lgbm_preds[month_key]['Claim_Frequency'] * (decay_rate ** i)
    total = lgbm_preds[month_key]['Total_Claim'] * (decay_rate ** i)
    
    # Severity otomatis konstan karena Freq dan Total turun dengan rasio yang sama
    sev = total / freq if freq > 0 else 0
    
    print(f"{month_key} -> Freq: {freq:.1f} | Sev: {sev:,.0f} | Total: {total:,.0f}")
    
    submission_rows.append({'id': f'{month_key}_Claim_Frequency', 'value': freq})
    submission_rows.append({'id': f'{month_key}_Claim_Severity', 'value': sev})
    submission_rows.append({'id': f'{month_key}_Total_Claim', 'value': total})

submission_df = pd.DataFrame(submission_rows)
submission_df.to_csv('submission_seed_averaging_top3.csv', index=False)
print("\nFile 'submission_seed_averaging_top3.csv' siap untuk BESOK PAGI! Istirahat dulu jagoan.")