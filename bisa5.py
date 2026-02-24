import pandas as pd
import numpy as np
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

print("Membaca data...")
df_klaim = pd.read_csv('Data_Klaim.csv')

# ==========================================
# 1. PREPROCESSING & SWEET-SPOT CAPPING
# ==========================================
date_col = 'Tanggal Pasien Masuk RS'
df_klaim[date_col] = pd.to_datetime(df_klaim[date_col], errors='coerce')
df_klaim = df_klaim.dropna(subset=[date_col])

df_klaim_paid = df_klaim[df_klaim['Status Klaim'] == 'PAID'].copy()

# MICRO-TUNE 1: Capping dinaikkan sedikit ke 98.2% agar tidak terlalu banyak memotong data riil
upper_limit = df_klaim_paid['Nominal Klaim Yang Disetujui'].quantile(0.982)
df_klaim_paid['Nominal_Klaim_Capped'] = np.clip(df_klaim_paid['Nominal Klaim Yang Disetujui'], a_min=0, a_max=upper_limit)

df_klaim_paid['YearMonth'] = df_klaim_paid[date_col].dt.to_period('M').astype(str)

monthly_data = df_klaim_paid.groupby('YearMonth').agg(
    Claim_Frequency=('Claim ID', 'count'),
    Total_Claim=('Nominal_Klaim_Capped', 'sum') 
).reset_index()

monthly_data['Date'] = pd.to_datetime(monthly_data['YearMonth'] + '-01')
monthly_data = monthly_data.sort_values('Date').reset_index(drop=True)

# ---------------------------------------------------------
# MICRO-TUNE 2: DYNAMIC IBNR COMPLETION
# ---------------------------------------------------------
max_idx = len(monthly_data) - 1
last_freq = monthly_data.loc[max_idx, 'Claim_Frequency']
prev_freq = monthly_data.loc[max_idx - 1, 'Claim_Frequency']

print(f"Bulan terakhir di data: {monthly_data.loc[max_idx, 'YearMonth']}")

# Jika bulan terakhir drop drastis (> 20%)
if last_freq < 0.8 * prev_freq:
    print("\n[DYNAMIC IBNR COMPLETION AKTIF]")
    
    # Hitung rata-rata 2 bulan penuh sebelumnya
    mean_freq_2mo = monthly_data['Claim_Frequency'].iloc[-3:-1].mean()
    mean_tot_2mo = monthly_data['Total_Claim'].iloc[-3:-1].mean()
    
    # Kita asumsikan bulan terakhir ini seharusnya bernilai 95% dari rata-rata bulan sebelumnya (karena tren memang sedang turun)
    filled_freq = mean_freq_2mo * 0.95
    filled_total = mean_tot_2mo * 0.95
    
    monthly_data.loc[max_idx, 'Claim_Frequency'] = filled_freq
    monthly_data.loc[max_idx, 'Total_Claim'] = filled_total
    
    print(f"Frekuensi digenapkan secara dinamis dari {last_freq:.0f} menjadi {filled_freq:.1f}")

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
# 3. LIGHTGBM (PENYESUAIAN LEARNING RATE)
# ==========================================
print("\nMelatih Model LightGBM (Micro-Tuned)...")
lgbm_preds = {'2025_08': {}, '2025_09': {}, '2025_10': {}, '2025_11': {}, '2025_12': {}}
current_ts_data = ts_data.copy()
features = [c for c in current_ts_data.columns if c not in ['YearMonth', 'Date', 'Claim_Frequency', 'Total_Claim']]

for target in targets:
    for pred_date in months_to_predict:
        train_df = current_ts_data[current_ts_data['Date'] < pred_date].dropna()
        X_train, y_train = train_df[features], train_df[target]
        
        # Learning rate sedikit diperlambat agar lebih presisi menangkap garis regresi
        model = lgb.LGBMRegressor(objective='mae', learning_rate=0.04, max_depth=4, n_estimators=150, random_state=42, verbose=-1)
        model.fit(X_train, y_train)
        
        if not (current_ts_data['Date'] == pred_date).any():
            new_row = pd.DataFrame({'Date': [pd.to_datetime(pred_date)]})
            current_ts_data = pd.concat([current_ts_data, new_row], ignore_index=True)
            
        current_ts_data = create_features(current_ts_data)
        X_test = current_ts_data[current_ts_data['Date'] == pred_date][features]
        
        pred_val = model.predict(X_test)[0]
        current_ts_data.loc[current_ts_data['Date'] == pred_date, target] = pred_val
        
        month_key = pred_date[:7].replace('-', '_')
        lgbm_preds[month_key][target] = pred_val

# ==========================================
# 4. MICRO-TUNE 3: MEDICAL INFLATION & DECAY
# ==========================================
submission_rows = []
print("\n--- HASIL PREDIKSI FINAL (INFLASI MEDIS + DECAY) ---")

freq_decay = 0.985  # Penurunan frekuensi klaim 1.5% per bulan
sev_inflation = 1.005 # Kenaikan biaya per klaim (Severity) 0.5% per bulan karena inflasi RS

for i, month_key in enumerate(lgbm_preds.keys()):
    # Frekuensi menurun sesuai tren
    freq = lgbm_preds[month_key]['Claim_Frequency'] * (freq_decay ** i)
    
    # Ambil tebakan awal Total Claim, lalu cari Severity dasarnya
    base_total = lgbm_preds[month_key]['Total_Claim']
    base_sev = base_total / lgbm_preds[month_key]['Claim_Frequency']
    
    # Severity mengalami INFLASI naik!
    sev = base_sev * (sev_inflation ** i)
    
    # Total Claim dihitung ulang dengan Severity yang sudah diinflasi
    total = freq * sev
    
    print(f"{month_key} -> Freq: {freq:.1f} | Sev: {sev:,.0f} | Total: {total:,.0f}")
    
    submission_rows.append({'id': f'{month_key}_Claim_Frequency', 'value': freq})
    submission_rows.append({'id': f'{month_key}_Claim_Severity', 'value': sev})
    submission_rows.append({'id': f'{month_key}_Total_Claim', 'value': total})

submission_df = pd.DataFrame(submission_rows)
submission_df.to_csv('submission_final_push.csv', index=False)
print("\nFile 'submission_final_push.csv' berhasil dibuat! Bismillah, semoga rekor < 3 pecah sekarang!")