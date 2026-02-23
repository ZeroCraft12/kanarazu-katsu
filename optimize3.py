import pandas as pd #SKOR 9
import numpy as np
import lightgbm as lgb
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# ==========================================
# 1. PERSIAPAN DATA KLAIM & FILTER TAHUN
# ==========================================
print("Membaca data...")
df_klaim = pd.read_csv('Data_Klaim.csv')

# Preprocessing Data Klaim
df_klaim['Tanggal Pasien Masuk RS'] = pd.to_datetime(df_klaim['Tanggal Pasien Masuk RS'])
date_col = 'Tanggal Pasien Masuk RS' 
df_klaim = df_klaim.dropna(subset=[date_col])

# Hanya gunakan klaim berstatus PAID
df_klaim_paid = df_klaim[df_klaim['Status Klaim'] == 'PAID'].copy()

# OUTLIER CAPPING - Agak diperketat ke persentil 95 karena target metrik sangat sensitif
upper_limit = df_klaim_paid['Nominal Klaim Yang Disetujui'].quantile(0.95)
df_klaim_paid['Nominal_Klaim_Capped'] = np.clip(df_klaim_paid['Nominal Klaim Yang Disetujui'], a_min=0, a_max=upper_limit)

df_klaim_paid['YearMonth'] = df_klaim_paid[date_col].dt.to_period('M').astype(str)

# ==========================================
# 2. AGREGASI & LOG-TRANSFORMATION (KUNCI SKOR < 3)
# ==========================================
print("Melakukan agregasi bulanan...")
monthly_data = df_klaim_paid.groupby('YearMonth').agg(
    Claim_Frequency=('Claim ID', 'count'),
    Total_Claim=('Nominal_Klaim_Capped', 'sum') 
).reset_index()

monthly_data['Date'] = pd.to_datetime(monthly_data['YearMonth'] + '-01')
monthly_data = monthly_data.sort_values('Date').reset_index(drop=True)

# Hitung Severity
monthly_data['Claim_Severity'] = monthly_data['Total_Claim'] / monthly_data['Claim_Frequency']

# FILTER WAKTU: Buang data sebelum 2022. Era Pandemi (2020-2021) polanya terlalu acak dan merusak model.
monthly_data = monthly_data[monthly_data['Date'] >= '2022-01-01'].reset_index(drop=True)

# THE SECRET WEAPON: Log Transformation menggunakan np.log1p (log(1 + x))
monthly_data['Log_Freq'] = np.log1p(monthly_data['Claim_Frequency'])
monthly_data['Log_Sev'] = np.log1p(monthly_data['Claim_Severity'])

# ==========================================
# 3. FEATURE ENGINEERING MENGGUNAKAN LOG
# ==========================================
def create_features(df):
    df = df.copy()
    df['Month'] = df['Date'].dt.month
    df['Quarter'] = df['Date'].dt.quarter
    
    # Lag Features dari Log-Transformed targets
    for col in ['Log_Freq', 'Log_Sev']:
        for i in [1, 2, 3, 6]: # Kita kurangi lag 12 agar model fokus pada tren jangka pendek (recent trends)
            df[f'{col}_lag_{i}'] = df[col].shift(i)
            
    # Rolling Averages (Muluskan tren)
    for col in ['Log_Freq', 'Log_Sev']:
        df[f'{col}_roll_mean_3'] = df[col].rolling(window=3).mean()
        df[f'{col}_roll_median_3'] = df[col].rolling(window=3).median()
        
    return df

ts_data = create_features(monthly_data)

# ==========================================
# 4. MODELING DENGAN AR-LGBM (Auto-Regressive)
# ==========================================
targets = ['Log_Freq', 'Log_Sev'] # KITA PREDIKSI LOG-NYA!
predictions = {'2025_08': {}, '2025_09': {}, '2025_10': {}, '2025_11': {}, '2025_12': {}}
months_to_predict = ['2025-08-01', '2025-09-01', '2025-10-01', '2025-11-01', '2025-12-01']

current_ts_data = ts_data.copy()
# Fitur tidak boleh mengandung nilai asli (hanya log, month, dan lag)
exclude_cols = ['YearMonth', 'Date', 'Claim_Frequency', 'Total_Claim', 'Claim_Severity', 'Log_Freq', 'Log_Sev']
features = [c for c in current_ts_data.columns if c not in exclude_cols]

print("Melatih model LightGBM dengan Log-Target...")

for target in targets:
    # Parameter dibuat sederhana (max_depth 3) untuk MENCEGAH OVERFITTING
    params = {
        'objective': 'rmse', 
        'learning_rate': 0.05,
        'max_depth': 3,       
        'num_leaves': 7,      
        'n_estimators': 80,   
        'random_state': 42,
        'verbose': -1
    }
    
    for pred_date in months_to_predict:
        train_df = current_ts_data[current_ts_data['Date'] < pred_date].dropna()
        
        X_train = train_df[features]
        y_train = train_df[target]
        
        model = lgb.LGBMRegressor(**params)
        model.fit(X_train, y_train)
        
        # Tambah baris kosong jika belum ada
        if not (current_ts_data['Date'] == pred_date).any():
            new_row = pd.DataFrame({'Date': [pd.to_datetime(pred_date)]})
            current_ts_data = pd.concat([current_ts_data, new_row], ignore_index=True)
            
        current_ts_data = create_features(current_ts_data)
        X_test = current_ts_data[current_ts_data['Date'] == pred_date][features]
        
        pred_log_value = model.predict(X_test)[0]
        current_ts_data.loc[current_ts_data['Date'] == pred_date, target] = pred_log_value
        
        # INVERSE TRANSFORM (Kembalikan ke angka asli dengan np.expm1)
        original_value = np.expm1(pred_log_value)
        
        # Mapping nama untuk output
        target_name = 'Claim_Frequency' if target == 'Log_Freq' else 'Claim_Severity'
        
        month_key = pred_date[:7].replace('-', '_')
        predictions[month_key][target_name] = original_value

# ==========================================
# 5. FORMATTING & CEK KEWAJARAN
# ==========================================
submission_rows = []
print("\nHasil Prediksi (Setelah di-Inverse Log):")

for month_key, preds in predictions.items():
    freq = preds['Claim_Frequency']
    sev = preds['Claim_Severity']
    
    # RUMUS: Total = Frequency * Severity
    total = freq * sev 
    
    print(f"{month_key} -> Freq: {freq:.1f} | Sev: {sev:,.0f} | Total: {total:,.0f}")
    
    submission_rows.append({'id': f'{month_key}_Claim_Frequency', 'value': freq})
    submission_rows.append({'id': f'{month_key}_Claim_Severity', 'value': sev})
    submission_rows.append({'id': f'{month_key}_Total_Claim', 'value': total})

submission_df = pd.DataFrame(submission_rows)
submission_df.to_csv('submission_log_lgbm.csv', index=False)
print("\nFile 'submission_log_lgbm.csv' berhasil dibuat! Silakan submit.")

# --- Visualisasi Evaluasi ---
hist_data = ts_data[(ts_data['Date'] >= '2024-01-01') & (ts_data['Date'] <= '2025-07-01')]
pred_df = pd.DataFrame([
    {'Date': pd.to_datetime(m.replace('_', '-') + '-01'), 'Total_Claim': predictions[m]['Claim_Frequency'] * predictions[m]['Claim_Severity']} 
    for m in predictions.keys()
])

plt.figure(figsize=(10, 5))
plt.plot(hist_data['Date'], hist_data['Total_Claim'], marker='o', label='Historis 2024-2025')
plt.plot(pred_df['Date'], pred_df['Total_Claim'], marker='s', color='red', linestyle='--', label='Prediksi (Log-Transformed)')
plt.title('Proyeksi Total Klaim dengan Log-Transform')
plt.xlabel('Bulan')
plt.ylabel('Total Klaim (Rupiah)')
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.tight_layout()
plt.savefig('plot_log_transform.png')