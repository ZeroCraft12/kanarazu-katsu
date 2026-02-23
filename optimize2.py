import pandas as pd #SKOR 6,9
import numpy as np
import lightgbm as lgb
import xgboost as xgb
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# ==========================================
# 1. PERSIAPAN DATA KLAIM & POLIS
# ==========================================
print("Membaca data...")
df_klaim = pd.read_csv('Data_Klaim.csv')
df_polis = pd.read_csv('Data_Polis.csv')

# Preprocessing Data Klaim
df_klaim['Tanggal Pasien Masuk RS'] = pd.to_datetime(df_klaim['Tanggal Pasien Masuk RS'])
date_col = 'Tanggal Pasien Masuk RS' 
df_klaim = df_klaim.dropna(subset=[date_col])
df_klaim['YearMonth'] = df_klaim[date_col].dt.to_period('M').astype(str)

# Outlier Capping (Winsorization) - Tetap pertahankan karena sangat efektif!
df_klaim_paid = df_klaim[df_klaim['Status Klaim'] == 'PAID'].copy()
upper_limit = df_klaim_paid['Nominal Klaim Yang Disetujui'].quantile(0.98)
df_klaim_paid['Nominal_Klaim_Capped'] = np.clip(df_klaim_paid['Nominal Klaim Yang Disetujui'], a_min=0, a_max=upper_limit)

print("Memproses Data Polis (Exposure)...")
# Preprocessing Data Polis untuk mendapatkan jumlah polis aktif per bulan
df_polis['Tanggal Efektif Polis'] = pd.to_datetime(df_polis['Tanggal Efektif Polis'].astype(str), format='%Y%m%d', errors='coerce')
df_polis = df_polis.dropna(subset=['Tanggal Efektif Polis'])
df_polis['YearMonth'] = df_polis['Tanggal Efektif Polis'].dt.to_period('M').astype(str)

# Hitung penambahan polis baru per bulan
policy_counts = df_polis.groupby('YearMonth').size().reset_index(name='New_Policies')

# Buat kerangka waktu berurutan agar tidak ada bulan yang terlewat
all_months = pd.date_range(start='2010-01-01', end='2025-12-01', freq='MS')
df_timeline = pd.DataFrame({'Date': all_months})
df_timeline['YearMonth'] = df_timeline['Date'].dt.to_period('M').astype(str)

# Gabungkan dengan timeline dan hitung KUMULATIF polis aktif
df_timeline = df_timeline.merge(policy_counts, on='YearMonth', how='left').fillna(0)
df_timeline['Cumulative_Policies'] = df_timeline['New_Policies'].cumsum()

# ==========================================
# 2. AGREGASI DATA BULANAN
# ==========================================
print("Melakukan agregasi bulanan...")
monthly_claims = df_klaim_paid.groupby('YearMonth').agg(
    Claim_Frequency=('Claim ID', 'count'),
    Total_Claim=('Nominal_Klaim_Capped', 'sum') 
).reset_index()

# Gabungkan data Klaim bulanan dengan data Polis (Timeline)
monthly_data = df_timeline.merge(monthly_claims, on='YearMonth', how='left').fillna(0)

# Filter hanya ambil dari tahun 2021 ke atas agar model fokus ke tren relevan (efek pandemi/post-pandemi)
monthly_data = monthly_data[monthly_data['Date'] >= '2021-01-01'].reset_index(drop=True)

# ==========================================
# 3. FEATURE ENGINEERING TINGKAT LANJUT
# ==========================================
def create_features(df):
    df = df.copy()
    df['Month'] = df['Date'].dt.month
    df['Quarter'] = df['Date'].dt.quarter
    
    # Lag Features (Nilai masa lalu)
    for col in ['Claim_Frequency', 'Total_Claim']:
        for i in [1, 2, 3, 6, 12]:
            df[f'{col}_lag_{i}'] = df[col].shift(i)
            
    # Rolling Median (Sangat robust terhadap outlier)
    for col in ['Claim_Frequency', 'Total_Claim']:
        df[f'{col}_roll_median_3'] = df[col].rolling(window=3).median()
        df[f'{col}_roll_median_6'] = df[col].rolling(window=6).median()
        
    return df

ts_data = create_features(monthly_data)

# ==========================================
# 4. MODELING DENGAN ENSEMBLE (LGBM + XGBoost)
# ==========================================
# Kita langsung memprediksi Total_Claim dan Claim_Frequency!
targets = ['Claim_Frequency', 'Total_Claim']
predictions = {'2025_08': {}, '2025_09': {}, '2025_10': {}, '2025_11': {}, '2025_12': {}}
months_to_predict = ['2025-08-01', '2025-09-01', '2025-10-01', '2025-11-01', '2025-12-01']

current_ts_data = ts_data.copy()
features = [c for c in current_ts_data.columns if c not in ['YearMonth', 'Date', 'Claim_Frequency', 'Total_Claim', 'New_Policies']]

print("Melatih model Ensemble (XGBoost + LightGBM)...")

for target in targets:
    for pred_date in months_to_predict:
        # Train data adalah bulan-bulan yang memiliki target non-zero (historis)
        train_df = current_ts_data[(current_ts_data['Date'] < pred_date) & (current_ts_data[target] > 0)].dropna()
        
        X_train = train_df[features]
        y_train = train_df[target]
        
        # 1. Model LightGBM dengan objective MAE (lebih kebal outlier)
        model_lgb = lgb.LGBMRegressor(
            objective='mae', 
            learning_rate=0.03, 
            max_depth=5, 
            num_leaves=20, 
            n_estimators=150, 
            random_state=42, 
            verbose=-1
        )
        model_lgb.fit(X_train, y_train)
        
        # 2. Model XGBoost dengan objective absolute error
        model_xgb = xgb.XGBRegressor(
            objective='reg:absoluteerror', 
            learning_rate=0.03, 
            max_depth=4, 
            n_estimators=150, 
            random_state=42
        )
        model_xgb.fit(X_train, y_train)
        
        # Re-calculate feature untuk bulan yang diprediksi
        current_ts_data = create_features(current_ts_data)
        X_test = current_ts_data[current_ts_data['Date'] == pred_date][features]
        
        # ENSEMBLE: Rata-rata dari prediksi LGBM dan XGBoost
        pred_lgb = model_lgb.predict(X_test)[0]
        pred_xgb = model_xgb.predict(X_test)[0]
        final_pred = (pred_lgb + pred_xgb) / 2.0
        
        # Batasi prediksi agar tidak kurang dari 0
        final_pred = max(final_pred, 1.0)
        
        # Update current_ts_data agar bulan depannya bisa pakai ini sebagai fitur lag
        current_ts_data.loc[current_ts_data['Date'] == pred_date, target] = final_pred
        
        # Simpan
        month_key = pred_date[:7].replace('-', '_')
        predictions[month_key][target] = final_pred

# ==========================================
# 5. FORMATTING & VISUALISASI MATPLOTLIB
# ==========================================
submission_rows = []

for month_key, preds in predictions.items():
    freq = preds['Claim_Frequency']
    total = preds['Total_Claim']
    
    # Severity kita turunkan / hitung secara matematis
    sev = total / freq if freq > 0 else 0
    
    submission_rows.append({'id': f'{month_key}_Claim_Frequency', 'value': freq})
    submission_rows.append({'id': f'{month_key}_Claim_Severity', 'value': sev})
    submission_rows.append({'id': f'{month_key}_Total_Claim', 'value': total})

submission_df = pd.DataFrame(submission_rows)
submission_df.to_csv('submission_ensemble_super.csv', index=False)
print("\nFile 'submission_ensemble_super.csv' sukses dibuat! Langsung siap submit.")
print(submission_df.head())

# --- Visualisasi Menggunakan Matplotlib ---
print("\nMenyimpan grafik prediksi klaim ke 'plot_prediksi_total_klaim.png'...")
hist_data = current_ts_data[(current_ts_data['Date'] >= '2024-01-01') & (current_ts_data['Date'] <= '2025-07-01')]
pred_data = current_ts_data[current_ts_data['Date'] >= '2025-08-01']

plt.figure(figsize=(10, 5))
plt.plot(hist_data['Date'], hist_data['Total_Claim'], marker='o', label='Historis (Real)')
plt.plot(pred_data['Date'], pred_data['Total_Claim'], marker='s', color='red', linestyle='--', label='Prediksi (XGB+LGBM)')
plt.title('Proyeksi Total Klaim (Jan 2024 - Des 2025)')
plt.xlabel('Bulan')
plt.ylabel('Total Klaim (Rupiah)')
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.tight_layout()
plt.savefig('plot_prediksi_total_klaim.png')
print("Grafik tersimpan! Silakan cek file .png di environment-mu.")