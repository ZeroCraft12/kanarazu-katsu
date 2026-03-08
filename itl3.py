import pandas as pd
import numpy as np
from sklearn.linear_model import BayesianRidge, Ridge
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, DotProduct
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

print("Membaca Data Klaim...")
df_klaim = pd.read_csv('Data_Klaim.csv')

# ==========================================
# 1. PONDASI DATA (BASE 3.6 YANG SOLID)
# ==========================================
date_col = 'Tanggal Pasien Masuk RS'
df_klaim[date_col] = pd.to_datetime(df_klaim[date_col], errors='coerce')
df_klaim = df_klaim.dropna(subset=[date_col])

df_klaim_paid = df_klaim[df_klaim['Status Klaim'] == 'PAID'].copy()

# Capping 98.0%
limit = df_klaim_paid['Nominal Klaim Yang Disetujui'].quantile(0.980)
df_klaim_paid['Nominal_Klaim_Capped'] = np.clip(df_klaim_paid['Nominal Klaim Yang Disetujui'], a_min=0, a_max=limit)

df_klaim_paid['YearMonth'] = df_klaim_paid[date_col].dt.to_period('M').astype(str)

monthly_data = df_klaim_paid.groupby('YearMonth').agg(
    Claim_Frequency=('Claim ID', 'count'),
    Total_Claim=('Nominal_Klaim_Capped', 'sum') 
).reset_index()

monthly_data['Date'] = pd.to_datetime(monthly_data['YearMonth'] + '-01')
monthly_data = monthly_data.sort_values('Date').reset_index(drop=True)

# ==========================================
# 2. PELONTAR IBNR 1.4x
# ==========================================
max_idx = len(monthly_data) - 1
last_freq = monthly_data.loc[max_idx, 'Claim_Frequency']
prev_freq = monthly_data.loc[max_idx-1, 'Claim_Frequency']

if last_freq < 0.7 * prev_freq:
    print("\n[IBNR 1.4x] Mengangkat data bulan terakhir sebagai launchpad...")
    monthly_data.loc[max_idx, 'Claim_Frequency'] *= 1.4
    monthly_data.loc[max_idx, 'Total_Claim'] *= 1.4

monthly_data = monthly_data[monthly_data['Date'] >= '2022-01-01'].reset_index(drop=True)
monthly_data['Total_Claim_B'] = monthly_data['Total_Claim'] / 1e9

# ==========================================
# 3. FOURIER & HEAVY KINEMATIC FEATURES
# ==========================================
def create_features(df):
    df = df.copy()
    
    # Tren Makro
    df['Time_Index'] = np.arange(1, len(df) + 1)
    
    # Gelombang Fourier (Kunci akurasi musiman tingkat tinggi)
    months_numeric = df['Date'].dt.month
    df['sin_M_1'] = np.sin(2 * np.pi * months_numeric / 12)
    df['cos_M_1'] = np.cos(2 * np.pi * months_numeric / 12)
    df['sin_M_2'] = np.sin(4 * np.pi * months_numeric / 12) # Harmoni kedua
    df['cos_M_2'] = np.cos(4 * np.pi * months_numeric / 12) # Harmoni kedua
    
    for col in ['Claim_Frequency', 'Total_Claim_B']:
        for i in [1, 2, 3, 6]:
            df[f'{col}_lag_{i}'] = df[col].shift(i)
        
        # Multiple Rolling Means untuk mendeteksi momentum mikro dan makro
        df[f'{col}_roll_mean_3'] = df[col].shift(1).rolling(window=3).mean()
        df[f'{col}_roll_mean_6'] = df[col].shift(1).rolling(window=6).mean()
        
    return df

ts_data = create_features(monthly_data)
months_to_predict = ['2025-08-01', '2025-09-01', '2025-10-01', '2025-11-01', '2025-12-01']
targets = ['Claim_Frequency', 'Total_Claim_B']

# ==========================================
# 4. THE HEAVYWEIGHT DYNAMIC ENSEMBLE
# ==========================================
print("\nMelatih Model Berat (Gaussian Process, Gradient Boosting, Dynamic Ensembling)...")
print("Proses ini akan memakan waktu komputasi yang lebih lama karena evaluasi dinamis. Harap tunggu...\n")

current_ts_data = ts_data.copy()
final_preds = {'2025_08': {}, '2025_09': {}, '2025_10': {}, '2025_11': {}, '2025_12': {}}

# Latih Baseline Holt-Winters Damped
hw_preds_dict = {}
for target in targets:
    hw_data = monthly_data[target].values
    try:
        hw_fit = ExponentialSmoothing(hw_data, trend='add', damped_trend=True, initialization_method='estimated').fit(optimized=True)
        hw_preds_dict[target] = hw_fit.forecast(steps=5)
    except:
        hw_preds_dict[target] = np.repeat(hw_data[-1], 5)

# Setup Kernels untuk Gaussian Process (Super Heavy)
kernel = DotProduct() + WhiteKernel(noise_level=0.5)

for i, pred_date in enumerate(months_to_predict):
    print(f"[*] Melakukan kalkulasi matriks berat untuk: {pred_date[:7]}")
    
    if not (current_ts_data['Date'] == pred_date).any():
        new_row = pd.DataFrame({'Date': [pd.to_datetime(pred_date)]})
        current_ts_data = pd.concat([current_ts_data, new_row], ignore_index=True)
        
    temp_df = create_features(current_ts_data)
    
    exclude_cols = ['YearMonth', 'Date', 'Claim_Frequency', 'Total_Claim', 'Total_Claim_B']
    features = [c for c in temp_df.columns if c not in exclude_cols]
    
    for target in targets:
        # Menyiapkan data historis
        train_df_full = temp_df[temp_df['Date'] < pred_date].dropna()
        X_train_full = train_df_full[features]
        y_train_full = train_df_full[target]
        X_test = temp_df[temp_df['Date'] == pred_date][features].fillna(0)
        
        # =======================================================
        # DYNAMIC WEIGHT ALLOCATION (The Game Changer)
        # =======================================================
        # AI membelah data latih untuk simulasi mundurnya waktu 3 bulan ke belakang
        if len(train_df_full) > 6:
            val_size = 3
            X_train_sim = X_train_full.iloc[:-val_size]
            y_train_sim = y_train_full.iloc[:-val_size]
            X_val_sim = X_train_full.iloc[-val_size:]
            y_val_sim = y_train_full.iloc[-val_size:]
            
            # Simulasi Model
            sim_bayes = make_pipeline(StandardScaler(), BayesianRidge()).fit(X_train_sim, y_train_sim)
            sim_gbr = make_pipeline(StandardScaler(), GradientBoostingRegressor(loss='huber', n_estimators=100, max_depth=3, random_state=42)).fit(X_train_sim, y_train_sim)
            sim_gp = make_pipeline(StandardScaler(), GaussianProcessRegressor(kernel=kernel, alpha=0.1, n_restarts_optimizer=3)).fit(X_train_sim, y_train_sim)
            
            # Hitung Error
            err_bayes = mean_absolute_error(y_val_sim, sim_bayes.predict(X_val_sim))
            err_gbr = mean_absolute_error(y_val_sim, sim_gbr.predict(X_val_sim))
            err_gp = mean_absolute_error(y_val_sim, sim_gp.predict(X_val_sim))
            
            # Ubah error menjadi bobot (semakin kecil error, semakin besar bobot)
            inv_errs = np.array([1/(err_bayes+1e-6), 1/(err_gbr+1e-6), 1/(err_gp+1e-6)])
            weights = inv_errs / np.sum(inv_errs)
        else:
            # Jika data terlalu sedikit, bobot dibagi rata
            weights = [0.4, 0.3, 0.3] 

        # =======================================================
        # PREDIKSI MASA DEPAN DENGAN FULL DATA
        # =======================================================
        model_bayes = make_pipeline(StandardScaler(), BayesianRidge()).fit(X_train_full, y_train_full)
        model_gbr = make_pipeline(StandardScaler(), GradientBoostingRegressor(loss='huber', n_estimators=100, max_depth=3, random_state=42)).fit(X_train_full, y_train_full)
        model_gp = make_pipeline(StandardScaler(), GaussianProcessRegressor(kernel=kernel, alpha=0.1, n_restarts_optimizer=3)).fit(X_train_full, y_train_full)
        
        pred_bayes = model_bayes.predict(X_test)[0]
        pred_gbr = model_gbr.predict(X_test)[0]
        pred_gp = model_gp.predict(X_test)[0]
        pred_hw = hw_preds_dict[target][i]
        
        # BLENDING DINAMIS: ML Dinamis (80%) + Baseline Holt-Winters (20%)
        # AI secara mandiri menentukan seberapa besar porsi Bayes, GBR, dan GP setiap bulannya!
        ml_blend = (weights[0] * pred_bayes) + (weights[1] * pred_gbr) + (weights[2] * pred_gp)
        final_val = (0.80 * ml_blend) + (0.20 * pred_hw)
        
        current_ts_data.loc[current_ts_data['Date'] == pred_date, target] = final_val
        
        target_name = 'Claim_Frequency' if target == 'Claim_Frequency' else 'Total_Claim'
        if target == 'Total_Claim_B':
            final_val = final_val * 1e9
            
        month_key = pred_date[:7].replace('-', '_')
        final_preds[month_key][target_name] = final_val

# ==========================================
# 5. EXPORT HASIL
# ==========================================
submission_rows = []
print("\n--- HASIL PREDIKSI (HEAVYWEIGHT DYNAMIC APEX) ---")

for month_key in final_preds.keys():
    freq = final_preds[month_key]['Claim_Frequency']
    total = final_preds[month_key]['Total_Claim']
    
    sev = total / freq if freq > 0 else 0
    
    print(f"{month_key} -> Freq: {freq:.1f} | Sev: {sev:,.0f} | Total: {total:,.0f}")
    
    submission_rows.append({'id': f'{month_key}_Claim_Frequency', 'value': freq})
    submission_rows.append({'id': f'{month_key}_Claim_Severity', 'value': sev})
    submission_rows.append({'id': f'{month_key}_Total_Claim', 'value': total})

submission_df = pd.DataFrame(submission_rows)
submission_df.to_csv('submission_heavyweight_apex.csv', index=False)
print("\n[LOCKED] File 'submission_heavyweight_apex.csv' siap!")
print("Algoritma Gaussian Process & Bobot Dinamis berhasil dieksekusi. Tidak ada lagi tebakan kaku! Bismillah < 3.0!")