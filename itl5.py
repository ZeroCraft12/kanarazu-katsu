import pandas as pd
import numpy as np
from sklearn.linear_model import BayesianRidge
import xgboost as xgb
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import warnings
warnings.filterwarnings('ignore')

print("Membaca Data Klaim...")
df_klaim = pd.read_csv('Data_Klaim.csv')

# ==========================================
# 1. PONDASI DATA (KEMBALI KE BASE 3.6 YANG SOLID)
# ==========================================
date_col = 'Tanggal Pasien Masuk RS'
df_klaim[date_col] = pd.to_datetime(df_klaim[date_col], errors='coerce')
df_klaim = df_klaim.dropna(subset=[date_col])

df_klaim_paid = df_klaim[df_klaim['Status Klaim'] == 'PAID'].copy()

# Capping 98.0% (Kunci Absolut Penstabil 3.6)
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
# 2. PELONTAR IBNR 1.4x (WAJIB)
# ==========================================
max_idx = len(monthly_data) - 1
last_freq = monthly_data.loc[max_idx, 'Claim_Frequency']
prev_freq = monthly_data.loc[max_idx-1, 'Claim_Frequency']

if last_freq < 0.7 * prev_freq:
    print("\n[IBNR 1.4x] Mengangkat data bulan terakhir sebagai launchpad...")
    monthly_data.loc[max_idx, 'Claim_Frequency'] *= 1.4
    monthly_data.loc[max_idx, 'Total_Claim'] *= 1.4

# HANYA gunakan data Pasca-Covid (2022+)
monthly_data = monthly_data[monthly_data['Date'] >= '2022-01-01'].reset_index(drop=True)

# THE BILLION SCALING
monthly_data['Total_Claim_B'] = monthly_data['Total_Claim'] / 1e9

# ==========================================
# 3. THE LOG-TRANSFORMATION (RAHASIA AKTUARIS)
# ==========================================
monthly_data['Log_Freq'] = np.log1p(monthly_data['Claim_Frequency'])
monthly_data['Log_Total_B'] = np.log1p(monthly_data['Total_Claim_B'])

# ==========================================
# 4. FITUR MINIMALIS (PADA ALAM LOGARITMA)
# ==========================================
def create_features(df):
    df = df.copy()
    
    # Tren Makro
    df['Time_Index'] = np.arange(1, len(df) + 1)
    df['Month'] = df['Date'].dt.month
    
    # Lag dari target yang sudah di-Log!
    for col in ['Log_Freq', 'Log_Total_B']:
        for i in [1, 2, 3]:
            df[f'{col}_lag_{i}'] = df[col].shift(i)
        
        # Momentum rata-rata 3 bulan
        df[f'{col}_roll_mean_3'] = df[col].shift(1).rolling(window=3).mean()
        
    return df

ts_data = create_features(monthly_data)
months_to_predict = ['2025-08-01', '2025-09-01', '2025-10-01', '2025-11-01', '2025-12-01']
log_targets = ['Log_Freq', 'Log_Total_B']

# ==========================================
# 5. THE GRADIENT TITAN ENSEMBLE
# ==========================================
print("\nMelatih Model Rekursif (The Gradient Titan - XGBoost & LightGBM)...")
current_ts_data = ts_data.copy()
final_preds = {'2025_08': {}, '2025_09': {}, '2025_10': {}, '2025_11': {}, '2025_12': {}}

for i, pred_date in enumerate(months_to_predict):
    
    if not (current_ts_data['Date'] == pred_date).any():
        new_row = pd.DataFrame({'Date': [pd.to_datetime(pred_date)]})
        current_ts_data = pd.concat([current_ts_data, new_row], ignore_index=True)
        
    temp_df = create_features(current_ts_data)
    
    exclude_cols = ['YearMonth', 'Date', 'Claim_Frequency', 'Total_Claim', 'Total_Claim_B', 'Log_Freq', 'Log_Total_B']
    features = [c for c in temp_df.columns if c not in exclude_cols]
    
    for log_target in log_targets:
        train_df = temp_df[temp_df['Date'] < pred_date].dropna()
        X_train = train_df[features]
        y_train = train_df[log_target]
        
        X_test = temp_df[temp_df['Date'] == pred_date][features].fillna(0)
        
        # 1. Bayesian Ridge (Stabilisator Linier)
        bayes = make_pipeline(StandardScaler(), BayesianRidge()).fit(X_train, y_train)
        
        # 2. XGBoost (Sangat kuat menangani pola non-linear)
        xg_reg = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.05, max_depth=3, random_state=42)
        xg_reg.fit(X_train, y_train)
        
        # 3. LightGBM (Cepat dan efisien, bagus untuk data kecil jika di-tune dengan benar)
        lgb_reg = lgb.LGBMRegressor(n_estimators=100, learning_rate=0.05, max_depth=3, random_state=42, min_child_samples=5)
        lgb_reg.fit(X_train, y_train)
        
        pred_bayes = bayes.predict(X_test)[0]
        pred_xgb = xg_reg.predict(X_test)[0]
        pred_lgb = lgb_reg.predict(X_test)[0]
        
        # BLEND: 40% Linear (Bayes), 30% XGBoost, 30% LightGBM
        final_log_val = (0.40 * pred_bayes) + (0.30 * pred_xgb) + (0.30 * pred_lgb)
        
        current_ts_data.loc[current_ts_data['Date'] == pred_date, log_target] = final_log_val
        
        actual_val = np.expm1(final_log_val)
        
        if log_target == 'Log_Freq':
            current_ts_data.loc[current_ts_data['Date'] == pred_date, 'Claim_Frequency'] = actual_val
            target_name = 'Claim_Frequency'
        else:
            current_ts_data.loc[current_ts_data['Date'] == pred_date, 'Total_Claim_B'] = actual_val
            target_name = 'Total_Claim'
            actual_val = actual_val * 1e9 
            
        month_key = pred_date[:7].replace('-', '_')
        final_preds[month_key][target_name] = actual_val

# ==========================================
# 6. EXPORT HASIL
# ==========================================
submission_rows = []
print("\n--- HASIL PREDIKSI (THE GRADIENT TITAN) ---")

for month_key in final_preds.keys():
    freq = final_preds[month_key]['Claim_Frequency']
    total = final_preds[month_key]['Total_Claim']
    
    sev = total / freq if freq > 0 else 0
    
    print(f"{month_key} -> Freq: {freq:.1f} | Sev: {sev:,.0f} | Total: {total:,.0f}")
    
    submission_rows.append({'id': f'{month_key}_Claim_Frequency', 'value': freq})
    submission_rows.append({'id': f'{month_key}_Claim_Severity', 'value': sev})
    submission_rows.append({'id': f'{month_key}_Total_Claim', 'value': total})

submission_df = pd.DataFrame(submission_rows)
submission_df.to_csv('submission_gradient_titan.csv', index=False)
print("\n[LOCKED] File 'submission_gradient_titan.csv' siap!")
print("XGBoost dan LightGBM telah mengambil alih! Gas < 3.0!")