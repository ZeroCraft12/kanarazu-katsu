import pandas as pd
import numpy as np
from sklearn.linear_model import BayesianRidge, Ridge
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import warnings
warnings.filterwarnings('ignore')

print("Membaca Data Klaim...")
df_klaim = pd.read_csv('Data_Klaim.csv')

# ==========================================
# 1. PONDASI DATA (PEMBERSIHAN SEMPURNA)
# ==========================================
date_col = 'Tanggal Pasien Masuk RS'
df_klaim[date_col] = pd.to_datetime(df_klaim[date_col], errors='coerce')
df_klaim = df_klaim.dropna(subset=[date_col])

df_klaim_paid = df_klaim[df_klaim['Status Klaim'] == 'PAID'].copy()

# Capping di 98.5% (Memberi sedikit ruang agar XGBoost bisa menangkap spike ekstrem)
limit = df_klaim_paid['Nominal Klaim Yang Disetujui'].quantile(0.985)
df_klaim_paid['Nominal_Klaim_Capped'] = np.clip(df_klaim_paid['Nominal Klaim Yang Disetujui'], a_min=0, a_max=limit)

df_klaim_paid['YearMonth'] = df_klaim_paid[date_col].dt.to_period('M').astype(str)

monthly_data = df_klaim_paid.groupby('YearMonth').agg(
    Claim_Frequency=('Claim ID', 'count'),
    Total_Claim=('Nominal_Klaim_Capped', 'sum') 
).reset_index()

monthly_data['Date'] = pd.to_datetime(monthly_data['YearMonth'] + '-01')

# INTERPOLASI: Mengisi bulan bolong dengan garis mulus agar model tidak kebingungan
monthly_data = monthly_data.drop(columns=['YearMonth']).set_index('Date').resample('MS').interpolate(method='linear').reset_index()

# HANYA gunakan data 2022 ke atas
monthly_data = monthly_data[monthly_data['Date'] >= '2022-01-01'].reset_index(drop=True)

# ==========================================
# 2. IBNR LAUNCHPAD (Pelontar Akhir Tahun)
# ==========================================
max_idx = len(monthly_data) - 1
last_freq = monthly_data.loc[max_idx, 'Claim_Frequency']
prev_freq = monthly_data.loc[max_idx-1, 'Claim_Frequency']

if last_freq < 0.7 * prev_freq:
    print("\n[IBNR 1.4x AKTIF] Mengangkat data bulan terakhir sebagai batu loncatan...")
    monthly_data.loc[max_idx, 'Claim_Frequency'] *= 1.4
    monthly_data.loc[max_idx, 'Total_Claim'] *= 1.4

monthly_data['Total_Claim_B'] = monthly_data['Total_Claim'] / 1e9

# ==========================================
# 3. KINEMATIC FEATURES
# ==========================================
def create_features(df):
    df = df.copy()
    
    # Time Index untuk Regresi Linier
    df['Time_Index'] = np.arange(1, len(df) + 1)
    # Month untuk XGBoost
    df['Month'] = df['Date'].dt.month
    
    for col in ['Claim_Frequency', 'Total_Claim_B']:
        for i in [1, 2, 3]:
            df[f'{col}_lag_{i}'] = df[col].shift(i)
        df[f'{col}_roll_mean_3'] = df[col].shift(1).rolling(window=3).mean()
        
    return df

ts_data = create_features(monthly_data)
months_to_predict = ['2025-08-01', '2025-09-01', '2025-10-01', '2025-11-01', '2025-12-01']
targets = ['Claim_Frequency', 'Total_Claim_B']

# ==========================================
# 4. THE RESIDUAL BOOST ENSEMBLE (GRANDMASTER LEVEL)
# ==========================================
print("\nMelatih Model Rekursif (Linear-Tree Residual Boosting)...")
current_ts_data = ts_data.copy()
final_preds = {'2025_08': {}, '2025_09': {}, '2025_10': {}, '2025_11': {}, '2025_12': {}}

for i, pred_date in enumerate(months_to_predict):
    
    if not (current_ts_data['Date'] == pred_date).any():
        new_row = pd.DataFrame({'Date': [pd.to_datetime(pred_date)]})
        current_ts_data = pd.concat([current_ts_data, new_row], ignore_index=True)
        
    temp_df = create_features(current_ts_data)
    
    for target in targets:
        train_df = temp_df[temp_df['Date'] < pred_date].dropna()
        X_test = temp_df[temp_df['Date'] == pred_date].fillna(0)
        
        # -------------------------------------------------------------
        # TAHAP 1: PONDASI LINIER (Makro Tren)
        # -------------------------------------------------------------
        lin_features = ['Time_Index', f'{target}_lag_1', f'{target}_lag_2']
        
        model_lin = make_pipeline(StandardScaler(), BayesianRidge())
        model_lin.fit(train_df[lin_features], train_df[target])
        
        train_pred_lin = model_lin.predict(train_df[lin_features])
        test_pred_lin = model_lin.predict(X_test[lin_features])[0]
        
        # -------------------------------------------------------------
        # TAHAP 2: XGBOOST RESIDUALS (Menangkap Musiman & Anomali)
        # -------------------------------------------------------------
        # Hitung sisa error dari model linier
        residuals = train_df[target] - train_pred_lin
        
        xgb_features = ['Month', f'{target}_lag_1', f'{target}_lag_2', f'{target}_roll_mean_3']
        
        # XGBoost dilatih HANYA untuk menebak error/residualnya!
        model_xgb = xgb.XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.05, objective='reg:squarederror', random_state=42)
        model_xgb.fit(train_df[xgb_features], residuals)
        
        test_pred_xgb_residual = model_xgb.predict(X_test[xgb_features])[0]
        
        # THE HYBRID PREDICTION (Linier + Sisa Error)
        pred_hybrid = test_pred_lin + test_pred_xgb_residual
        
        # -------------------------------------------------------------
        # TAHAP 3: ANCHOR REGRESSION (Pengaman Ekstrem)
        # -------------------------------------------------------------
        # Kita kombinasikan dengan Ridge murni agar tebakannya punya jangkar/anchor yang stabil
        all_features = lin_features + ['Month', f'{target}_roll_mean_3']
        model_anchor = make_pipeline(StandardScaler(), Ridge(alpha=2.0))
        model_anchor.fit(train_df[all_features], train_df[target])
        pred_anchor = model_anchor.predict(X_test[all_features])[0]
        
        # FINAL BLEND: 60% Hybrid Tingkat Dewa + 40% Anchor Stabilisator
        final_val = (0.60 * pred_hybrid) + (0.40 * pred_anchor)
        
        current_ts_data.loc[current_ts_data['Date'] == pred_date, target] = final_val
        
        if target == 'Total_Claim_B':
            final_val_real = final_val * 1e9
            target_name = 'Total_Claim'
        else:
            final_val_real = final_val
            target_name = 'Claim_Frequency'
            
        month_key = pred_date[:7].replace('-', '_')
        final_preds[month_key][target_name] = final_val_real

# ==========================================
# 5. EXPORT HASIL
# ==========================================
submission_rows = []
print("\n--- HASIL PREDIKSI (THE RESIDUAL BOOST APEX) ---")

for month_key in final_preds.keys():
    freq = final_preds[month_key]['Claim_Frequency']
    total = final_preds[month_key]['Total_Claim']
    
    sev = total / freq if freq > 0 else 0
    
    print(f"{month_key} -> Freq: {freq:.1f} | Sev: {sev:,.0f} | Total: {total:,.0f}")
    
    submission_rows.append({'id': f'{month_key}_Claim_Frequency', 'value': freq})
    submission_rows.append({'id': f'{month_key}_Claim_Severity', 'value': sev})
    submission_rows.append({'id': f'{month_key}_Total_Claim', 'value': total})

submission_df = pd.DataFrame(submission_rows)
submission_df.to_csv('submission_residual_boost.csv', index=False)
print("\n[LOCKED] File 'submission_residual_boost.csv' siap!")
print("Linear memprediksi Tren, XGBoost memprediksi Sisa Error. Ini adalah formasi absolut. GAS < 3.0!")