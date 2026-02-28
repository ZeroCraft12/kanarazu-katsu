import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

# Coba muat CatBoost (Sangat direkomendasikan untuk Kaggle tabular)
try:
    from catboost import CatBoostRegressor
    USE_CATBOOST = True
    print("[SYSTEM] CatBoost Activating...")
except ImportError:
    USE_CATBOOST = False
    print("[SYSTEM] CatBoost not found, fallback to XGBoost.")

print("Membaca Data V2 (Mathematical Challenge Festival ITB 2026-2)...")
df_klaim = pd.read_csv('Data_Klaim.csv')

# ==========================================
# 1. PREPROCESSING (ADAPTED FOR V2 DATA)
# ==========================================
date_col = 'Tanggal Pasien Masuk RS'
df_klaim[date_col] = pd.to_datetime(df_klaim[date_col], errors='coerce')
df_klaim = df_klaim.dropna(subset=[date_col])

df_klaim_paid = df_klaim[df_klaim['Status Klaim'] == 'PAID'].copy()

# Capping di 98.5% (Optimal sweet-spot untuk mencegah 1-2 pasien merusak model)
upper_limit = df_klaim_paid['Nominal Klaim Yang Disetujui'].quantile(0.985)
df_klaim_paid['Nominal_Klaim_Capped'] = np.clip(df_klaim_paid['Nominal Klaim Yang Disetujui'], a_min=0, a_max=upper_limit)

df_klaim_paid['YearMonth'] = df_klaim_paid[date_col].dt.to_period('M').astype(str)

monthly_data = df_klaim_paid.groupby('YearMonth').agg(
    Claim_Frequency=('Claim ID', 'count'),
    Total_Claim=('Nominal_Klaim_Capped', 'sum') 
).reset_index()

monthly_data['Date'] = pd.to_datetime(monthly_data['YearMonth'] + '-01')
monthly_data = monthly_data.sort_values('Date').reset_index(drop=True)

# Hitung Severity aktual historis
monthly_data['Claim_Severity'] = monthly_data['Total_Claim'] / monthly_data['Claim_Frequency']

# ---------------------------------------------------------
# SMART IBNR SMOOTHING (Mencegah Anjlok di Bulan Terakhir)
# ---------------------------------------------------------
max_idx = len(monthly_data) - 1
last_freq = monthly_data.loc[max_idx, 'Claim_Frequency']
prev_freq = monthly_data.loc[max_idx - 1, 'Claim_Frequency']

if last_freq < 0.8 * prev_freq:
    print(f"\n[CRITICAL] Bulan terakhir ({monthly_data.loc[max_idx, 'YearMonth']}) terdeteksi anjlok (Incomplete Data).")
    print("Menerapkan Smart IBNR Smoothing: Menyamakan dengan rata-rata 3 bulan terakhir yang stabil...")
    
    # Ambil rata-rata 3 bulan sebelum bulan terakhir
    avg_freq_3m = monthly_data['Claim_Frequency'].iloc[-4:-1].mean()
    avg_sev_3m = monthly_data['Claim_Severity'].iloc[-4:-1].mean()
    
    monthly_data.loc[max_idx, 'Claim_Frequency'] = avg_freq_3m
    monthly_data.loc[max_idx, 'Claim_Severity'] = avg_sev_3m
    monthly_data.loc[max_idx, 'Total_Claim'] = avg_freq_3m * avg_sev_3m

# Fokus pada data pasca-Covid yang stabil
monthly_data = monthly_data[monthly_data['Date'] >= '2022-01-01'].reset_index(drop=True)

# ---------------------------------------------------------
# THE GOLDEN SECRET: LOG-TRANSFORM FREQ & SEVERITY
# ---------------------------------------------------------
# Kita TIDAK memprediksi Total Klaim. Kita memprediksi Log Frekuensi dan Log Severity.
# Ini adalah metode paling kebal terhadap perubahan data dari panitia.
monthly_data['Log_Freq'] = np.log1p(monthly_data['Claim_Frequency'])
monthly_data['Log_Sev'] = np.log1p(monthly_data['Claim_Severity'])

# ==========================================
# 2. HEAVY FEATURE ENGINEERING
# ==========================================
def create_features(df):
    df = df.copy()
    df['Month'] = df['Date'].dt.month
    df['Quarter'] = df['Date'].dt.quarter
    
    for col in ['Log_Freq', 'Log_Sev']:
        # Lags historis
        for i in [1, 2, 3, 6, 12]:
            df[f'{col}_lag_{i}'] = df[col].shift(i)
        
        # Exponential Moving Average (Sangat peka terhadap tren terbaru)
        df[f'{col}_ewma_3'] = df[col].shift(1).ewm(span=3, adjust=False).mean()
        df[f'{col}_ewma_6'] = df[col].shift(1).ewm(span=6, adjust=False).mean()
        
    return df

ts_data = create_features(monthly_data)
months_to_predict = ['2025-08-01', '2025-09-01', '2025-10-01', '2025-11-01', '2025-12-01']
targets = ['Log_Freq', 'Log_Sev'] # PREDIKSI LOG TARGET!

# ==========================================
# 3. DIAMOND ENSEMBLE PREDICTION (LGBM + CATBOOST)
# ==========================================
print("\nMelatih Model dengan Log-Target (CatBoost + LightGBM)...")
final_predictions = {'2025_08': {}, '2025_09': {}, '2025_10': {}, '2025_11': {}, '2025_12': {}}
current_ts_data = ts_data.copy()

# Singkirkan kolom yang tidak boleh dibaca AI
exclude_cols = ['YearMonth', 'Date', 'Claim_Frequency', 'Claim_Severity', 'Total_Claim', 'Log_Freq', 'Log_Sev']
features = [c for c in current_ts_data.columns if c not in exclude_cols]

for target in targets:
    for pred_date in months_to_predict:
        train_df = current_ts_data[current_ts_data['Date'] < pred_date].dropna()
        X_train = train_df[features].fillna(0)
        y_train = train_df[target]
        
        # Model 1: LightGBM
        m_lgb = lgb.LGBMRegressor(objective='rmse', learning_rate=0.03, max_depth=3, n_estimators=150, random_state=42, verbose=-1)
        m_lgb.fit(X_train, y_train)
        
        # Model 2: CatBoost atau XGBoost
        if USE_CATBOOST:
            m_cb = CatBoostRegressor(iterations=200, learning_rate=0.03, depth=3, random_seed=42, verbose=0)
            m_cb.fit(X_train, y_train)
        else:
            m_cb = xgb.XGBRegressor(objective='reg:squarederror', learning_rate=0.03, max_depth=3, n_estimators=150, random_state=42)
            m_cb.fit(X_train, y_train)
            
        # Siapkan baris prediksi
        if not (current_ts_data['Date'] == pred_date).any():
            new_row = pd.DataFrame({'Date': [pd.to_datetime(pred_date)]})
            current_ts_data = pd.concat([current_ts_data, new_row], ignore_index=True)
            
        temp_ts = create_features(current_ts_data)
        X_test = temp_ts[temp_ts['Date'] == pred_date][features].fillna(0)
        
        # Prediksi Log (Ensemble 50:50)
        pred_log_val = (0.5 * m_lgb.predict(X_test)[0]) + (0.5 * m_cb.predict(X_test)[0])
        current_ts_data.loc[current_ts_data['Date'] == pred_date, target] = pred_log_val
        
        # INVERSE TRANSFORM: Kembalikan dari skala Logaritma ke angka asli Miliaran
        original_value = np.expm1(pred_log_val)
        
        target_name = 'Claim_Frequency' if target == 'Log_Freq' else 'Claim_Severity'
        month_key = pred_date[:7].replace('-', '_')
        final_predictions[month_key][target_name] = original_value

# ==========================================
# 4. EXPORT & DECOUPLING CALCULATION
# ==========================================
submission_rows = []
print("\n--- HASIL PREDIKSI (DIAMOND ACTUARIAL) ---")
print("Catatan: Total Klaim dihitung dari Freq x Severity secara terpisah. Paling aman dari goncangan data baru.")

# Kita TIDAK menambah decay_rate buatan manusia. Kita biarkan data berbicara murni.
for month_key in final_predictions.keys():
    freq = final_predictions[month_key]['Claim_Frequency']
    sev = final_predictions[month_key]['Claim_Severity']
    
    # RUMUS MUTLAK AKTUARIA
    total = freq * sev
    
    print(f"{month_key} -> Freq: {freq:.1f} | Sev: {sev:,.0f} | Total: {total:,.0f}")
    
    submission_rows.append({'id': f'{month_key}_Claim_Frequency', 'value': freq})
    submission_rows.append({'id': f'{month_key}_Claim_Severity', 'value': sev})
    submission_rows.append({'id': f'{month_key}_Total_Claim', 'value': total})

submission_df = pd.DataFrame(submission_rows)
submission_df.to_csv('submission_diamond_actuarial.csv', index=False)
print("\n[LOCKED & LOADED] File 'submission_diamond_actuarial.csv' berhasil dibuat!")
print("Ini adalah adaptasi terbaik untuk data Versi 2 dari panitia.")