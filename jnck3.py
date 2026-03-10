import pandas as pd
import numpy as np
from sklearn.linear_model import BayesianRidge, Ridge, HuberRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
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

# Capping 98.0% (Kunci Absolut Penstabil)
limit = df_klaim_paid['Nominal Klaim Yang Disetujui'].quantile(0.980)
df_klaim_paid['Nominal_Klaim_Capped'] = np.clip(df_klaim_paid['Nominal Klaim Yang Disetujui'], a_min=0, a_max=limit)

df_klaim_paid['YearMonth'] = df_klaim_paid[date_col].dt.to_period('M').astype(str)

monthly_data = df_klaim_paid.groupby('YearMonth').agg(
    Claim_Frequency=('Claim ID', 'count'),
    Total_Claim=('Nominal_Klaim_Capped', 'sum') 
).reset_index()

monthly_data['Date'] = pd.to_datetime(monthly_data['YearMonth'] + '-01')
monthly_data = monthly_data.sort_values('Date').reset_index(drop=True)
monthly_data['Month'] = monthly_data['Date'].dt.month

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

monthly_data = monthly_data[monthly_data['Date'] >= '2022-01-01'].reset_index(drop=True)
monthly_data['Total_Claim_B'] = monthly_data['Total_Claim'] / 1e9

# ==========================================
# 3. CLASSICAL DECOMPOSITION (THE GRANDMASTER TRICK)
# ==========================================
# Menghitung Rata-rata Global
freq_global_mean = monthly_data['Claim_Frequency'].mean()
total_global_mean = monthly_data['Total_Claim_B'].mean()

# Menghitung Indeks Musiman (Berapa kali lipat bulan ini dibanding rata-rata global)
freq_idx_map = (monthly_data.groupby('Month')['Claim_Frequency'].mean() / freq_global_mean).to_dict()
total_idx_map = (monthly_data.groupby('Month')['Total_Claim_B'].mean() / total_global_mean).to_dict()

# DESEASONALIZE: Membuang efek musiman agar data menjadi tren lurus yang mulus!
monthly_data['Des_Freq'] = monthly_data['Claim_Frequency'] / monthly_data['Month'].map(freq_idx_map)
monthly_data['Des_Total_B'] = monthly_data['Total_Claim_B'] / monthly_data['Month'].map(total_idx_map)

# ==========================================
# 4. KINEMATIC FEATURES PADA DATA YANG SUDAH MULUS
# ==========================================
def create_features(df):
    df = df.copy()
    # Kunci: AI hanya akan melihat tren murni (Time_Index) tanpa perlu pusing soal Musim
    df['Time_Index'] = np.arange(1, len(df) + 1)
    
    # Lags diterapkan pada data yang sudah di-Deseasonalize
    for col in ['Des_Freq', 'Des_Total_B']:
        for i in [1, 2, 3, 6]:
            df[f'{col}_lag_{i}'] = df[col].shift(i)
        df[f'{col}_roll_mean_3'] = df[col].shift(1).rolling(window=3).mean()
        
    return df

ts_data = create_features(monthly_data)
months_to_predict = ['2025-08-01', '2025-09-01', '2025-10-01', '2025-11-01', '2025-12-01']

# Target yang ditebak AI adalah Tren Mulusnya!
des_targets = ['Des_Freq', 'Des_Total_B']

# ==========================================
# 5. THE PURE TREND ENSEMBLE
# ==========================================
print("\nMelatih Model Rekursif (The Classical Decomposition)...")
current_ts_data = ts_data.copy()
final_preds = {'2025_08': {}, '2025_09': {}, '2025_10': {}, '2025_11': {}, '2025_12': {}}

for i, pred_date in enumerate(months_to_predict):
    
    if not (current_ts_data['Date'] == pred_date).any():
        new_row = pd.DataFrame({'Date': [pd.to_datetime(pred_date)]})
        new_row['Month'] = pd.to_datetime(pred_date).month
        current_ts_data = pd.concat([current_ts_data, new_row], ignore_index=True)
        
    temp_df = create_features(current_ts_data)
    
    exclude_cols = ['YearMonth', 'Date', 'Month', 'Claim_Frequency', 'Total_Claim', 'Total_Claim_B', 'Des_Freq', 'Des_Total_B']
    features = [c for c in temp_df.columns if c not in exclude_cols]
    
    for des_target in des_targets:
        train_df = temp_df[temp_df['Date'] < pred_date].dropna()
        X_train = train_df[features]
        y_train = train_df[des_target] # Latih AI HANYA pada tren yang mulus
        
        X_test = temp_df[temp_df['Date'] == pred_date][features].fillna(0)
        
        # 3 Mesin Linear Penarik Garis Terbaik
        bayes = make_pipeline(StandardScaler(), BayesianRidge()).fit(X_train, y_train)
        ridge = make_pipeline(StandardScaler(), Ridge(alpha=1.0)).fit(X_train, y_train)
        huber = make_pipeline(StandardScaler(), HuberRegressor(epsilon=1.35)).fit(X_train, y_train)
        
        pred_bayes = bayes.predict(X_test)[0]
        pred_ridge = ridge.predict(X_test)[0]
        pred_huber = huber.predict(X_test)[0]
        
        # BLEND EMAS (40/30/30) PADA GARIS TREN MULUS
        pred_des_val = (0.40 * pred_bayes) + (0.30 * pred_ridge) + (0.30 * pred_huber)
        
        # Simpan tebakan Tren Mulus untuk iterasi (Lag) bulan depan
        current_ts_data.loc[current_ts_data['Date'] == pred_date, des_target] = pred_des_val
        
        # ========================================================
        # THE MAGIC: RESEASONALIZE (Mengembalikan Efek Musiman Asli)
        # ========================================================
        pred_month = pd.to_datetime(pred_date).month
        
        if des_target == 'Des_Freq':
            # Kalikan Tren Mulus dengan Indeks Musiman Frekuensi
            actual_val = pred_des_val * freq_idx_map.get(pred_month, 1.0)
            current_ts_data.loc[current_ts_data['Date'] == pred_date, 'Claim_Frequency'] = actual_val
            target_name = 'Claim_Frequency'
        else:
            # Kalikan Tren Mulus dengan Indeks Musiman Total Biaya
            actual_val_b = pred_des_val * total_idx_map.get(pred_month, 1.0)
            current_ts_data.loc[current_ts_data['Date'] == pred_date, 'Total_Claim_B'] = actual_val_b
            actual_val = actual_val_b * 1e9 # Kembalikan ke miliaran
            target_name = 'Total_Claim'
            
        month_key = pred_date[:7].replace('-', '_')
        final_preds[month_key][target_name] = actual_val

# ==========================================
# 6. EXPORT HASIL
# ==========================================
submission_rows = []
print("\n--- HASIL PREDIKSI (THE CLASSICAL DECOMPOSITION) ---")

for month_key in final_preds.keys():
    freq = final_preds[month_key]['Claim_Frequency']
    total = final_preds[month_key]['Total_Claim']
    
    sev = total / freq if freq > 0 else 0
    
    print(f"{month_key} -> Freq: {freq:.1f} | Sev: {sev:,.0f} | Total: {total:,.0f}")
    
    submission_rows.append({'id': f'{month_key}_Claim_Frequency', 'value': freq})
    submission_rows.append({'id': f'{month_key}_Claim_Severity', 'value': sev})
    submission_rows.append({'id': f'{month_key}_Total_Claim', 'value': total})

submission_df = pd.DataFrame(submission_rows)
submission_df.to_csv('submission_classical_decomposition.csv', index=False)
print("\n[LOCKED] File 'submission_classical_decomposition.csv' siap!")
print("Musiman di ekstrak paksa, Tren di prediksi mulus, Musiman dikembalikan. INI ADALAH APEX! GAS < 3.0!")    