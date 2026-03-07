import pandas as pd
import numpy as np
from sklearn.linear_model import BayesianRidge, RidgeCV, HuberRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import warnings
warnings.filterwarnings('ignore')

print("Membaca Data Klaim...")
df_klaim = pd.read_csv('Data_Klaim.csv')

# ==========================================
# 1. PONDASI DATA (KEMBALI KE BASE 3.6 YANG SOLID)
# ==========================================
# Kita buang interpolasi dan filter "> 0" yang ternyata malah menambah noise di skor 4.x
date_col = 'Tanggal Pasien Masuk RS'
df_klaim[date_col] = pd.to_datetime(df_klaim[date_col], errors='coerce')
df_klaim = df_klaim.dropna(subset=[date_col])

df_klaim_paid = df_klaim[df_klaim['Status Klaim'] == 'PAID'].copy()

# Capping 98.0% (Kunci Absolut Penstabil Skor 3.6)
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
# 2. RESTORASI THE 1.4x LAUNCHPAD (Wajib!)
# ==========================================
max_idx = len(monthly_data) - 1
last_freq = monthly_data.loc[max_idx, 'Claim_Frequency']
prev_freq = monthly_data.loc[max_idx-1, 'Claim_Frequency']

# IBNR 1.4x: Batu loncatan paling pas agar AI tidak under-predict di akhir tahun
if last_freq < 0.7 * prev_freq:
    print("\n[IBNR 1.4x RESTORED] Mengangkat data bulan terakhir sebagai launchpad...")
    monthly_data.loc[max_idx, 'Claim_Frequency'] *= 1.4
    monthly_data.loc[max_idx, 'Total_Claim'] *= 1.4

# HANYA gunakan data 2022 ke atas
monthly_data = monthly_data[monthly_data['Date'] >= '2022-01-01'].reset_index(drop=True)

# THE BILLION SCALING
monthly_data['Total_Claim_B'] = monthly_data['Total_Claim'] / 1e9

# ==========================================
# 3. KINEMATIC & CATEGORICAL FEATURES (THE MAGIC!)
# ==========================================
def create_features(df):
    df = df.copy()
    
    # KUNCI 1: Time Index untuk Tren Makro (Garis miring naik)
    df['Time_Index'] = np.arange(1, len(df) + 1)
    
    # KUNCI 2: ONE-HOT ENCODING MONTHS (Solusi Musiman Tanpa Holt-Winters!)
    # Alih-alih angka 1-12, kita buat 12 kolom saklar. 
    # AI akan menghafal karakter spesifik setiap bulan secara mandiri!
    for m in range(1, 13):
        df[f'Is_Month_{m}'] = (df['Date'].dt.month == m).astype(int)
    
    for col in ['Claim_Frequency', 'Total_Claim_B']:
        # Lags yang sudah terbukti
        for i in [1, 2, 3, 6]:
            df[f'{col}_lag_{i}'] = df[col].shift(i)
        df[f'{col}_roll_mean_3'] = df[col].shift(1).rolling(window=3).mean()
        
    return df

ts_data = create_features(monthly_data)
months_to_predict = ['2025-08-01', '2025-09-01', '2025-10-01', '2025-11-01', '2025-12-01']
targets = ['Claim_Frequency', 'Total_Claim_B']

# ==========================================
# 4. THE PURE MACHINE LEARNING ENSEMBLE
# ==========================================
print("\nMelatih Model Rekursif (The OHE Actuary)...")
current_ts_data = ts_data.copy()
final_preds = {'2025_08': {}, '2025_09': {}, '2025_10': {}, '2025_11': {}, '2025_12': {}}
        
for i, pred_date in enumerate(months_to_predict):
    
    if not (current_ts_data['Date'] == pred_date).any():
        new_row = pd.DataFrame({'Date': [pd.to_datetime(pred_date)]})
        current_ts_data = pd.concat([current_ts_data, new_row], ignore_index=True)
        
    temp_df = create_features(current_ts_data)
    
    exclude_cols = ['YearMonth', 'Date', 'Claim_Frequency', 'Total_Claim', 'Total_Claim_B']
    features = [c for c in temp_df.columns if c not in exclude_cols]
    
    for target in targets:
        train_df = temp_df[temp_df['Date'] < pred_date].dropna()
        X_train = train_df[features]
        y_train = train_df[target]
        
        X_test = temp_df[temp_df['Date'] == pred_date][features].fillna(0)
        
        # 3 MESIN LINEAR TERKUAT SCIKIT-LEARN (Bebas error statsmodels!)
        bayes = make_pipeline(StandardScaler(), BayesianRidge()).fit(X_train, y_train)
        ridge = make_pipeline(StandardScaler(), RidgeCV(alphas=[0.1, 1.0, 5.0])).fit(X_train, y_train)
        huber = make_pipeline(StandardScaler(), HuberRegressor(epsilon=1.35)).fit(X_train, y_train)
        
        pred_bayes = bayes.predict(X_test)[0]
        pred_ridge = ridge.predict(X_test)[0]
        pred_huber = huber.predict(X_test)[0]
        
        # BLEND: 40% Bayes, 30% Ridge, 30% Huber
        final_val = (0.40 * pred_bayes) + (0.30 * pred_ridge) + (0.30 * pred_huber)
        
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
print("\n--- HASIL PREDIKSI (THE OHE ACTUARY) ---")

for month_key in final_preds.keys():
    freq = final_preds[month_key]['Claim_Frequency']
    total = final_preds[month_key]['Total_Claim']
    
    sev = total / freq if freq > 0 else 0
    
    print(f"{month_key} -> Freq: {freq:.1f} | Sev: {sev:,.0f} | Total: {total:,.0f}")
    
    submission_rows.append({'id': f'{month_key}_Claim_Frequency', 'value': freq})
    submission_rows.append({'id': f'{month_key}_Claim_Severity', 'value': sev})
    submission_rows.append({'id': f'{month_key}_Total_Claim', 'value': total})

submission_df = pd.DataFrame(submission_rows)
submission_df.to_csv('submission_ohe_actuary.csv', index=False)
print("\n[LOCKED] File 'submission_ohe_actuary.csv' siap!")
print("One-Hot Encoding berhasil mengubah model linear biasa menjadi AI Super-Seasonal! Gas < 3.0!")