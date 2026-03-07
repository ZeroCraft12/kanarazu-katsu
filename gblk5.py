import pandas as pd
import numpy as np
from sklearn.linear_model import BayesianRidge, LassoCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import warnings
warnings.filterwarnings('ignore')

print("Membaca Data Klaim...")
df_klaim = pd.read_csv('Data_Klaim.csv')

# ==========================================
# 1. PEMBERSIHAN & PONDASI DATA (ANTI-BOLONG)
# ==========================================
date_col = 'Tanggal Pasien Masuk RS'
df_klaim[date_col] = pd.to_datetime(df_klaim[date_col], errors='coerce')
df_klaim = df_klaim.dropna(subset=[date_col])

df_klaim_paid = df_klaim[df_klaim['Status Klaim'] == 'PAID'].copy()

# KEMBALI KE CAPPING 98.0% (Kunci Absolut Penstabil 3.6)
limit = df_klaim_paid['Nominal Klaim Yang Disetujui'].quantile(0.980)
df_klaim_paid['Nominal_Klaim_Capped'] = np.clip(df_klaim_paid['Nominal Klaim Yang Disetujui'], a_min=0, a_max=limit)

df_klaim_paid['YearMonth'] = df_klaim_paid[date_col].dt.to_period('M').astype(str)

monthly_data = df_klaim_paid.groupby('YearMonth').agg(
    Claim_Frequency=('Claim ID', 'count'),
    Total_Claim=('Nominal_Klaim_Capped', 'sum') 
).reset_index()

monthly_data['Date'] = pd.to_datetime(monthly_data['YearMonth'] + '-01')

# THE RESAMPLE FIX (Pembunuh Skor Kembar)
# Mengisi bulan-bulan yang kosong/tidak ada klaim dengan 0 agar rantai waktu sempurna!
monthly_data = monthly_data.set_index('Date').resample('MS').asfreq().fillna(0).reset_index()
monthly_data['YearMonth'] = monthly_data['Date'].dt.to_period('M').astype(str)

# ==========================================
# 2. THE 3.6 IBNR PATCH (Kembali ke Formula Juara)
# ==========================================
max_idx = len(monthly_data) - 1
last_freq = monthly_data.loc[max_idx, 'Claim_Frequency']
prev_freq = monthly_data.loc[max_idx-1, 'Claim_Frequency']

# Jika bulan terakhir drop signifikan, angkat 1.4x (Hanya ini yang terbukti sukses di skor 3.6)
if last_freq < 0.7 * prev_freq:
    print("\n[IBNR PATCH] Mengangkat data bulan terakhir yang belum lengkap...")
    monthly_data.loc[max_idx, 'Claim_Frequency'] *= 1.4
    monthly_data.loc[max_idx, 'Total_Claim'] *= 1.4

# HANYA gunakan data 2022 ke atas (Menghindari anomali Covid)
monthly_data = monthly_data[monthly_data['Date'] >= '2022-01-01'].reset_index(drop=True)

# THE BILLION SCALING
monthly_data['Total_Claim_B'] = monthly_data['Total_Claim'] / 1e9

# ==========================================
# 3. KINEMATIC FEATURES (TANPA KUADRATIK / ANTI-LEDAKAN)
# ==========================================
def create_features(df):
    df = df.copy()
    
    # Month sebagai indikator Musim bagi Random Forest
    df['Month'] = df['Date'].dt.month
    
    for col in ['Claim_Frequency', 'Total_Claim_B']:
        # Lag natural
        for i in [1, 2, 3, 6]:
            df[f'{col}_lag_{i}'] = df[col].shift(i)
        
        # Roll Mean untuk menstabilkan tren linier
        df[f'{col}_roll_mean_3'] = df[col].shift(1).rolling(window=3).mean()
        
    return df

ts_data = create_features(monthly_data)
months_to_predict = ['2025-08-01', '2025-09-01', '2025-10-01', '2025-11-01', '2025-12-01']
targets = ['Claim_Frequency', 'Total_Claim_B']

# ==========================================
# 4. THE FOREST ENSEMBLE (BEBAS STATSMODELS)
# ==========================================
print("\nMelatih Model Resampled Forest...")
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
        
        # MODEL 1: Bayesian Ridge (Pondasi Linier Stabil)
        bayes = make_pipeline(StandardScaler(), BayesianRidge()).fit(X_train, y_train)
        
        # MODEL 2: LassoCV (AI yang mencari Alpha terbaik sendiri, menghentikan skor kembar!)
        lasso = make_pipeline(StandardScaler(), LassoCV(cv=3)).fit(X_train, y_train)
        
        # MODEL 3: Random Forest (Mencegah prediksi meledak ke angka aneh dan menangkap musim)
        rf = RandomForestRegressor(n_estimators=100, max_depth=4, random_state=42).fit(X_train, y_train)
        
        pred_bayes = bayes.predict(X_test)[0]
        pred_lasso = lasso.predict(X_test)[0]
        pred_rf = rf.predict(X_test)[0]
        
        # THE GOLDEN BLEND: 40% Bayes, 30% Lasso, 30% RF
        final_val = (0.40 * pred_bayes) + (0.30 * pred_lasso) + (0.30 * pred_rf)
        
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
print("\n--- HASIL PREDIKSI (THE RESAMPLED FOREST) ---")

for month_key in final_preds.keys():
    freq = final_preds[month_key]['Claim_Frequency']
    total = final_preds[month_key]['Total_Claim']
    
    sev = total / freq if freq > 0 else 0
    
    print(f"{month_key} -> Freq: {freq:.1f} | Sev: {sev:,.0f} | Total: {total:,.0f}")
    
    submission_rows.append({'id': f'{month_key}_Claim_Frequency', 'value': freq})
    submission_rows.append({'id': f'{month_key}_Claim_Severity', 'value': sev})
    submission_rows.append({'id': f'{month_key}_Total_Claim', 'value': total})

submission_df = pd.DataFrame(submission_rows)
submission_df.to_csv('submission_resampled_forest.csv', index=False)
print("\n[LOCKED] File 'submission_resampled_forest.csv' siap!")
print("Rantai data diperbaiki. Random Forest & LassoCV aktif. Tidak akan ada ledakan dan skor kembar lagi!")