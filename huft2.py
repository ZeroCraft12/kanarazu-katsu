import pandas as pd
import numpy as np
from sklearn.linear_model import BayesianRidge, LassoCV, RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import warnings
warnings.filterwarnings('ignore')

print("Membaca Data Klaim...")
df_klaim = pd.read_csv('Data_Klaim.csv')

# ==========================================
# 1. PEMBERSIHAN DATA & INTERPOLASI
# ==========================================
date_col = 'Tanggal Pasien Masuk RS'
df_klaim[date_col] = pd.to_datetime(df_klaim[date_col], errors='coerce')
df_klaim = df_klaim.dropna(subset=[date_col])

df_klaim_paid = df_klaim[df_klaim['Status Klaim'] == 'PAID'].copy()

# Buang data <= 0 (refund/salah input) yang merusak rata-rata
df_klaim_paid = df_klaim_paid[df_klaim_paid['Nominal Klaim Yang Disetujui'] > 0]

# Capping 98.0% (Kunci Absolut Penstabil 3.6)
limit = df_klaim_paid['Nominal Klaim Yang Disetujui'].quantile(0.980)
df_klaim_paid['Nominal_Klaim_Capped'] = np.clip(df_klaim_paid['Nominal Klaim Yang Disetujui'], a_min=0, a_max=limit)

df_klaim_paid['YearMonth'] = df_klaim_paid[date_col].dt.to_period('M').astype(str)

monthly_data = df_klaim_paid.groupby('YearMonth').agg(
    Claim_Frequency=('Claim ID', 'count'),
    Total_Claim=('Nominal_Klaim_Capped', 'sum') 
).reset_index()

monthly_data['Date'] = pd.to_datetime(monthly_data['YearMonth'] + '-01')

# THE INTERPOLATION FIX (Menjaga Rantai Waktu Tetap Utuh)
monthly_data = monthly_data.drop(columns=['YearMonth']).set_index('Date').resample('MS').interpolate(method='linear').reset_index()
monthly_data['YearMonth'] = monthly_data['Date'].dt.to_period('M').astype(str)

# HANYA gunakan data 2022 ke atas
monthly_data = monthly_data[monthly_data['Date'] >= '2022-01-01'].reset_index(drop=True)

# ==========================================
# 2. RESTORASI THE 1.4x LAUNCHPAD
# ==========================================
max_idx = len(monthly_data) - 1
last_freq = monthly_data.loc[max_idx, 'Claim_Frequency']
prev_freq = monthly_data.loc[max_idx-1, 'Claim_Frequency']

# KEMBALI KE FORMULA JUARA 3.6!
# Rata-rata 3 bulan terbukti terlalu rendah. Kita butuh 1.4x sebagai batu loncatan akhir tahun.
if last_freq < 0.8 * prev_freq:
    print("\n[IBNR 1.4x RESTORED] Mengangkat data bulan terakhir sebagai launchpad Q3/Q4...")
    monthly_data.loc[max_idx, 'Claim_Frequency'] *= 1.4
    monthly_data.loc[max_idx, 'Total_Claim'] *= 1.4

# THE BILLION SCALING
monthly_data['Total_Claim_B'] = monthly_data['Total_Claim'] / 1e9

# ==========================================
# 3. THE DAMPED & STAIRCASE FEATURES
# ==========================================
def create_features(df):
    df = df.copy()
    
    # KUNCI 1: Damped Trend (Menggantikan fungsi Holt-Winters Damped)
    # Membuat AI menarik garis tren yang melandai (logaritmik), tidak tembus langit.
    time_index = np.arange(1, len(df) + 1)
    df['Log_Time'] = np.log1p(time_index)
    
    # KUNCI 2: Staircase Seasonality (Menggantikan fitur Month 1-12)
    # Memberikan sinyal instan ke AI bahwa bulan-bulan ini adalah "Musim Tinggi"
    df['is_Q3'] = df['Date'].dt.month.isin([7, 8, 9]).astype(int)
    df['is_Q4'] = df['Date'].dt.month.isin([10, 11, 12]).astype(int)
    
    for col in ['Claim_Frequency', 'Total_Claim_B']:
        for i in [1, 2, 3, 6]:
            df[f'{col}_lag_{i}'] = df[col].shift(i)
        df[f'{col}_roll_mean_3'] = df[col].shift(1).rolling(window=3).mean()
        
    return df

ts_data = create_features(monthly_data)
months_to_predict = ['2025-08-01', '2025-09-01', '2025-10-01', '2025-11-01', '2025-12-01']
targets = ['Claim_Frequency', 'Total_Claim_B']

# ==========================================
# 4. CROSS-VALIDATED ENSEMBLE (BEBAS KEMBAR)
# ==========================================
print("\nMelatih Model Cross-Validated (The Damped Actuary)...")
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
        
        # 3 Algoritma Regresi Papan Atas Scikit-Learn
        bayes = make_pipeline(StandardScaler(), BayesianRidge()).fit(X_train, y_train)
        ridge_cv = make_pipeline(StandardScaler(), RidgeCV(alphas=[0.1, 1.0, 5.0, 10.0])).fit(X_train, y_train)
        lasso_cv = make_pipeline(StandardScaler(), LassoCV(cv=3)).fit(X_train, y_train)
        
        pred_bayes = bayes.predict(X_test)[0]
        pred_ridge = ridge_cv.predict(X_test)[0]
        pred_lasso = lasso_cv.predict(X_test)[0]
        
        # BLEND 40/30/30 
        final_val = (0.40 * pred_bayes) + (0.30 * pred_ridge) + (0.30 * pred_lasso)
        
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
print("\n--- HASIL PREDIKSI (THE DAMPED ACTUARY) ---")

for month_key in final_preds.keys():
    freq = final_preds[month_key]['Claim_Frequency']
    total = final_preds[month_key]['Total_Claim']
    
    sev = total / freq if freq > 0 else 0
    
    print(f"{month_key} -> Freq: {freq:.1f} | Sev: {sev:,.0f} | Total: {total:,.0f}")
    
    submission_rows.append({'id': f'{month_key}_Claim_Frequency', 'value': freq})
    submission_rows.append({'id': f'{month_key}_Claim_Severity', 'value': sev})
    submission_rows.append({'id': f'{month_key}_Total_Claim', 'value': total})

submission_df = pd.DataFrame(submission_rows)
submission_df.to_csv('submission_damped_actuary.csv', index=False)
print("\n[LOCKED] File 'submission_damped_actuary.csv' siap!")
print("Launchpad 1.4x dipulihkan. Fitur Log_Time & Kuartal aktif. Gas skor < 3.0!")