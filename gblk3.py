import pandas as pd
import numpy as np
from sklearn.linear_model import BayesianRidge
from sklearn.ensemble import GradientBoostingRegressor
from statsmodels.tsa.arima.model import ARIMA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import warnings
warnings.filterwarnings('ignore')

print("Membaca Data Klaim...")
df_klaim = pd.read_csv('Data_Klaim.csv')

# ==========================================
# 1. PEMBERSIHAN DATA TINGKAT LANJUT
# ==========================================
date_col = 'Tanggal Pasien Masuk RS'
df_klaim[date_col] = pd.to_datetime(df_klaim[date_col], errors='coerce')
df_klaim = df_klaim.dropna(subset=[date_col])

df_klaim_paid = df_klaim[df_klaim['Status Klaim'] == 'PAID'].copy()

# EXTRA CLEANING: Buang nilai negatif/0 (Refund/Data Batal) yang sering merusak Mean
df_klaim_paid = df_klaim_paid[df_klaim_paid['Nominal Klaim Yang Disetujui'] > 0]

# Capping di 98.0% (Kunci Absolut Penstabil Data)
limit = df_klaim_paid['Nominal Klaim Yang Disetujui'].quantile(0.980)
df_klaim_paid['Nominal_Klaim_Capped'] = np.clip(df_klaim_paid['Nominal Klaim Yang Disetujui'], a_min=0, a_max=limit)

df_klaim_paid['YearMonth'] = df_klaim_paid[date_col].dt.to_period('M').astype(str)

monthly_data = df_klaim_paid.groupby('YearMonth').agg(
    Claim_Frequency=('Claim ID', 'count'),
    Total_Claim=('Nominal_Klaim_Capped', 'sum') 
).reset_index()

monthly_data['Date'] = pd.to_datetime(monthly_data['YearMonth'] + '-01')
monthly_data = monthly_data.sort_values('Date').reset_index(drop=True)

# HANYA gunakan data 2022 ke atas (Menghindari anomali Covid 2021)
monthly_data = monthly_data[monthly_data['Date'] >= '2022-01-01'].reset_index(drop=True)

# THE BILLION SCALING
monthly_data['Total_Claim_B'] = monthly_data['Total_Claim'] / 1e9

# ==========================================
# 2. FEATURE ENGINEERING (DISEDERHANAKAN AGAR TIDAK BOCOR)
# ==========================================
def create_features(df):
    df = df.copy()
    
    # Time Index dan Month
    df['Time_Index'] = np.arange(1, len(df) + 1)
    df['Month'] = df['Date'].dt.month
    
    for col in ['Claim_Frequency', 'Total_Claim_B']:
        # Hanya gunakan Lag 1 dan 3 untuk menghindari Multicollinearity berlebih
        for i in [1, 3]:
            df[f'{col}_lag_{i}'] = df[col].shift(i)
        
        # Roll Mean untuk menstabilkan tren
        df[f'{col}_roll_mean_3'] = df[col].shift(1).rolling(window=3).mean()
        
    return df

ts_data = create_features(monthly_data)
months_to_predict = ['2025-08-01', '2025-09-01', '2025-10-01', '2025-11-01', '2025-12-01']
targets = ['Claim_Frequency', 'Total_Claim_B']

# ==========================================
# 3. THE ARIMA & BOOSTING CONVERGENCE
# ==========================================
print("\nMelatih Model Rekursif (ARIMA & Gradient Boosting)...")
current_ts_data = ts_data.copy()
final_preds = {'2025_08': {}, '2025_09': {}, '2025_10': {}, '2025_11': {}, '2025_12': {}}

# 1. Latih Model ARIMA (Menggantikan Holt-Winters)
arima_preds_dict = {}
for target in targets:
    arima_data = monthly_data[target].values
    try:
        # ARIMA(1,1,1): AutoRegressive=1, Integrated(Differencing)=1, MovingAverage=1
        arima_fit = ARIMA(arima_data, order=(1, 1, 1)).fit()
        arima_preds_dict[target] = arima_fit.forecast(steps=5)
    except:
        # Fallback aman jika terjadi masalah matriks
        arima_preds_dict[target] = np.repeat(arima_data[-1], 5)
        
# 2. Latih Regresi & Boosting secara bertahap
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
        
        # Model 1: Bayesian Ridge (Standardisasi Wajib)
        bayes = make_pipeline(StandardScaler(), BayesianRidge()).fit(X_train, y_train)
        
        # Model 2: Gradient Boosting (Huber Loss - Sangat tahan terhadap Outlier ekstrem)
        # Menggunakan pohon keputusan, jadi algoritmanya dijamin berbeda 100% dari regresi linier biasa
        gbr = GradientBoostingRegressor(loss='huber', n_estimators=50, max_depth=2, random_state=42).fit(X_train, y_train)
        
        pred_bayes = bayes.predict(X_test)[0]
        pred_gbr = gbr.predict(X_test)[0]
        pred_arima = arima_preds_dict[target][i]
        
        # BLENDING BARU: 40% ARIMA (Tren Makro), 40% Bayes (Tren Linier), 20% GBR (Lonjakan non-linier akhir tahun)
        final_val = (0.40 * pred_arima) + (0.40 * pred_bayes) + (0.20 * pred_gbr)
        
        current_ts_data.loc[current_ts_data['Date'] == pred_date, target] = final_val
        
        target_name = 'Claim_Frequency' if target == 'Claim_Frequency' else 'Total_Claim'
        if target == 'Total_Claim_B':
            final_val = final_val * 1e9
            
        month_key = pred_date[:7].replace('-', '_')
        final_preds[month_key][target_name] = final_val

# ==========================================
# 4. EXPORT HASIL
# ==========================================
submission_rows = []
print("\n--- HASIL PREDIKSI (ARIMA & BOOSTING CONVERGENCE) ---")

for month_key in final_preds.keys():
    freq = final_preds[month_key]['Claim_Frequency']
    total = final_preds[month_key]['Total_Claim']
    
    sev = total / freq if freq > 0 else 0
    
    print(f"{month_key} -> Freq: {freq:.1f} | Sev: {sev:,.0f} | Total: {total:,.0f}")
    
    submission_rows.append({'id': f'{month_key}_Claim_Frequency', 'value': freq})
    submission_rows.append({'id': f'{month_key}_Claim_Severity', 'value': sev})
    submission_rows.append({'id': f'{month_key}_Total_Claim', 'value': total})

submission_df = pd.DataFrame(submission_rows)
submission_df.to_csv('submission_arima_boosting.csv', index=False)
print("\n[LOCKED] File 'submission_arima_boosting.csv' siap!")
print("Mesin telah dirombak ke ARIMA dan Gradient Boosting. Dijamin angka baru dan lebih tajam!")