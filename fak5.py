import pandas as pd
import numpy as np
from sklearn.linear_model import PoissonRegressor, GammaRegressor, BayesianRidge
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import warnings
warnings.filterwarnings('ignore')

print("Membaca Data Klaim...")
df_klaim = pd.read_csv('Data_Klaim.csv')

# ==========================================
# 1. PONDASI DATA (ERA NORMAL 2022+)
# ==========================================
date_col = 'Tanggal Pasien Masuk RS'
df_klaim[date_col] = pd.to_datetime(df_klaim[date_col], errors='coerce')
df_klaim = df_klaim.dropna(subset=[date_col])

df_klaim_paid = df_klaim[df_klaim['Status Klaim'] == 'PAID'].copy()

# Capping di 98.0% (Pahlawan penstabil data)
limit = df_klaim_paid['Nominal Klaim Yang Disetujui'].quantile(0.980)
df_klaim_paid['Nominal_Klaim_Capped'] = np.clip(df_klaim_paid['Nominal Klaim Yang Disetujui'], a_min=0, a_max=limit)

df_klaim_paid['YearMonth'] = df_klaim_paid[date_col].dt.to_period('M').astype(str)

monthly_data = df_klaim_paid.groupby('YearMonth').agg(
    Claim_Frequency=('Claim ID', 'count'),
    Total_Claim=('Nominal_Klaim_Capped', 'sum') 
).reset_index()

monthly_data['Date'] = pd.to_datetime(monthly_data['YearMonth'] + '-01')
monthly_data = monthly_data.sort_values('Date').reset_index(drop=True)

# HANYA gunakan data 2022 ke atas (Bebas anomali Covid)
monthly_data = monthly_data[monthly_data['Date'] >= '2022-01-01'].reset_index(drop=True)

# ==========================================
# 2. TARGET ACTUARIAL MURNI
# ==========================================
# Di algoritma GLM, kita memprediksi Frekuensi dan Severity (dalam Jutaan)
monthly_data['Severity_M'] = (monthly_data['Total_Claim'] / monthly_data['Claim_Frequency']) / 1e6

# ==========================================
# 3. GLM FEATURE ENGINEERING
# ==========================================
def create_features(df):
    df = df.copy()
    
    # Logaritma Waktu (Menarik garis tren ke masa depan secara perlahan/logaritmik)
    df['Time_Index'] = np.arange(1, len(df) + 1)
    df['Log_Time'] = np.log1p(df['Time_Index'])
    
    # Gelombang Musiman Trigonometri
    months = df['Date'].dt.month
    df['sin_M'] = np.sin(2 * np.pi * months / 12)
    df['cos_M'] = np.cos(2 * np.pi * months / 12)
    
    for col in ['Claim_Frequency', 'Severity_M']:
        # Lags jangka pendek (1, 2, 3) + Roll Mean
        for i in [1, 2, 3]:
            df[f'{col}_lag_{i}'] = df[col].shift(i)
        
        df[f'{col}_roll_mean_3'] = df[col].shift(1).rolling(window=3).mean()
        
    return df

ts_data = create_features(monthly_data)
months_to_predict = ['2025-08-01', '2025-09-01', '2025-10-01', '2025-11-01', '2025-12-01']
targets = ['Claim_Frequency', 'Severity_M']

# ==========================================
# 4. GLM ACTUARIAL ENSEMBLE TRAINING
# ==========================================
print("\nMelatih Model GLM Actuarial (Poisson & Gamma)...")
current_ts_data = ts_data.copy()
final_preds = {'2025_08': {}, '2025_09': {}, '2025_10': {}, '2025_11': {}, '2025_12': {}}

# Latih Holt-Winters dengan PENGAMAN ERROR
hw_preds_dict = {}
for target in targets:
    hw_data = monthly_data[target].values
    try:
        # Jika data cukup, pakai Seasonal
        hw_fit = ExponentialSmoothing(hw_data, trend='add', seasonal='add', seasonal_periods=12, damped_trend=True, initialization_method='estimated').fit(optimized=True)
        hw_preds_dict[target] = hw_fit.forecast(steps=5)
    except Exception as e:
        try:
            # Jika data kurang dari 24 bulan, otomatis fallback ke Non-Seasonal agar tidak crash
            hw_fit = ExponentialSmoothing(hw_data, trend='add', damped_trend=True, initialization_method='estimated').fit(optimized=True)
            hw_preds_dict[target] = hw_fit.forecast(steps=5)
        except:
            # Pengaman absolut jika algoritma statsmodels gagal total
            hw_preds_dict[target] = np.repeat(hw_data[-1], 5)
        
# Ekstrapolasi Masa Depan
for i, pred_date in enumerate(months_to_predict):
    
    if not (current_ts_data['Date'] == pred_date).any():
        new_row = pd.DataFrame({'Date': [pd.to_datetime(pred_date)]})
        current_ts_data = pd.concat([current_ts_data, new_row], ignore_index=True)
        
    temp_df = create_features(current_ts_data)
    
    exclude_cols = ['YearMonth', 'Date', 'Claim_Frequency', 'Total_Claim', 'Severity_M']
    features = [c for c in temp_df.columns if c not in exclude_cols]
    
    for target in targets:
        train_df = temp_df[temp_df['Date'] < pred_date].dropna()
        X_train = train_df[features]
        y_train = train_df[target]
        
        X_test = temp_df[temp_df['Date'] == pred_date][features].fillna(0)
        
        # PENGGUNAAN STANDARD SCALER + PIPELINE (Mencegah error bias)
        bayes = make_pipeline(StandardScaler(), BayesianRidge()).fit(X_train, y_train)
        
        # === THE GLM MAGIC ===
        if target == 'Claim_Frequency':
            # POISSON Regression untuk menebak JUMLAH ORANG SAKIT (Count Data)
            glm = make_pipeline(StandardScaler(), PoissonRegressor(alpha=0.5, max_iter=1000)).fit(X_train, y_train)
        else:
            # GAMMA Regression untuk menebak NILAI UANG (Continuous, Positif, Skewed)
            glm = make_pipeline(StandardScaler(), GammaRegressor(alpha=0.5, max_iter=1000)).fit(X_train, y_train)
        
        pred_bayes = bayes.predict(X_test)[0]
        pred_glm = glm.predict(X_test)[0]
        pred_hw = hw_preds_dict[target][i]
        
        # THE NEW APEX BLEND: GLM mendominasi 50% karena ini adalah ahlinya asuransi
        final_val = (0.50 * pred_glm) + (0.30 * pred_bayes) + (0.20 * pred_hw)
        
        current_ts_data.loc[current_ts_data['Date'] == pred_date, target] = final_val
        
        month_key = pred_date[:7].replace('-', '_')
        final_preds[month_key][target] = final_val

# ==========================================
# 5. EXPORT HASIL
# ==========================================
submission_rows = []
print("\n--- HASIL PREDIKSI (THE GLM APEX) ---")

for month_key in final_preds.keys():
    freq = final_preds[month_key]['Claim_Frequency']
    sev_m = final_preds[month_key]['Severity_M']
    
    # Rekalkulasi ke Uang Riil (Miliaran)
    sev = sev_m * 1e6
    total = freq * sev
    
    print(f"{month_key} -> Freq: {freq:.1f} | Sev: {sev:,.0f} | Total: {total:,.0f}")
    
    submission_rows.append({'id': f'{month_key}_Claim_Frequency', 'value': freq})
    submission_rows.append({'id': f'{month_key}_Claim_Severity', 'value': sev})
    submission_rows.append({'id': f'{month_key}_Total_Claim', 'value': total})

submission_df = pd.DataFrame(submission_rows)
submission_df.to_csv('submission_glm_apex_final.csv', index=False)
print("\n[LOCKED] File 'submission_glm_apex_final.csv' siap!")
print("Ini adalah algoritma Industri Asuransi Murni. Tidak akan ada lagi skor kembar! Gas!")