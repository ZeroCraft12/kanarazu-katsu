import pandas as pd
import numpy as np
import lightgbm as lgb
from prophet import Prophet
import warnings
warnings.filterwarnings('ignore')

print("Membaca data...")
df_klaim = pd.read_csv('Data_Klaim.csv')

# ==========================================
# 1. PREPROCESSING & MISSING PUZZLE FIX (IBNR)
# ==========================================
# Kembali menggunakan Tanggal Pasien Masuk RS karena ini menghasilkan skor terbaikmu (6.2)
date_col = 'Tanggal Pasien Masuk RS'
df_klaim[date_col] = pd.to_datetime(df_klaim[date_col], errors='coerce')
df_klaim = df_klaim.dropna(subset=[date_col])

df_klaim_paid = df_klaim[df_klaim['Status Klaim'] == 'PAID'].copy()

# Capping Outlier (Persentil 98)
upper_limit = df_klaim_paid['Nominal Klaim Yang Disetujui'].quantile(0.98)
df_klaim_paid['Nominal_Klaim_Capped'] = np.clip(df_klaim_paid['Nominal Klaim Yang Disetujui'], a_min=0, a_max=upper_limit)

df_klaim_paid['YearMonth'] = df_klaim_paid[date_col].dt.to_period('M').astype(str)

# Agregasi
monthly_data = df_klaim_paid.groupby('YearMonth').agg(
    Claim_Frequency=('Claim ID', 'count'),
    Total_Claim=('Nominal_Klaim_Capped', 'sum') 
).reset_index()

monthly_data['Date'] = pd.to_datetime(monthly_data['YearMonth'] + '-01')
monthly_data = monthly_data.sort_values('Date').reset_index(drop=True)

# ---------------------------------------------------------
# THE MISSING PUZZLE: BUANG BULAN TERAKHIR (JEBAKAN IBNR)
# ---------------------------------------------------------
max_date = monthly_data['Date'].max()
print(f"Bulan terakhir di data: {max_date.strftime('%Y-%m')}")
print("Membuang bulan terakhir dari data latih karena data klaimnya pasti belum lengkap (IBNR Trap)...")

# Potong 1 bulan terakhir & Potong data sebelum 2022
monthly_data = monthly_data[(monthly_data['Date'] >= '2022-01-01') & (monthly_data['Date'] < max_date)].reset_index(drop=True)


# ==========================================
# 2. MODEL 1: LIGHTGBM (Sang Juara Skor 6.2)
# ==========================================
print("\nMelatih Model 1: LightGBM...")
def create_features(df):
    df = df.copy()
    df['Month'] = df['Date'].dt.month
    for col in ['Claim_Frequency', 'Total_Claim']:
        for i in [1, 2, 3, 6, 12]:
            df[f'{col}_lag_{i}'] = df[col].shift(i)
        df[f'{col}_roll_mean_3'] = df[col].shift(1).rolling(window=3).mean()
    return df

ts_data = create_features(monthly_data)
months_to_predict = ['2025-08-01', '2025-09-01', '2025-10-01', '2025-11-01', '2025-12-01']
targets = ['Claim_Frequency', 'Total_Claim']

lgbm_preds = {'2025_08': {}, '2025_09': {}, '2025_10': {}, '2025_11': {}, '2025_12': {}}
current_ts_data = ts_data.copy()
features = [c for c in current_ts_data.columns if c not in ['YearMonth', 'Date', 'Claim_Frequency', 'Total_Claim']]

for target in targets:
    for pred_date in months_to_predict:
        train_df = current_ts_data[current_ts_data['Date'] < pred_date].dropna()
        X_train, y_train = train_df[features], train_df[target]
        
        # LGBM params dari iterasi terbaik
        model = lgb.LGBMRegressor(objective='rmse', learning_rate=0.05, max_depth=4, n_estimators=100, random_state=42, verbose=-1)
        model.fit(X_train, y_train)
        
        if not (current_ts_data['Date'] == pred_date).any():
            new_row = pd.DataFrame({'Date': [pd.to_datetime(pred_date)]})
            current_ts_data = pd.concat([current_ts_data, new_row], ignore_index=True)
            
        current_ts_data = create_features(current_ts_data)
        X_test = current_ts_data[current_ts_data['Date'] == pred_date][features]
        
        pred_val = model.predict(X_test)[0]
        current_ts_data.loc[current_ts_data['Date'] == pred_date, target] = pred_val
        
        month_key = pred_date[:7].replace('-', '_')
        lgbm_preds[month_key][target] = pred_val


# ==========================================
# 3. MODEL 2: META PROPHET (Sang Stabilizer Skor 6.6)
# ==========================================
print("Melatih Model 2: Meta Prophet...")
prophet_preds = {'2025_08': {}, '2025_09': {}, '2025_10': {}, '2025_11': {}, '2025_12': {}}

for target in targets:
    prophet_df = monthly_data[['Date', target]].rename(columns={'Date': 'ds', target: 'y'})
    
    m = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False, changepoint_prior_scale=0.05)
    m.fit(prophet_df)
    
    future = pd.DataFrame({'ds': pd.to_datetime(months_to_predict)})
    forecast = m.predict(future)
    
    for _, row in forecast.iterrows():
        month_key = row['ds'].strftime('%Y_%m')
        prophet_preds[month_key][target] = row['yhat']


# ==========================================
# 4. THE GRANDMASTER BLEND (ENSEMBLE)
# ==========================================
print("\n--- MENGGABUNGKAN MODEL (BLENDING) ---")
submission_rows = []

for month_key in lgbm_preds.keys():
    # KUNCI JUARA: 60% LGBM (karena skor aslinya lebih bagus) + 40% Prophet
    freq_lgb = lgbm_preds[month_key]['Claim_Frequency']
    freq_pro = prophet_preds[month_key]['Claim_Frequency']
    final_freq = (0.6 * freq_lgb) + (0.4 * freq_pro)
    
    total_lgb = lgbm_preds[month_key]['Total_Claim']
    total_pro = prophet_preds[month_key]['Total_Claim']
    final_total = (0.6 * total_lgb) + (0.4 * total_pro)
    
    # Severity selalu dihitung belakangan (Total / Frekuensi)
    final_sev = final_total / final_freq if final_freq > 0 else 0
    
    print(f"{month_key} -> Freq: {final_freq:.1f} | Sev: {final_sev:,.0f} | Total: {final_total:,.0f}")
    
    submission_rows.append({'id': f'{month_key}_Claim_Frequency', 'value': final_freq})
    submission_rows.append({'id': f'{month_key}_Claim_Severity', 'value': final_sev})
    submission_rows.append({'id': f'{month_key}_Total_Claim', 'value': final_total})

submission_df = pd.DataFrame(submission_rows)
submission_df.to_csv('submission_ultimate_blend.csv', index=False)
print("\nFile 'submission_ultimate_blend.csv' berhasil dibuat! Siap Submit!")