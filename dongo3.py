import pandas as pd
import numpy as np
from sklearn.linear_model import BayesianRidge, Ridge
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import warnings
warnings.filterwarnings('ignore')

print("Membaca Data Klaim...")
df_klaim = pd.read_csv('Data_Klaim.csv')

# ==========================================
# 1. PONDASI DATA (KEMBALI KE RAW SCALE 3.6)
# ==========================================
date_col = 'Tanggal Pasien Masuk RS'
df_klaim[date_col] = pd.to_datetime(df_klaim[date_col], errors='coerce')
df_klaim = df_klaim.dropna(subset=[date_col])

df_klaim_paid = df_klaim[df_klaim['Status Klaim'] == 'PAID'].copy()

# Capping di 98.0% (Pahlawan Mutlak Skor 3.6)
limit = df_klaim_paid['Nominal Klaim Yang Disetujui'].quantile(0.980)
df_klaim_paid['Nominal_Klaim_Capped'] = np.clip(df_klaim_paid['Nominal Klaim Yang Disetujui'], a_min=0, a_max=limit)

df_klaim_paid['YearMonth'] = df_klaim_paid[date_col].dt.to_period('M').astype(str)

monthly_data = df_klaim_paid.groupby('YearMonth').agg(
    Claim_Frequency=('Claim ID', 'count'),
    Total_Claim=('Nominal_Klaim_Capped', 'sum') 
).reset_index()

monthly_data['Date'] = pd.to_datetime(monthly_data['YearMonth'] + '-01')
monthly_data = monthly_data.sort_values('Date').reset_index(drop=True)

# ---------------------------------------------------------
# IBNR PATCH (Pengaman Data Terakhir)
# ---------------------------------------------------------
max_idx = len(monthly_data) - 1
last_freq = monthly_data.loc[max_idx, 'Claim_Frequency']
prev_freq = monthly_data.loc[max_idx - 1, 'Claim_Frequency']

if last_freq < 0.7 * prev_freq:
    print(f"\n[IBNR COMPLETION AKTIF] Menggenapkan data bulan terakhir x1.4")
    monthly_data.loc[max_idx, 'Claim_Frequency'] *= 1.4
    monthly_data.loc[max_idx, 'Total_Claim'] *= 1.4

# THE SWEET SPOT: HANYA data post-Covid (2022 ke atas)
monthly_data = monthly_data[monthly_data['Date'] >= '2022-01-01'].reset_index(drop=True)

# THE BILLION SCALING (RAHASIA UTAMA SKOR 3.6)
monthly_data['Total_Claim_B'] = monthly_data['Total_Claim'] / 1e9

# ==========================================
# 2. AUTOREGRESSIVE FEATURES (KEMBALI KE 3.6)
# ==========================================
def create_features(df):
    df = df.copy()
    
    # KITA KEMBALIKAN 'Month' KARENA BERFUNGSI SEBAGAI PROXY LONJAKAN AKHIR TAHUN
    df['Month'] = df['Date'].dt.month
    
    for col in ['Claim_Frequency', 'Total_Claim_B']:
        # Lags yang terbukti paling stabil di 3.6 (1, 2, 3, 6)
        for i in [1, 2, 3, 6]:
            df[f'{col}_lag_{i}'] = df[col].shift(i)
        
        # Roll Mean untuk stabilisator tren
        df[f'{col}_roll_mean_3'] = df[col].shift(1).rolling(window=3).mean()
        
    return df

ts_data = create_features(monthly_data)
months_to_predict = ['2025-08-01', '2025-09-01', '2025-10-01', '2025-11-01', '2025-12-01']
targets = ['Claim_Frequency', 'Total_Claim_B']

# ==========================================
# 3. THE 3.6 GOLDEN ENSEMBLE TRAINING (+ MULTIPLICATIVE)
# ==========================================
print("\nMelatih Model Recursive (The Multiplicative Revival)...")
current_ts_data = ts_data.copy()
final_preds = {'2025_08': {}, '2025_09': {}, '2025_10': {}, '2025_11': {}, '2025_12': {}}

# 1. Latih Holt-Winters
hw_preds_dict = {}
for target in targets:
    hw_data = monthly_data[target].values
    try:
        # KUNCI PENEMBUS < 3.0: seasonal='mul'
        # Membiarkan lonjakan akhir tahun membesar secara proporsional sesuai inflasi
        hw_fit = ExponentialSmoothing(hw_data, trend='add', seasonal='mul', seasonal_periods=12, damped_trend=True).fit(optimized=True)
        hw_preds_dict[target] = hw_fit.forecast(steps=5)
    except:
        hw_fit = ExponentialSmoothing(hw_data, trend='add', damped_trend=True).fit(optimized=True)
        hw_preds_dict[target] = hw_fit.forecast(steps=5)
        
# 2. Latih Regresi secara bertahap (Bulan demi bulan)
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
        
        bayes = BayesianRidge().fit(X_train, y_train)
        ridge = Ridge(alpha=1.0).fit(X_train, y_train) 
        
        pred_bayes = bayes.predict(X_test)[0]
        pred_ridge = ridge.predict(X_test)[0]
        pred_hw = hw_preds_dict[target][i]
        
        # RASIO EMAS 3.6 SECARA ABSOLUT!
        final_val = (0.40 * pred_bayes) + (0.30 * pred_ridge) + (0.30 * pred_hw)
        
        # Simpan tebakan
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
print("\n--- HASIL PREDIKSI (THE MULTIPLICATIVE REVIVAL) ---")

for month_key in final_preds.keys():
    freq = final_preds[month_key]['Claim_Frequency']
    total = final_preds[month_key]['Total_Claim']
    
    # Severity murni sebagai turunan, mencegah angka miliaran meledak
    sev = total / freq if freq > 0 else 0
    
    print(f"{month_key} -> Freq: {freq:.1f} | Sev: {sev:,.0f} | Total: {total:,.0f}")
    
    submission_rows.append({'id': f'{month_key}_Claim_Frequency', 'value': freq})
    submission_rows.append({'id': f'{month_key}_Claim_Severity', 'value': sev})
    submission_rows.append({'id': f'{month_key}_Total_Claim', 'value': total})

submission_df = pd.DataFrame(submission_rows)
submission_df.to_csv('submission_multiplicative_revival.csv', index=False)
print("\n[LOCKED] File 'submission_multiplicative_revival.csv' siap!")
print("Kembali ke akar 3.6 dengan optimasi Multiplicative Holt-Winters! Gas < 3.0!")