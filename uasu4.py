import pandas as pd
import numpy as np
from sklearn.linear_model import BayesianRidge, Ridge, HuberRegressor
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import warnings
warnings.filterwarnings('ignore')

print("Membaca Data Klaim...")
df_klaim = pd.read_csv('Data_Klaim.csv')

# ==========================================
# 1. PONDASI DATA (PREPROCESSING & CAPPING)
# ==========================================
date_col = 'Tanggal Pasien Masuk RS'
df_klaim[date_col] = pd.to_datetime(df_klaim[date_col], errors='coerce')
df_klaim = df_klaim.dropna(subset=[date_col])

df_klaim_paid = df_klaim[df_klaim['Status Klaim'] == 'PAID'].copy()

# Capping di 98.0% (Paling aman untuk regresi linear)
upper_limit = df_klaim_paid['Nominal Klaim Yang Disetujui'].quantile(0.98)
df_klaim_paid['Nominal_Klaim_Capped'] = np.clip(df_klaim_paid['Nominal Klaim Yang Disetujui'], a_min=0, a_max=upper_limit)

df_klaim_paid['YearMonth'] = df_klaim_paid[date_col].dt.to_period('M').astype(str)

monthly_data = df_klaim_paid.groupby('YearMonth').agg(
    Claim_Frequency=('Claim ID', 'count'),
    Total_Claim=('Nominal_Klaim_Capped', 'sum') 
).reset_index()

monthly_data['Date'] = pd.to_datetime(monthly_data['YearMonth'] + '-01')
monthly_data = monthly_data.sort_values('Date').reset_index(drop=True)

# ---------------------------------------------------------
# IBNR COMPLETION (KUNCI STABILITAS SKOR 4.8)
# ---------------------------------------------------------
max_idx = len(monthly_data) - 1
last_freq = monthly_data.loc[max_idx, 'Claim_Frequency']
prev_freq = monthly_data.loc[max_idx - 1, 'Claim_Frequency']

if last_freq < 0.7 * prev_freq:
    print(f"\n[IBNR COMPLETION AKTIF] Menggenapkan data bulan terakhir yang anjlok x1.4")
    monthly_data.loc[max_idx, 'Claim_Frequency'] = last_freq * 1.4
    monthly_data.loc[max_idx, 'Total_Claim'] = monthly_data.loc[max_idx, 'Total_Claim'] * 1.4

# THE GRANDMASTER TRICK: 
# Ambil dari 2021 untuk "modal" penghitungan Lag 12 bulan, 
# tapi nanti kita latih AI HANYA pada data 2022 ke atas!
monthly_data = monthly_data[monthly_data['Date'] >= '2021-01-01'].reset_index(drop=True)

# ---------------------------------------------------------
# TARGET LOG-TRANSFORM
# ---------------------------------------------------------
monthly_data['Log_Freq'] = np.log1p(monthly_data['Claim_Frequency'])
monthly_data['Log_Total'] = np.log1p(monthly_data['Total_Claim'])

# ==========================================
# 2. FITUR UNTUK MICRO-DATA (LINEAR FRIENDLY)
# ==========================================
def create_features(df):
    df = df.copy()
    df['Month'] = df['Date'].dt.month
    
    for col in ['Log_Freq', 'Log_Total']:
        # Lag ditambah ke 12 untuk menangkap siklus tahunan penuh
        for i in [1, 2, 3, 6, 12]: 
            df[f'{col}_lag_{i}'] = df[col].shift(i)
        
        # Rata-rata & EWMA (Sangat disukai oleh Bayesian/Ridge)
        df[f'{col}_roll_mean_3'] = df[col].shift(1).rolling(window=3).mean()
        df[f'{col}_ewma_3'] = df[col].shift(1).ewm(span=3, adjust=False).mean()
        
    return df

ts_data = create_features(monthly_data)
months_to_predict = ['2025-08-01', '2025-09-01', '2025-10-01', '2025-11-01', '2025-12-01']
targets = ['Log_Freq', 'Log_Total']

# ==========================================
# 3. STATISTICAL ENSEMBLE (BAYESIAN + RIDGE + HUBER + HW)
# ==========================================
print("\nMelatih Model Statistik Presisi Tinggi (~40 Baris)...")
final_preds = {'2025_08': {}, '2025_09': {}, '2025_10': {}, '2025_11': {}, '2025_12': {}}
current_ts_data = ts_data.copy()

exclude_cols = ['YearMonth', 'Date', 'Claim_Frequency', 'Total_Claim', 'Log_Freq', 'Log_Total']
features = [c for c in current_ts_data.columns if c not in exclude_cols]

# KUNCI PERBAIKAN BUG: Holt-Winters dilatih SEKALI di awal secara independen
hw_forecasts_dict = {}
for target in targets:
    hw_data_source = current_ts_data[current_ts_data['Date'] >= '2022-01-01'][target].values
    try:
        hw_model = ExponentialSmoothing(hw_data_source, trend='add', seasonal='add', seasonal_periods=12, damped_trend=True)
        hw_fit = hw_model.fit(optimized=True)
        hw_forecasts_dict[target] = hw_fit.forecast(steps=5)
    except:
        hw_model = ExponentialSmoothing(hw_data_source, trend='add', damped_trend=True)
        hw_fit = hw_model.fit(optimized=True)
        hw_forecasts_dict[target] = hw_fit.forecast(steps=5)

# PERBAIKAN LOOP: Kita melangkah BULAN DEMI BULAN agar tebakannya tidak bocor menjadi NaN
for i, pred_date in enumerate(months_to_predict):
    
    # 1. Buat wadah bulan masa depan
    if not (current_ts_data['Date'] == pred_date).any():
        new_row = pd.DataFrame({'Date': [pd.to_datetime(pred_date)]})
        current_ts_data = pd.concat([current_ts_data, new_row], ignore_index=True)
        
    # 2. Update fitur untuk mendapatkan Lag dari tebakan bulan sebelumnya
    temp_ts = create_features(current_ts_data)
    X_test = temp_ts[temp_ts['Date'] == pred_date][features].fillna(0)
    
    # 3. Prediksi kedua target secara berurutan untuk SATU bulan ini saja
    for target in targets:
        train_df = temp_ts[(temp_ts['Date'] >= '2022-01-01') & (temp_ts['Date'] < pred_date)].dropna()
        X_train = train_df[features].fillna(0)
        y_train = train_df[target]
        
        # Latih model
        model_bayes = BayesianRidge()
        model_bayes.fit(X_train, y_train)
        
        model_ridge = Ridge(alpha=10.0)
        model_ridge.fit(X_train, y_train)
        
        model_huber = HuberRegressor(epsilon=1.35)
        model_huber.fit(X_train, y_train)
        
        # Prediksi
        pred_bayes = model_bayes.predict(X_test)[0]
        pred_ridge = model_ridge.predict(X_test)[0]
        pred_huber = model_huber.predict(X_test)[0]
        pred_hw = hw_forecasts_dict[target][i]
        
        # Blending Ultimate
        pred_log_final = (0.35 * pred_bayes) + (0.25 * pred_ridge) + (0.15 * pred_huber) + (0.25 * pred_hw)
        
        # Simpan tebakan ini agar bisa dipakai sebagai 'masa lalu' untuk tebakan bulan depannya!
        current_ts_data.loc[current_ts_data['Date'] == pred_date, target] = pred_log_final
        
        # Kembalikan skala Log ke angka nyata
        pred_original = np.expm1(pred_log_final)
        
        target_name = 'Claim_Frequency' if target == 'Log_Freq' else 'Total_Claim'
        month_key = pred_date[:7].replace('-', '_')
        final_preds[month_key][target_name] = pred_original

# ==========================================
# 4. EXPORT HASIL
# ==========================================
submission_rows = []
print("\n--- HASIL PREDIKSI (MICRO-TUNED BAYESIAN) ---")
print("Kita peras presisinya! Bismillah target < 3.0!")

for month_key in final_preds.keys():
    freq = final_preds[month_key]['Claim_Frequency']
    total = final_preds[month_key]['Total_Claim']
    
    sev = total / freq if freq > 0 else 0
    
    print(f"{month_key} -> Freq: {freq:.1f} | Sev: {sev:,.0f} | Total: {total:,.0f}")
    
    submission_rows.append({'id': f'{month_key}_Claim_Frequency', 'value': freq})
    submission_rows.append({'id': f'{month_key}_Claim_Severity', 'value': sev})
    submission_rows.append({'id': f'{month_key}_Total_Claim', 'value': total})

submission_df = pd.DataFrame(submission_rows)
submission_df.to_csv('submission_bayesian_microtuned.csv', index=False)
print("\nFile 'submission_bayesian_microtuned.csv' berhasil dibuat! Gas Submit!")