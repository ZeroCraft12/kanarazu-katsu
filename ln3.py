import pandas as pd
import numpy as np
from sklearn.linear_model import BayesianRidge, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import warnings
warnings.filterwarnings('ignore')

print("Membaca Data Klaim...")
df_klaim = pd.read_csv('Data_Klaim.csv')

# ==========================================
# 1. PONDASI DATA (BASE 3.6 SANG JUARA)
# ==========================================
date_col = 'Tanggal Pasien Masuk RS'
df_klaim[date_col] = pd.to_datetime(df_klaim[date_col], errors='coerce')
df_klaim = df_klaim.dropna(subset=[date_col])

df_klaim_paid = df_klaim[df_klaim['Status Klaim'] == 'PAID'].copy()

# Capping dikembalikan ke 98.0% untuk kestabilan tanpa ledakan
limit = df_klaim_paid['Nominal Klaim Yang Disetujui'].quantile(0.980)
df_klaim_paid['Nominal_Klaim_Capped'] = np.clip(df_klaim_paid['Nominal Klaim Yang Disetujui'], a_min=0, a_max=limit)

df_klaim_paid['YearMonth'] = df_klaim_paid[date_col].dt.to_period('M').astype(str)

monthly_data = df_klaim_paid.groupby('YearMonth').agg(
    Claim_Frequency=('Claim ID', 'count'),
    Total_Claim=('Nominal_Klaim_Capped', 'sum') 
).reset_index()

monthly_data['Date'] = pd.to_datetime(monthly_data['YearMonth'] + '-01')
monthly_data = monthly_data.sort_values('Date').reset_index(drop=True)

# Interpolasi untuk menambal bulan yang kosong
monthly_data = monthly_data.drop(columns=['YearMonth']).set_index('Date').resample('MS').interpolate(method='linear').reset_index()

# HANYA gunakan data Pasca-Covid (2022+)
monthly_data = monthly_data[monthly_data['Date'] >= '2022-01-01'].reset_index(drop=True)

# ==========================================
# 2. THE SOFT LANDING (MENCEGAH LEDAKAN PERKALIAN)
# ==========================================
max_idx = len(monthly_data) - 1
last_freq = monthly_data.loc[max_idx, 'Claim_Frequency']
prev_freq = monthly_data.loc[max_idx-1, 'Claim_Frequency']

# Jika bulan terakhir (Juli) anjlok, JANGAN KALI 1.4x!
# Ganti dengan rata-rata Mei & Juni. Ini pijakan yang sangat aman dan stabil.
if last_freq < 0.75 * prev_freq:
    print("\n[SOFT LANDING AKTIF] Bulan terakhir anjlok. Mengganti dengan rata-rata 2 bulan sebelumnya...")
    mean_freq = monthly_data['Claim_Frequency'].iloc[-3:-1].mean()
    mean_total = monthly_data['Total_Claim'].iloc[-3:-1].mean()
    
    monthly_data.loc[max_idx, 'Claim_Frequency'] = mean_freq
    monthly_data.loc[max_idx, 'Total_Claim'] = mean_total

monthly_data['Total_Claim_B'] = monthly_data['Total_Claim'] / 1e9

# ==========================================
# 3. GLOBAL HOLT-WINTERS (TREN MURNI TANPA MUSIMAN)
# ==========================================
# Holt-Winters hanya menarik garis lurus melandai. Tidak ada musiman perkalian yang meledak!
print("\nMelatih Global Holt Damped (Tren Aman)...")
hw_preds = {}
for target in ['Claim_Frequency', 'Total_Claim_B']:
    y_vals = monthly_data[target].values
    try:
        model_hw = ExponentialSmoothing(
            y_vals, trend='add', seasonal=None, damped_trend=True, initialization_method='estimated'
        ).fit(optimized=True)
        hw_preds[target] = model_hw.forecast(5)
    except:
        hw_preds[target] = np.repeat(y_vals[-1], 5)

# ==========================================
# 4. KINEMATIC FEATURES & RECURSIVE ML
# ==========================================
def create_features(df):
    df = df.copy()
    df['Month'] = df['Date'].dt.month
    for col in ['Claim_Frequency', 'Total_Claim_B']:
        # Lag 6 yang sangat penting untuk memori musiman
        for i in [1, 2, 3, 6]:
            df[f'{col}_lag_{i}'] = df[col].shift(i)
        df[f'{col}_roll_mean_3'] = df[col].shift(1).rolling(window=3).mean()
    return df

ts_data = create_features(monthly_data)
months_to_predict = ['2025-08-01', '2025-09-01', '2025-10-01', '2025-11-01', '2025-12-01']
submission_months = ['2025_08', '2025_09', '2025_10', '2025_11', '2025_12']

print("Melatih Model ML Rekursif (BayesianRidge & Ridge)...")
current_ts_data = ts_data.copy()

for pred_date in months_to_predict:
    if not (current_ts_data['Date'] == pred_date).any():
        new_row = pd.DataFrame({'Date': [pd.to_datetime(pred_date)]})
        current_ts_data = pd.concat([current_ts_data, new_row], ignore_index=True)
        
    temp_df = create_features(current_ts_data)
    exclude_cols = ['Date', 'Claim_Frequency', 'Total_Claim', 'Total_Claim_B']
    features = [c for c in temp_df.columns if c not in exclude_cols]
    
    for target in ['Claim_Frequency', 'Total_Claim_B']:
        train_df = temp_df[temp_df['Date'] < pred_date].dropna()
        X_train = train_df[features]
        y_train = train_df[target]
        X_test = temp_df[temp_df['Date'] == pred_date][features].fillna(0)
        
        bayes = make_pipeline(StandardScaler(), BayesianRidge()).fit(X_train, y_train)
        ridge = make_pipeline(StandardScaler(), Ridge(alpha=2.0)).fit(X_train, y_train)
        
        # ML menebak musiman menggunakan kombinasi Bayes dan Ridge
        ml_pred_val = (0.50 * bayes.predict(X_test)[0]) + (0.50 * ridge.predict(X_test)[0])
        current_ts_data.loc[current_ts_data['Date'] == pred_date, target] = ml_pred_val

# ==========================================
# 5. THE ZERO EXPLOSION BLEND & EXPORT
# ==========================================
submission_rows = []
print("\n--- HASIL PREDIKSI (THE ZERO EXPLOSION ACTUARY) ---")

for i, month_key in enumerate(submission_months):
    date_val = months_to_predict[i]
    
    # Ambil tebakan Musiman ML
    ml_freq = current_ts_data.loc[current_ts_data['Date'] == date_val, 'Claim_Frequency'].values[0]
    ml_total_b = current_ts_data.loc[current_ts_data['Date'] == date_val, 'Total_Claim_B'].values[0]
    
    # Ambil tebakan Tren Makro HW
    hw_freq = hw_preds['Claim_Frequency'][i]
    hw_total_b = hw_preds['Total_Claim_B'][i]
    
    # BLEND ANTI-MELEDAK (50% Tren Stabil HW + 50% Momentum Musiman ML)
    final_freq = max(0, (0.50 * hw_freq) + (0.50 * ml_freq))
    final_total_b = max(0, (0.50 * hw_total_b) + (0.50 * ml_total_b))
    
    final_total = final_total_b * 1e9
    sev = final_total / final_freq if final_freq > 0 else 0
    
    print(f"{month_key} -> Freq: {final_freq:.1f} | Sev: {sev:,.0f} | Total: {final_total:,.0f}")
    
    submission_rows.append({'id': f'{month_key}_Claim_Frequency', 'value': final_freq})
    submission_rows.append({'id': f'{month_key}_Claim_Severity', 'value': sev})
    submission_rows.append({'id': f'{month_key}_Total_Claim', 'value': final_total})

submission_df = pd.DataFrame(submission_rows)
submission_df.to_csv('submission_zero_explosion.csv', index=False)
print("\n[LOCKED] File 'submission_zero_explosion.csv' siap!")
print("Semua faktor pengali (multiplier) telah dihapus. Garis prediksi sekarang dijamin Mulus & Aman. GAS < 3.0!")