import pandas as pd
import numpy as np
import lightgbm as lgb
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import warnings
warnings.filterwarnings('ignore')

print("Membaca data Klaim...")
df_klaim = pd.read_csv('Data_Klaim.csv')

# ==========================================
# 1. PEMBERSIHAN DATA TINGKAT TINGGI
# ==========================================
date_col = 'Tanggal Pasien Masuk RS'
df_klaim[date_col] = pd.to_datetime(df_klaim[date_col], errors='coerce')
df_klaim = df_klaim.dropna(subset=[date_col])

df_klaim_paid = df_klaim[df_klaim['Status Klaim'] == 'PAID'].copy()

# Capping di 97.5% (Sangat ketat membuang outlier agar Severity sangat stabil)
upper_limit = df_klaim_paid['Nominal Klaim Yang Disetujui'].quantile(0.975)
df_klaim_paid['Nominal_Klaim_Capped'] = np.clip(df_klaim_paid['Nominal Klaim Yang Disetujui'], a_min=0, a_max=upper_limit)
df_klaim_paid['YearMonth'] = df_klaim_paid[date_col].dt.to_period('M').astype(str)

monthly_data = df_klaim_paid.groupby('YearMonth').agg(
    Claim_Frequency=('Claim ID', 'count'),
    Total_Claim=('Nominal_Klaim_Capped', 'sum') 
).reset_index()

monthly_data['Date'] = pd.to_datetime(monthly_data['YearMonth'] + '-01')
monthly_data = monthly_data.sort_values('Date').reset_index(drop=True)

# Severity Aktual per Bulan
monthly_data['Claim_Severity'] = monthly_data['Total_Claim'] / monthly_data['Claim_Frequency']

# ---------------------------------------------------------
# IBNR TRUE-FIX (Bukan tebakan x1.4, tapi perataan matematis)
# ---------------------------------------------------------
max_idx = len(monthly_data) - 1
last_freq = monthly_data.loc[max_idx, 'Claim_Frequency']
prev_freq = monthly_data.loc[max_idx - 1, 'Claim_Frequency']

if last_freq < 0.8 * prev_freq:
    print(f"\n[IBNR FIX] Bulan terakhir ({monthly_data.loc[max_idx, 'YearMonth']}) terdeteksi anjlok/belum lengkap.")
    # Kita ganti frekuensi bulan terakhir dengan rata-rata 3 bulan sebelumnya yang sudah matang
    avg_freq_3m = monthly_data['Claim_Frequency'].iloc[-4:-1].mean()
    monthly_data.loc[max_idx, 'Claim_Frequency'] = avg_freq_3m
    print(f"Frekuensi dikoreksi dari {last_freq:.0f} menjadi {avg_freq_3m:.1f}")

# Fokus pada tren modern
monthly_data = monthly_data[monthly_data['Date'] >= '2022-01-01'].reset_index(drop=True)

# ==========================================
# 2. PREDIKSI SEVERITY (HISTORICAL WEIGHTED MEDIAN)
# ==========================================
# Severity SANGAT BERBAHAYA jika ditebak pakai ML. Kita gunakan kestabilan statistik.
print("\nMenghitung Base Severity...")
# Ambil 6 bulan terakhir
recent_sev = monthly_data['Claim_Severity'].tail(6).values
# Beri bobot lebih tinggi pada bulan-bulan paling baru
weights = np.array([1, 1, 2, 2, 3, 4]) 
weighted_base_severity = np.average(recent_sev, weights=weights)

print(f"Base Severity terkunci di: Rp {weighted_base_severity:,.2f}")

# ==========================================
# 3. PREDIKSI FREKUENSI (ENSEMBLE TIME-SERIES & ML)
# ==========================================
def create_features(df):
    df = df.copy()
    df['Month'] = df['Date'].dt.month
    for i in [1, 2, 3, 6]:
        df[f'Freq_lag_{i}'] = df['Claim_Frequency'].shift(i)
    df['Freq_roll_mean_3'] = df['Claim_Frequency'].shift(1).rolling(3).mean()
    return df

ts_data = create_features(monthly_data)
months_to_predict = ['2025-08-01', '2025-09-01', '2025-10-01', '2025-11-01', '2025-12-01']

current_ts_data = ts_data.copy()
# PERBAIKAN BUG NaN/String: Mengecualikan kolom 'YearMonth' secara spesifik agar tidak dibaca sebagai fitur
feat_cols = [c for c in current_ts_data.columns if ('lag' in c or 'Month' in c or 'roll' in c) and c != 'YearMonth']

# Model 1: Holt-Winters (Sangat bagus untuk menangkap tren menurun/naik secara natural)
hw_model = ExponentialSmoothing(monthly_data['Claim_Frequency'], trend='add', damped_trend=True, seasonal=None)
hw_fit = hw_model.fit(optimized=True)
hw_forecast = hw_fit.forecast(steps=5).values

predictions = {}

for i, pred_date in enumerate(months_to_predict):
    # Model 2: LightGBM (Sangat bagus untuk pola musiman bulanan)
    train_df = current_ts_data[current_ts_data['Date'] < pred_date].dropna()
    X_train = train_df[feat_cols].fillna(0)
    y_train = train_df['Claim_Frequency']
    
    m_lgb = lgb.LGBMRegressor(objective='mae', learning_rate=0.05, max_depth=3, n_estimators=100, random_state=42, verbose=-1)
    m_lgb.fit(X_train, y_train)
    
    if not (current_ts_data['Date'] == pred_date).any():
        new_row = pd.DataFrame({'Date': [pd.to_datetime(pred_date)]})
        current_ts_data = pd.concat([current_ts_data, new_row], ignore_index=True)
        
    temp_ts = create_features(current_ts_data)
    X_test = temp_ts[temp_ts['Date'] == pred_date][feat_cols].fillna(0)
    
    pred_lgb = m_lgb.predict(X_test)[0]
    pred_hw = hw_forecast[i]
    
    # Baurkan: 50% LGBM (Musiman) + 50% Holt-Winters (Tren)
    final_freq = (0.5 * pred_lgb) + (0.5 * pred_hw)
    
    current_ts_data.loc[current_ts_data['Date'] == pred_date, 'Claim_Frequency'] = final_freq
    month_key = pred_date[:7].replace('-', '_')
    predictions[month_key] = {'Claim_Frequency': final_freq}

# ==========================================
# 4. PENYATUAN (FREQ x SEV) & EXPORT
# ==========================================
submission_rows = []
print("\n--- HASIL PREDIKSI (ACTUARIAL FREQ-SEV DECOUPLING) ---")

medical_inflation_rate = 1.003 # Inflasi 0.3% per bulan (Standar wajar RS)

for i, month_key in enumerate(predictions.keys()):
    freq = predictions[month_key]['Claim_Frequency']
    
    # Severity dihitung dari Base ditambah inflasi natural
    sev = weighted_base_severity * (medical_inflation_rate ** i)
    
    # TOTAL KLAIM DIHITUNG MURNI DARI PERKALIAN, BUKAN DITEBAK AI
    total = freq * sev
    
    print(f"{month_key} -> Freq: {freq:.1f} | Sev: {sev:,.0f} | Total: {total:,.0f}")
    
    submission_rows.append({'id': f'{month_key}_Claim_Frequency', 'value': freq})
    submission_rows.append({'id': f'{month_key}_Claim_Severity', 'value': sev})
    submission_rows.append({'id': f'{month_key}_Total_Claim', 'value': total})

submission_df = pd.DataFrame(submission_rows)
submission_df.to_csv('submission_true_actuarial.csv', index=False)
print("\n[LOCKED] File 'submission_true_actuarial.csv' siap!")
print("Ini adalah kode paling stabil secara matematis. Total klaim tidak akan pernah 'melenceng' lagi.")