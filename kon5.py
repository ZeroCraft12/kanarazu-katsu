import pandas as pd
import numpy as np
from sklearn.linear_model import BayesianRidge, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from statsmodels.tsa.holtwinters import Holt
import warnings
warnings.filterwarnings('ignore')

print("Membaca Data Klaim...")
df_klaim = pd.read_csv('Data_Klaim.csv')

# ==========================================
# 1. PONDASI DATA (BASE MURNI 3.6)
# ==========================================
date_col = 'Tanggal Pasien Masuk RS'
df_klaim[date_col] = pd.to_datetime(df_klaim[date_col], errors='coerce')
df_klaim = df_klaim.dropna(subset=[date_col])

df_klaim_paid = df_klaim[df_klaim['Status Klaim'] == 'PAID'].copy()

# Capping dikembalikan ke 98.0% (Standar Emas kita)
limit = df_klaim_paid['Nominal Klaim Yang Disetujui'].quantile(0.980)
df_klaim_paid['Nominal_Klaim_Capped'] = np.clip(df_klaim_paid['Nominal Klaim Yang Disetujui'], a_min=0, a_max=limit)

df_klaim_paid['YearMonth'] = df_klaim_paid[date_col].dt.to_period('M').astype(str)

monthly_data = df_klaim_paid.groupby('YearMonth').agg(
    Claim_Frequency=('Claim ID', 'count'),
    Total_Claim=('Nominal_Klaim_Capped', 'sum') 
).reset_index()

monthly_data['Date'] = pd.to_datetime(monthly_data['YearMonth'] + '-01')
monthly_data = monthly_data.drop(columns=['YearMonth']).set_index('Date').resample('MS').interpolate(method='linear').reset_index()

# HANYA gunakan data Pasca-Covid (2022+)
monthly_data = monthly_data[monthly_data['Date'] >= '2022-01-01'].reset_index(drop=True)

# ==========================================
# 2. PELONTAR IBNR 1.4x (SANG PENYELAMAT 3.6)
# ==========================================
max_idx = len(monthly_data) - 1
last_freq = monthly_data.loc[max_idx, 'Claim_Frequency']
prev_freq = monthly_data.loc[max_idx-1, 'Claim_Frequency']

# Ini krusial agar tren makro tidak menukik ke bawah karena data Juli yang belum selesai direkap RS
if last_freq < 0.7 * prev_freq:
    print("\n[IBNR 1.4x AKTIF] Mengangkat data bulan terakhir sebagai batu loncatan...")
    monthly_data.loc[max_idx, 'Claim_Frequency'] *= 1.4
    monthly_data.loc[max_idx, 'Total_Claim'] *= 1.4

monthly_data['Total_Claim_B'] = monthly_data['Total_Claim'] / 1e9

# ==========================================
# 3. TARGET ENCODING (MENGHANCURKAN EFEK DOMINO)
# ==========================================
# Kita hitung RATA-RATA ASLI dari setiap bulan di masa lalu (2022 - 2024).
# AI tidak perlu lagi menebak-nebak musiman, kita berikan "bocoran" pastinya!
hist_data = monthly_data[monthly_data['Date'].dt.year < 2025]
freq_map = hist_data.groupby(hist_data['Date'].dt.month)['Claim_Frequency'].mean().to_dict()
total_map = hist_data.groupby(hist_data['Date'].dt.month)['Total_Claim_B'].mean().to_dict()

monthly_data['Month'] = monthly_data['Date'].dt.month
monthly_data['Time_Index'] = np.arange(1, len(monthly_data) + 1)

# Fitur Ajaib: Menempelkan rata-rata sejarah ke masing-masing baris
monthly_data['Hist_Freq'] = monthly_data['Month'].map(freq_map)
monthly_data['Hist_Total'] = monthly_data['Month'].map(total_map)

# Siapkan DataFrame Masa Depan (Agustus - Desember 2025)
future_dates = pd.date_range(start='2025-08-01', periods=5, freq='MS')
future_df = pd.DataFrame({'Date': future_dates})
future_df['Month'] = future_df['Date'].dt.month
future_df['Time_Index'] = np.arange(len(monthly_data) + 1, len(monthly_data) + 6)
# Tempelkan bocoran rata-rata sejarah ke bulan-bulan prediksi!
future_df['Hist_Freq'] = future_df['Month'].map(freq_map)
future_df['Hist_Total'] = future_df['Month'].map(total_map)

# ==========================================
# 4. DIRECT FORECASTING ENSEMBLE
# ==========================================
# TIDAK ADA LOOPING BULAN! AI langsung memprediksi 5 bulan sekaligus!
print("\nMelatih Model Target-Encoded (Direct Forecasting)...")
targets = ['Claim_Frequency', 'Total_Claim_B']
final_preds = {}

for target in targets:
    # Pilih fitur spesifik sesuai target yang mau ditebak
    feat_col = 'Hist_Freq' if target == 'Claim_Frequency' else 'Hist_Total'
    train_X = monthly_data[['Time_Index', feat_col]]
    train_y = monthly_data[target]
    
    test_X = future_df[['Time_Index', feat_col]]
    
    # 1. BAYESIAN RIDGE (StandarScaler kembali berkuasa!)
    bayes = make_pipeline(StandardScaler(), BayesianRidge()).fit(train_X, train_y)
    pred_bayes = bayes.predict(test_X)
    
    # 2. RIDGE ALPHA 1.0 (Penstabil Tren)
    ridge = make_pipeline(StandardScaler(), Ridge(alpha=1.0)).fit(train_X, train_y)
    pred_ridge = ridge.predict(test_X)
    
    # 3. HOLT DAMPED (Garis Lurus Melandai)
    try:
        holt = Holt(train_y.values, damped_trend=True, initialization_method='estimated').fit(optimized=True)
        pred_holt = holt.forecast(5)
    except:
        pred_holt = np.repeat(train_y.values[-1], 5)
        
    # THE GOLDEN BLEND (45% Bayes, 35% Ridge, 20% Holt)
    # Sangat kuat menangkap inflasi, sambil menyalin persis lonjakan akhir tahun dari data sejarah!
    final_preds[target] = (0.45 * pred_bayes) + (0.35 * pred_ridge) + (0.20 * pred_holt)

# ==========================================
# 5. REKONSTRUKSI & EXPORT HASIL
# ==========================================
submission_rows = []
print("\n--- HASIL PREDIKSI (TARGET ENCODED GRANDMASTER) ---")

for i in range(5):
    month_key = future_dates[i].strftime('%Y_%m')
    
    freq = final_preds['Claim_Frequency'][i]
    total_b = final_preds['Total_Claim_B'][i]
    total = total_b * 1e9  # Kembalikan ke miliaran
    
    sev = total / freq if freq > 0 else 0
    
    print(f"{month_key} -> Freq: {freq:.1f} | Sev: {sev:,.0f} | Total: {total:,.0f}")
    
    submission_rows.append({'id': f'{month_key}_Claim_Frequency', 'value': freq})
    submission_rows.append({'id': f'{month_key}_Claim_Severity', 'value': sev})
    submission_rows.append({'id': f'{month_key}_Total_Claim', 'value': total})

submission_df = pd.DataFrame(submission_rows)
submission_df.to_csv('submission_target_encoded.csv', index=False)
print("\n[LOCKED] File 'submission_target_encoded.csv' siap!")
print("Efek domino (Lag Error) DIHANCURKAN! AI menebak langsung berdasarkan sejarah asli. GAS < 3.0!")