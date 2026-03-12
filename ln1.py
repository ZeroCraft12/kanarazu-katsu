import pandas as pd
import numpy as np
from sklearn.linear_model import BayesianRidge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import warnings
warnings.filterwarnings('ignore')

print("Membaca Data Klaim...")
df_klaim = pd.read_csv('Data_Klaim.csv')

# ==========================================
# 1. PONDASI DATA (KEMBALI KE BASE MURNI)
# ==========================================
date_col = 'Tanggal Pasien Masuk RS'
df_klaim[date_col] = pd.to_datetime(df_klaim[date_col], errors='coerce')
df_klaim = df_klaim.dropna(subset=[date_col])

df_klaim_paid = df_klaim[df_klaim['Status Klaim'] == 'PAID'].copy()

# Capping di 98.0% (Standar Emas kita)
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
# 2. PELONTAR IBNR 1.4x (SANG PENYELAMAT Q3)
# ==========================================
max_idx = len(monthly_data) - 1
last_freq = monthly_data.loc[max_idx, 'Claim_Frequency']
prev_freq = monthly_data.loc[max_idx-1, 'Claim_Frequency']

# Angkat bulan Juli jika belum selesai direkap oleh RS
if last_freq < 0.7 * prev_freq:
    print("\n[IBNR 1.4x AKTIF] Mengangkat data bulan Juli 2025 sebagai Launchpad...")
    monthly_data.loc[max_idx, 'Claim_Frequency'] *= 1.4
    monthly_data.loc[max_idx, 'Total_Claim'] *= 1.4

monthly_data['Total_Claim_B'] = monthly_data['Total_Claim'] / 1e9
monthly_data['Year'] = monthly_data['Date'].dt.year
monthly_data['Month'] = monthly_data['Date'].dt.month

# ==========================================
# 3. THE CHAIN LADDER METHOD (RASIO AKTUAL)
# ==========================================
# Kita hitung "Berapa kali lipat" pertumbuhan dari bulan ke bulan di tahun-tahun sebelumnya
print("\nMenghitung Rasio Pertumbuhan Chain Ladder (MoM)...")
mom_freq = {}
mom_total = {}

for target, mom_dict in zip(['Claim_Frequency', 'Total_Claim_B'], [mom_freq, mom_total]):
    pivot = monthly_data.pivot_table(index='Year', columns='Month', values=target)
    for m in [8, 9, 10, 11, 12]:
        # Rasio Bulan M dibagi Bulan M-1
        valid_data = pivot[[m, m-1]].dropna()
        ratios = valid_data[m] / (valid_data[m-1] + 1e-6)
        
        # Clip rasio agar anomali ekstrem dari masa lalu tidak merusak rata-rata
        clipped_ratios = np.clip(ratios, 0.5, 1.8) 
        mom_dict[m] = clipped_ratios.mean()

# ==========================================
# 4. PREDIKSI MASA DEPAN (THE HYBRID BLEND)
# ==========================================
print("Melatih Model (Chain Ladder + Bayesian Ridge)...")

# A. PREDIKSI CHAIN LADDER (Multiplikasi Berantai)
cl_preds_freq = []
cl_preds_total = []

# Mulai dari nilai Juli 2025 yang sudah di-IBNR
curr_freq = monthly_data['Claim_Frequency'].iloc[-1]
curr_total = monthly_data['Total_Claim_B'].iloc[-1]

for m in [8, 9, 10, 11, 12]:
    curr_freq = curr_freq * mom_freq[m]
    curr_total = curr_total * mom_total[m]
    cl_preds_freq.append(curr_freq)
    cl_preds_total.append(curr_total)

# B. PREDIKSI BAYESIAN RIDGE (Penstabil Tren Makro)
# Gunakan Fourier (sin/cos) agar AI Linier pintar membaca musim tanpa error
monthly_data['Time_Index'] = np.arange(1, len(monthly_data) + 1)
monthly_data['sin1'] = np.sin(2 * np.pi * monthly_data['Month'] / 12)
monthly_data['cos1'] = np.cos(2 * np.pi * monthly_data['Month'] / 12)

future_dates = pd.date_range('2025-08-01', periods=5, freq='MS')
future_df = pd.DataFrame({'Date': future_dates})
future_df['Month'] = future_df['Date'].dt.month
future_df['Time_Index'] = np.arange(len(monthly_data) + 1, len(monthly_data) + 6)
future_df['sin1'] = np.sin(2 * np.pi * future_df['Month'] / 12)
future_df['cos1'] = np.cos(2 * np.pi * future_df['Month'] / 12)

ml_features = ['Time_Index', 'sin1', 'cos1']

bayes_freq = make_pipeline(StandardScaler(), BayesianRidge()).fit(monthly_data[ml_features], monthly_data['Claim_Frequency'])
ml_preds_freq = bayes_freq.predict(future_df[ml_features])

bayes_total = make_pipeline(StandardScaler(), BayesianRidge()).fit(monthly_data[ml_features], monthly_data['Total_Claim_B'])
ml_preds_total = bayes_total.predict(future_df[ml_features])

# C. THE ULTIMATE BLEND (60% Chain Ladder Aktuaria, 40% AI Linier)
final_freqs = (0.60 * np.array(cl_preds_freq)) + (0.40 * ml_preds_freq)
final_totals_b = (0.60 * np.array(cl_preds_total)) + (0.40 * ml_preds_total)

# ==========================================
# 5. REKONSTRUKSI & EXPORT HASIL
# ==========================================
submission_rows = []
print("\n--- HASIL PREDIKSI (THE CHAIN LADDER ACTUARY) ---")

for i in range(5):
    month_key = future_dates[i].strftime('%Y_%m')
    
    freq = final_freqs[i]
    total_b = final_totals_b[i]
    total = total_b * 1e9  # Kembalikan ke miliaran
    
    sev = total / freq if freq > 0 else 0
    
    print(f"{month_key} -> Freq: {freq:.1f} | Sev: {sev:,.0f} | Total: {total:,.0f}")
    
    submission_rows.append({'id': f'{month_key}_Claim_Frequency', 'value': freq})
    submission_rows.append({'id': f'{month_key}_Claim_Severity', 'value': sev})
    submission_rows.append({'id': f'{month_key}_Total_Claim', 'value': total})

submission_df = pd.DataFrame(submission_rows)
submission_df.to_csv('submission_chain_ladder.csv', index=False)
print("\n[LOCKED] File 'submission_chain_ladder.csv' siap!")
print("Rasio Pertumbuhan Chain Ladder aktif! AI kini mengalikan tren dengan sempurna. GAS < 3.0!")