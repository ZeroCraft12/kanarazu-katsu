import pandas as pd #SKOR 10
import numpy as np
import lightgbm as lgb
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.linear_model import HuberRegressor
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# ==========================================
# 1. PERSIAPAN DATA & DYNAMIC WINSORIZATION
# ==========================================
print("Membaca data...")
df_klaim = pd.read_csv('Data_Klaim.csv')

df_klaim['Tanggal Pasien Masuk RS'] = pd.to_datetime(df_klaim['Tanggal Pasien Masuk RS'])
date_col = 'Tanggal Pasien Masuk RS' 
df_klaim = df_klaim.dropna(subset=[date_col])

df_klaim_paid = df_klaim[df_klaim['Status Klaim'] == 'PAID'].copy()

# Teknik Ekstrem: Hapus klaim yang nilai rupiahnya benar-benar tidak wajar 
# (Persentil 93 - Kita buang 7% klaim tertinggi agar tidak merusak ekstrapolasi tren)
upper_limit = df_klaim_paid['Nominal Klaim Yang Disetujui'].quantile(0.93)
df_klaim_paid['Nominal_Klaim_Capped'] = np.clip(df_klaim_paid['Nominal Klaim Yang Disetujui'], a_min=0, a_max=upper_limit)

df_klaim_paid['YearMonth'] = df_klaim_paid[date_col].dt.to_period('M').astype(str)

# ==========================================
# 2. AGREGASI & FOKUS PADA RECENT TRENDS
# ==========================================
print("Melakukan agregasi bulanan...")
monthly_data = df_klaim_paid.groupby('YearMonth').agg(
    Claim_Frequency=('Claim ID', 'count'),
    Total_Claim=('Nominal_Klaim_Capped', 'sum') 
).reset_index()

monthly_data['Date'] = pd.to_datetime(monthly_data['YearMonth'] + '-01')
monthly_data = monthly_data.sort_values('Date').reset_index(drop=True)

# THE GOLDEN RULE KAGGLE: Potong data masa lalu yang tidak relevan!
# Kita HANYA pakai data dari Januari 2023 ke atas. 
# Data Covid/post-Covid (2020-2022) sangat berantakan dan menipu arah tren AI.
train_data = monthly_data[monthly_data['Date'] >= '2023-01-01'].reset_index(drop=True)

# Siapkan time-index (0, 1, 2, 3...) untuk algoritma Regresi Linear
train_data['Time_Index'] = np.arange(len(train_data))

# ==========================================
# 3. OVERKILL ENSEMBLE MODELING (HOLT-WINTERS + HUBER + HEURISTIC)
# ==========================================
targets = ['Claim_Frequency', 'Total_Claim']
months_to_predict = ['2025-08-01', '2025-09-01', '2025-10-01', '2025-11-01', '2025-12-01']
predictions = {'2025_08': {}, '2025_09': {}, '2025_10': {}, '2025_11': {}, '2025_12': {}}

print("Melatih Ensemble Models (Membaca Tren Ekstrapolasi)...")

for target in targets:
    y_train = train_data[target].values
    x_train = train_data['Time_Index'].values.reshape(-1, 1)
    
    # -----------------------------------------------------
    # MODEL 1: Huber Regressor (Robust Linear Trend)
    # Sangat kuat mengabaikan outlier dan menarik garis lurus tren masa lalu ke masa depan
    # -----------------------------------------------------
    huber = HuberRegressor(epsilon=1.35)
    huber.fit(x_train, y_train)
    
    # -----------------------------------------------------
    # MODEL 2: Holt-Winters (Exponential Smoothing)
    # Model time series kelas dewa. Damped_trend=True agar tren turun/naik tidak bablas sampai minus/tak terhingga
    # -----------------------------------------------------
    try:
        hw_model = ExponentialSmoothing(y_train, trend='add', damped_trend=True, seasonal=None)
        hw_fit = hw_model.fit(optimized=True)
    except:
        # Fallback jika statsmodels gagal fit
        hw_fit = None
        
    # -----------------------------------------------------
    # MODEL 3: Recent Weighted Heuristic (Mencuri ide dari skormu yang 8)
    # Rata-rata 4 bulan terakhir dengan bobot paling berat di bulan ke-1 terakhir
    # -----------------------------------------------------
    last_4_months = y_train[-4:]
    weights = np.array([0.1, 0.2, 0.3, 0.4]) # Bulan terakhir bobotnya 40%
    heuristic_base = np.sum(last_4_months * weights)
    
    # --- PROSES PREDIKSI MASA DEPAN ---
    start_index = len(train_data)
    
    for i, pred_date in enumerate(months_to_predict):
        future_index = start_index + i
        
        # 1. Prediksi Regresi
        pred_huber = huber.predict([[future_index]])[0]
        
        # 2. Prediksi Holt-Winters
        if hw_fit is not None:
            pred_hw = hw_fit.forecast(steps=i+1)[-1]
        else:
            pred_hw = pred_huber # Fallback
            
        # 3. Prediksi Heuristik (Tarik baseline dan asumsikan turun 2% tiap bulan seperti realita industri asuransi)
        pred_heur = heuristic_base * (0.98 ** (i + 1)) 
        
        # 4. THE ULTIMATE BLEND (Bobot dicampur)
        # 40% Holt-Winters (Sangat presisi untuk trend)
        # 30% Regresi Kebal Outlier
        # 30% Heuristik Manual (Penahan agar AI tidak berhalusinasi)
        final_pred = (0.40 * pred_hw) + (0.30 * pred_huber) + (0.30 * pred_heur)
        
        # Safety net: Tidak boleh lebih kecil dari 50% nilai minimum historis 2023-2025
        min_historical = np.min(y_train)
        final_pred = max(final_pred, min_historical * 0.5)
        
        month_key = pred_date[:7].replace('-', '_')
        predictions[month_key][target] = final_pred

# ==========================================
# 4. FORMATTING, PENYESUAIAN SEVERITY & EXPORT
# ==========================================
submission_rows = []
print("\nHasil Ekstrapolasi Pamungkas:")

pred_plot_data = []

for month_key, preds in predictions.items():
    freq = preds['Claim_Frequency']
    total = preds['Total_Claim']
    
    # Severity dihitung mundur secara logis
    sev = total / freq if freq > 0 else 0
    
    print(f"{month_key} -> Freq: {freq:.1f} | Sev: {sev:,.0f} | Total: {total:,.0f}")
    
    submission_rows.append({'id': f'{month_key}_Claim_Frequency', 'value': freq})
    submission_rows.append({'id': f'{month_key}_Claim_Severity', 'value': sev})
    submission_rows.append({'id': f'{month_key}_Total_Claim', 'value': total})
    
    pred_plot_data.append({'Date': pd.to_datetime(month_key.replace('_', '-') + '-01'), 'Total_Claim': total})

submission_df = pd.DataFrame(submission_rows)
submission_df.to_csv('submission_grandmaster_extrapolate.csv', index=False)
print("\nFile 'submission_grandmaster_extrapolate.csv' berhasil dibuat! Bismillah submit ini.")

# --- Visualisasi Evaluasi Ekstrapolasi ---
pred_df = pd.DataFrame(pred_plot_data)

plt.figure(figsize=(12, 6))
plt.plot(train_data['Date'], train_data['Total_Claim'], marker='o', label='Historis (2023 - 2025)', color='royalblue')
plt.plot(pred_df['Date'], pred_df['Total_Claim'], marker='s', color='crimson', linestyle='--', linewidth=2, label='Prediksi Ekstrapolasi (HW + Huber + Heuristic)')

# Tambah garis pemisah antara masa lalu dan masa depan
plt.axvline(x=pd.to_datetime('2025-07-15'), color='black', linestyle=':', alpha=0.5)

plt.title('Proyeksi Total Klaim: Ekstrapolasi Linear & Eksponensial', fontsize=14, fontweight='bold')
plt.xlabel('Bulan', fontsize=12)
plt.ylabel('Total Klaim (Rupiah)', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(fontsize=11)
plt.tight_layout()
plt.savefig('plot_grandmaster_extrapolate.png')
print("Grafik disimpan! Cek arah garis merah, ini harusnya mulus mengikuti tren turun/naik terakhir.")