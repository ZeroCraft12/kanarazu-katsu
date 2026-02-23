import pandas as pd #SKOR 6,2
import numpy as np
import lightgbm as lgb
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# ==========================================
# 1. PERSIAPAN DATA & PREPROCESSING
# ==========================================
print("Membaca data...")
# Ganti path sesuai dengan environment Kaggle kamu
df_klaim = pd.read_csv('Data_Klaim.csv')
df_polis = pd.read_csv('Data_Polis.csv')

# Ubah format tanggal menjadi datetime
df_klaim['Tanggal Pasien Masuk RS'] = pd.to_datetime(df_klaim['Tanggal Pasien Masuk RS'])
df_klaim['Tanggal Pembayaran Klaim'] = pd.to_datetime(df_klaim['Tanggal Pembayaran Klaim'])

# TENTUKAN TANGGAL ACUAN (Biasanya 'Tanggal Pasien Masuk RS' lebih akurat untuk memprediksi frekuensi penyakit)
date_col = 'Tanggal Pasien Masuk RS' 
df_klaim = df_klaim.dropna(subset=[date_col])

# Ekstrak Tahun-Bulan untuk agregasi (Format: YYYY-MM)
df_klaim['YearMonth'] = df_klaim[date_col].dt.to_period('M')

# Hanya ambil klaim yang statusnya 'PAID' (Jika ada status lain seperti REJECTED, abaikan)
df_klaim_paid = df_klaim[df_klaim['Status Klaim'] == 'PAID'].copy()

# ==========================================
# 2. RAHASIA KAGGLE: OUTLIER CAPPING (WINSORIZATION)
# ==========================================
# Jangan biarkan 1 klaim miliaran merusak prediksi bulanan kita.
# Kita batasi maksimal klaim di persentil 98 atau 99.
upper_limit = df_klaim_paid['Nominal Klaim Yang Disetujui'].quantile(0.98)
print(f"Membatasi nilai klaim maksimal di: Rp {upper_limit:,.2f}")

df_klaim_paid['Nominal_Klaim_Capped'] = np.clip(df_klaim_paid['Nominal Klaim Yang Disetujui'], a_min=0, a_max=upper_limit)

# ==========================================
# 3. AGREGASI DATA BULANAN (TIME SERIES)
# ==========================================
print("Melakukan agregasi bulanan...")
monthly_data = df_klaim_paid.groupby('YearMonth').agg(
    Claim_Frequency=('Claim ID', 'count'),
    Total_Claim=('Nominal_Klaim_Capped', 'sum') # Menggunakan data yang sudah dicapping
).reset_index()

# Ubah kembali period ke string atau datetime untuk kemudahan proses
monthly_data['YearMonth'] = monthly_data['YearMonth'].astype(str)
monthly_data['Date'] = pd.to_datetime(monthly_data['YearMonth'] + '-01')
monthly_data = monthly_data.sort_values('Date').reset_index(drop=True)

# Hitung Severity = Total Claim / Frequency
monthly_data['Claim_Severity'] = monthly_data['Total_Claim'] / monthly_data['Claim_Frequency']

# ==========================================
# 4. FEATURE ENGINEERING (MEMBUAT FITUR UNTUK ML)
# ==========================================
def create_features(df):
    df = df.copy()
    df['Month'] = df['Date'].dt.month
    df['Year'] = df['Date'].dt.year
    df['Quarter'] = df['Date'].dt.quarter
    
    # Lag Features (Nilai di bulan-bulan sebelumnya)
    for col in ['Claim_Frequency', 'Claim_Severity']:
        for i in [1, 2, 3, 6, 12]: # Melihat ke belakang 1, 2, 3, 6, dan 12 bulan
            df[f'{col}_lag_{i}'] = df[col].shift(i)
            
    # Rolling Features (Rata-rata bergerak, gunakan median agar lebih robust seperti percobaanmu sebelumnya)
    for col in ['Claim_Frequency', 'Claim_Severity']:
        df[f'{col}_roll_median_3'] = df[col].rolling(window=3).median()
        df[f'{col}_roll_median_6'] = df[col].rolling(window=6).median()
        df[f'{col}_roll_mean_3'] = df[col].rolling(window=3).mean()
        
    return df

ts_data = create_features(monthly_data)

# ==========================================
# 5. MODELING & PREDIKSI (AGUSTUS - DESEMBER 2025)
# ==========================================
# Kita akan memprediksi Frequency dan Severity secara terpisah.
# Mengapa Total Claim tidak diprediksi ML? Lebih aman menghitung secara matematis: Total = Freq * Severity.

targets = ['Claim_Frequency', 'Claim_Severity']
predictions = {'2025_08': {}, '2025_09': {}, '2025_10': {}, '2025_11': {}, '2025_12': {}}
months_to_predict = ['2025-08-01', '2025-09-01', '2025-10-01', '2025-11-01', '2025-12-01']

# Hapus data yang lag-nya NaN (bulan-bulan awal)
train_data = ts_data.dropna().copy()

features = [c for c in train_data.columns if c not in ['YearMonth', 'Date', 'Claim_Frequency', 'Total_Claim', 'Claim_Severity']]

print("Melatih model LightGBM dan memprediksi masa depan...")

# Iterasi secara Auto-Regressive (Prediksi bulan Ags, gunakan hasilnya untuk memprediksi Sep, dst)
current_ts_data = ts_data.copy()

for target in targets:
    # Set parameter LightGBM
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'learning_rate': 0.05,
        'max_depth': 5,
        'num_leaves': 15,
        'random_state': 42,
        'verbose': -1
    }
    
    for pred_date in months_to_predict:
        # Data latih adalah semua data SEBELUM bulan prediksi
        train_df = current_ts_data[current_ts_data['Date'] < pred_date].dropna()
        
        X_train = train_df[features]
        y_train = train_df[target]
        
        # Inisialisasi dan latih model
        model = lgb.LGBMRegressor(**params, n_estimators=100)
        model.fit(X_train, y_train)
        
        # Siapkan baris kosong untuk bulan yang diprediksi jika belum ada
        if not (current_ts_data['Date'] == pred_date).any():
            new_row = pd.DataFrame({'Date': [pd.to_datetime(pred_date)]})
            current_ts_data = pd.concat([current_ts_data, new_row], ignore_index=True)
            
        # Re-calculate features termasuk baris baru
        current_ts_data = create_features(current_ts_data)
        
        # Ambil fitur bulan yang mau diprediksi
        X_test = current_ts_data[current_ts_data['Date'] == pred_date][features]
        
        # Prediksi
        pred_value = model.predict(X_test)[0]
        
        # Jika memprediksi Frekuensi, karena 'sub_med_minus20' bekerja baik, 
        # kita tambahkan sedikit penalti tren menurun (misal: kalikan 0.9 atau biarkan model LightGBM mempelajarinya)
        # Di sini kita biarkan LightGBM yang menentukan berdasarkan lag
        
        # Masukkan prediksi kembali ke dataset agar bisa dipakai sebagai Lag untuk bulan depannya
        current_ts_data.loc[current_ts_data['Date'] == pred_date, target] = pred_value
        
        # Simpan untuk submission
        month_key = pred_date[:7].replace('-', '_')
        predictions[month_key][target] = pred_value

# ==========================================
# 6. FORMATTING FILE SUBMISSION
# ==========================================
submission_rows = []

for month_key, preds in predictions.items():
    freq = preds['Claim_Frequency']
    sev = preds['Claim_Severity']
    
    # RUMUS SAKTI: Kembalikan nilai ke Total. Pastikan konsisten!
    total = freq * sev 
    
    submission_rows.append({'id': f'{month_key}_Claim_Frequency', 'value': freq})
    submission_rows.append({'id': f'{month_key}_Claim_Severity', 'value': sev})
    submission_rows.append({'id': f'{month_key}_Total_Claim', 'value': total})

submission_df = pd.DataFrame(submission_rows)

# Simpan hasil akhir
submission_df.to_csv('submission_lgbm_optimized.csv', index=False)
print("File submission_lgbm_optimized.csv berhasil dibuat! Siap disubmit.")
print(submission_df.head(15))

#hooreeeee
print("kanarazu katsu, im tired")