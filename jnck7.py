import pandas as pd
import numpy as np
from sklearn.linear_model import BayesianRidge, RidgeCV, ElasticNetCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import warnings
warnings.filterwarnings('ignore')

print("Membaca Data Klaim...")
df_klaim = pd.read_csv('Data_Klaim.csv')

# ==========================================
# 1. PONDASI DATA (KEMBALI KE BASE MURNI 3.6)
# ==========================================
date_col = 'Tanggal Pasien Masuk RS'
df_klaim[date_col] = pd.to_datetime(df_klaim[date_col], errors='coerce')
df_klaim = df_klaim.dropna(subset=[date_col])

df_klaim_paid = df_klaim[df_klaim['Status Klaim'] == 'PAID'].copy()

# Capping di 98.0% (Kunci Absolut Penstabil Regresi)
limit = df_klaim_paid['Nominal Klaim Yang Disetujui'].quantile(0.980)
df_klaim_paid['Nominal_Klaim_Capped'] = np.clip(df_klaim_paid['Nominal Klaim Yang Disetujui'], a_min=0, a_max=limit)

df_klaim_paid['YearMonth'] = df_klaim_paid[date_col].dt.to_period('M').astype(str)

monthly_data = df_klaim_paid.groupby('YearMonth').agg(
    Claim_Frequency=('Claim ID', 'count'),
    Total_Claim=('Nominal_Klaim_Capped', 'sum') 
).reset_index()

monthly_data['Date'] = pd.to_datetime(monthly_data['YearMonth'] + '-01')
monthly_data = monthly_data.sort_values('Date').reset_index(drop=True)

# HANYA gunakan data Pasca-Covid
monthly_data = monthly_data[monthly_data['Date'] >= '2022-01-01'].reset_index(drop=True)

# ==========================================
# 2. THE ORACLE IBNR DROP (TEKNIK GRANDMASTER)
# ==========================================
months_to_predict = ['2025-08-01', '2025-09-01', '2025-10-01', '2025-11-01', '2025-12-01']
submission_months = ['2025_08', '2025_09', '2025_10', '2025_11', '2025_12']

max_idx = len(monthly_data) - 1
last_freq = monthly_data.loc[max_idx, 'Claim_Frequency']
prev_freq = monthly_data.loc[max_idx-1, 'Claim_Frequency']

# Jika bulan terakhir anjlok drastis, HAPUS dari data latih!
# Biarkan AI menebaknya ulang menggunakan data yang 100% sempurna.
if last_freq < 0.8 * prev_freq:
    dropped_date_obj = monthly_data['Date'].iloc[-1]
    dropped_date = dropped_date_obj.strftime('%Y-%m-%d')
    print(f"\n[ORACLE DROP] Data bulan {dropped_date} belum lengkap (anjlok).")
    print("-> Menghapus bulan tersebut dari data latih agar tidak merusak tren AI!")
    monthly_data = monthly_data.iloc[:-1].reset_index(drop=True)
    
    # Tambahkan bulan yang dihapus ke antrean prediksi agar AI menambalnya secara matematis
    if dropped_date not in months_to_predict:
        months_to_predict.insert(0, dropped_date)

monthly_data['Total_Claim_B'] = monthly_data['Total_Claim'] / 1e9

# ==========================================
# 3. KINEMATIC, FOURIER & DAMPED FEATURES
# ==========================================
def create_features(df):
    df = df.copy()
    
    df['Time_Index'] = np.arange(1, len(df) + 1)
    
    # KUNCI BARU: Log_Time!
    # Memungkinkan Regresi Linier untuk menarik garis melengkung (Damped Trend)
    # Persis seperti yang dilakukan Holt-Winters Damped tanpa risiko error library!
    df['Log_Time'] = np.log1p(df['Time_Index'])
    
    # Fourier menggantikan Holt-Winters Seasonal (Bebas Error/Crash)
    months = df['Date'].dt.month
    df['sin1'] = np.sin(2 * np.pi * months / 12)
    df['cos1'] = np.cos(2 * np.pi * months / 12)
    
    for col in ['Claim_Frequency', 'Total_Claim_B']:
        for i in [1, 2, 3]: # Lag standar yang aman
            df[f'{col}_lag_{i}'] = df[col].shift(i)
        df[f'{col}_roll_mean_3'] = df[col].shift(1).rolling(window=3).mean()
        
    return df

ts_data = create_features(monthly_data)
targets = ['Claim_Frequency', 'Total_Claim_B']

# ==========================================
# 4. THE ELASTIC ORACLE ENSEMBLE
# ==========================================
print("\nMelatih Model Rekursif (The Elastic Oracle)...")
current_ts_data = ts_data.copy()
final_preds = {m: {} for m in submission_months}

for i, pred_date in enumerate(months_to_predict):
    
    if not (current_ts_data['Date'] == pred_date).any():
        new_row = pd.DataFrame({'Date': [pd.to_datetime(pred_date)]})
        current_ts_data = pd.concat([current_ts_data, new_row], ignore_index=True)
        
    temp_df = create_features(current_ts_data)
    
    exclude_cols = ['YearMonth', 'Date', 'Claim_Frequency', 'Total_Claim', 'Total_Claim_B']
    features = [c for c in temp_df.columns if c not in exclude_cols]
    
    for target in targets:
        # AI HANYA BELAJAR DARI DATA YANG 100% SUDAH SETTLED
        train_df = temp_df[temp_df['Date'] < pred_date].dropna()
        X_train = train_df[features]
        y_train = train_df[target]
        
        X_test = temp_df[temp_df['Date'] == pred_date][features].fillna(0)
        
        # TRINITAS BARU: Membuang Huber yang menekan lonjakan, menggantinya dengan ElasticNetCV
        bayes = make_pipeline(StandardScaler(), BayesianRidge()).fit(X_train, y_train)
        ridge = make_pipeline(StandardScaler(), RidgeCV(alphas=[0.1, 1.0, 5.0, 10.0])).fit(X_train, y_train)
        elastic = make_pipeline(StandardScaler(), ElasticNetCV(cv=3, l1_ratio=[0.1, 0.5, 0.9], random_state=42)).fit(X_train, y_train)
        
        pred_bayes = bayes.predict(X_test)[0]
        pred_ridge = ridge.predict(X_test)[0]
        pred_elastic = elastic.predict(X_test)[0]
        
        # THE DAMPED BLEND: 40% Bayes, 30% Ridge, 30% ElasticNet
        # AI kini mengizinkan lonjakan Q4 muncul dengan natural
        final_val = (0.40 * pred_bayes) + (0.30 * pred_ridge) + (0.30 * pred_elastic)
        
        current_ts_data.loc[current_ts_data['Date'] == pred_date, target] = final_val
        
        if target == 'Total_Claim_B':
            final_val_real = final_val * 1e9
            target_name = 'Total_Claim'
        else:
            final_val_real = final_val
            target_name = 'Claim_Frequency'
            
        month_key = pred_date[:7].replace('-', '_')
        
        # HANYA SIMPAN BULAN AGUSTUS - DESEMBER UNTUK SUBMISSION KAGGLE
        if month_key in submission_months:
            final_preds[month_key][target_name] = final_val_real

# ==========================================
# 5. EXPORT HASIL
# ==========================================
submission_rows = []
print("\n--- HASIL PREDIKSI (THE ELASTIC ORACLE) ---")

for month_key in submission_months:
    freq = final_preds[month_key].get('Claim_Frequency', 0)
    total = final_preds[month_key].get('Total_Claim', 0)
    
    sev = total / freq if freq > 0 else 0
    
    print(f"{month_key} -> Freq: {freq:.1f} | Sev: {sev:,.0f} | Total: {total:,.0f}")
    
    submission_rows.append({'id': f'{month_key}_Claim_Frequency', 'value': freq})
    submission_rows.append({'id': f'{month_key}_Claim_Severity', 'value': sev})
    submission_rows.append({'id': f'{month_key}_Total_Claim', 'value': total})

submission_df = pd.DataFrame(submission_rows)
submission_df.to_csv('submission_elastic_oracle.csv', index=False)
print("\n[LOCKED] File 'submission_elastic_oracle.csv' siap!")
print("Huber dihapus, ElasticNet & Damped Log_Time masuk. Lonjakan Q4 siap meledakkan skor < 3.0!")