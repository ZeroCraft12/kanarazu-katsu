import pandas as pd
import numpy as np
from sklearn.linear_model import BayesianRidge, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import warnings
warnings.filterwarnings('ignore')

print("Membaca Data Klaim...")
df_klaim = pd.read_csv('Data_Klaim.csv')

# ==========================================
# 1. PONDASI DATA (BASE 3.6 YANG SUCI)
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

# HANYA gunakan data Pasca-Covid yang trennya stabil
monthly_data = monthly_data[monthly_data['Date'] >= '2022-01-01'].reset_index(drop=True)

# ==========================================
# 2. THE ORACLE DROP (MENCIPTAKAN DATA 100% SEMPURNA)
# ==========================================
months_to_predict = ['2025-08-01', '2025-09-01', '2025-10-01', '2025-11-01', '2025-12-01']
submission_months = ['2025_08', '2025_09', '2025_10', '2025_11', '2025_12']

max_idx = len(monthly_data) - 1
last_freq = monthly_data.loc[max_idx, 'Claim_Frequency']
prev_freq = monthly_data.loc[max_idx-1, 'Claim_Frequency']

# JIKA JULI ANJLOK KARENA RS TELAT REKAP -> HAPUS DARI DATA LATIH!
# Kita paksa AI hanya belajar dari data Januari 2022 sampai Juni 2025 yang sudah final (settled).
if last_freq < 0.8 * prev_freq:
    dropped_date_obj = monthly_data['Date'].iloc[-1]
    dropped_date = dropped_date_obj.strftime('%Y-%m-%d')
    print(f"\n[ORACLE DROP] Data bulan {dropped_date} belum lengkap (anjlok).")
    print("-> Menghapus bulan tersebut dari data latih! AI akan menambalnya.")
    monthly_data = monthly_data.iloc[:-1].reset_index(drop=True)
    
    # Masukkan bulan yang dihapus ke antrean prediksi
    if dropped_date not in months_to_predict:
        months_to_predict.insert(0, dropped_date)

monthly_data['Total_Claim_B'] = monthly_data['Total_Claim'] / 1e9

# ==========================================
# 3. KINEMATIC FEATURES (RINGAN & FOKUS)
# ==========================================
def create_features(df):
    df = df.copy()
    
    df['Time_Index'] = np.arange(1, len(df) + 1)
    df['Month'] = df['Date'].dt.month
    
    for col in ['Claim_Frequency', 'Total_Claim_B']:
        for i in [1, 2, 3]: 
            df[f'{col}_lag_{i}'] = df[col].shift(i)
        df[f'{col}_roll_mean_3'] = df[col].shift(1).rolling(window=3).mean()
        
    return df

ts_data = create_features(monthly_data)
targets = ['Claim_Frequency', 'Total_Claim_B']

# ==========================================
# 4. THE ANCHOR ORACLE ENSEMBLE (KUNCI < 3.0)
# ==========================================
print("\nMelatih Model Rekursif (The Anchor Oracle)...")
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
        train_df = temp_df[temp_df['Date'] < pred_date].dropna()
        X_train = train_df[features]
        y_train = train_df[target]
        
        X_test = temp_df[temp_df['Date'] == pred_date][features].fillna(0)
        
        # -------------------------------------------------------------
        # ALGORITMA 1 & 2: PENARIK TREN (Bayesian & Ridge)
        # -------------------------------------------------------------
        bayes = make_pipeline(StandardScaler(), BayesianRidge()).fit(X_train, y_train)
        ridge = make_pipeline(StandardScaler(), Ridge(alpha=1.0)).fit(X_train, y_train)
        
        pred_bayes = bayes.predict(X_test)[0]
        pred_ridge = ridge.predict(X_test)[0]
        
        # -------------------------------------------------------------
        # ALGORITMA 3: THE SEASONAL ANCHOR (Rahasia Skor 3.6)
        # -------------------------------------------------------------
        # Hitung indeks musiman dari data historis
        global_mean = y_train.mean()
        seasonal_indices = train_df.groupby('Month')[target].mean() / global_mean
        
        # Ambil nilai valid terakhir (Bulan lalu)
        last_valid_val = y_train.iloc[-1]
        
        # Terapkan indeks musiman pada nilai terakhir untuk mencegah garis meledak ke atas
        target_month = pd.to_datetime(pred_date).month
        seasonality_ratio = seasonal_indices.get(target_month, 1.0)
        
        # Ini adalah Jangkar (Anchor) yang akan menarik turun agresivitas Regresi Linier
        pred_anchor = last_valid_val * seasonality_ratio
        
        # -------------------------------------------------------------
        # THE GRAND BLEND (Menggabungkan Tren dan Jangkar)
        # -------------------------------------------------------------
        # 35% Bayes, 35% Ridge, 30% Seasonal Anchor (Pengaman Mutlak)
        final_val = (0.35 * pred_bayes) + (0.35 * pred_ridge) + (0.30 * pred_anchor)
        
        current_ts_data.loc[current_ts_data['Date'] == pred_date, target] = final_val
        
        if target == 'Total_Claim_B':
            final_val_real = final_val * 1e9
            target_name = 'Total_Claim'
        else:
            final_val_real = final_val
            target_name = 'Claim_Frequency'
            
        month_key = pred_date[:7].replace('-', '_')
        
        # HANYA simpan prediksi Agustus-Desember untuk hasil akhir
        if month_key in submission_months:
            final_preds[month_key][target_name] = final_val_real

# ==========================================
# 5. EXPORT HASIL
# ==========================================
submission_rows = []
print("\n--- HASIL PREDIKSI (THE ANCHOR ORACLE) ---")

for month_key in submission_months:
    freq = final_preds[month_key].get('Claim_Frequency', 0)
    total = final_preds[month_key].get('Total_Claim', 0)
    
    sev = total / freq if freq > 0 else 0
    
    print(f"{month_key} -> Freq: {freq:.1f} | Sev: {sev:,.0f} | Total: {total:,.0f}")
    
    submission_rows.append({'id': f'{month_key}_Claim_Frequency', 'value': freq})
    submission_rows.append({'id': f'{month_key}_Claim_Severity', 'value': sev})
    submission_rows.append({'id': f'{month_key}_Total_Claim', 'value': total})

submission_df = pd.DataFrame(submission_rows)
submission_df.to_csv('submission_anchor_oracle.csv', index=False)
print("\n[LOCKED] File 'submission_anchor_oracle.csv' siap!")
print("Rahasia 3.6 berhasil dibongkar dan direplikasi menjadi Anchor Matematis. GAS < 3.0!")