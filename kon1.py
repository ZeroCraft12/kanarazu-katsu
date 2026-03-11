import pandas as pd
import numpy as np
from sklearn.linear_model import BayesianRidge, Ridge
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
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

# Capping di 98.0% (Kunci Absolut Penstabil Regresi dari versi 3.6)
limit = df_klaim_paid['Nominal Klaim Yang Disetujui'].quantile(0.980)
df_klaim_paid['Nominal_Klaim_Capped'] = np.clip(df_klaim_paid['Nominal Klaim Yang Disetujui'], a_min=0, a_max=limit)

df_klaim_paid['YearMonth'] = df_klaim_paid[date_col].dt.to_period('M').astype(str)

monthly_data = df_klaim_paid.groupby('YearMonth').agg(
    Claim_Frequency=('Claim ID', 'count'),
    Total_Claim=('Nominal_Klaim_Capped', 'sum') 
).reset_index()

monthly_data['Date'] = pd.to_datetime(monthly_data['YearMonth'] + '-01')
monthly_data = monthly_data.sort_values('Date').reset_index(drop=True)

# INTERPOLASI: Menjamin rantai waktu 100% tersambung agar Holt-Winters tidak crash!
monthly_data = monthly_data.drop(columns=['YearMonth']).set_index('Date').resample('MS').interpolate(method='linear').reset_index()
monthly_data['YearMonth'] = monthly_data['Date'].dt.to_period('M').astype(str)

# HANYA gunakan data Pasca-Covid
monthly_data = monthly_data[monthly_data['Date'] >= '2022-01-01'].reset_index(drop=True)

# ==========================================
# 2. THE ORACLE IBNR DROP (DATA TERBAIK 4.0)
# ==========================================
months_to_predict = ['2025-08-01', '2025-09-01', '2025-10-01', '2025-11-01', '2025-12-01']
submission_months = ['2025_08', '2025_09', '2025_10', '2025_11', '2025_12']

max_idx = len(monthly_data) - 1
last_freq = monthly_data.loc[max_idx, 'Claim_Frequency']
prev_freq = monthly_data.loc[max_idx-1, 'Claim_Frequency']

# Membuang bulan terakhir (Juli) jika anjlok agar AI belajar murni dari tren yang sudah matang
if last_freq < 0.8 * prev_freq:
    dropped_date_obj = monthly_data['Date'].iloc[-1]
    dropped_date = dropped_date_obj.strftime('%Y-%m-%d')
    print(f"\n[ORACLE DROP] Data bulan {dropped_date} belum lengkap (anjlok).")
    print("-> Menghapus bulan tersebut dari data latih. AI akan menambalnya!")
    monthly_data = monthly_data.iloc[:-1].reset_index(drop=True)
    
    # Masukkan ke antrean tebakan
    if dropped_date not in months_to_predict:
        months_to_predict.insert(0, dropped_date)

monthly_data['Total_Claim_B'] = monthly_data['Total_Claim'] / 1e9

# ==========================================
# 3. KINEMATIC FEATURES (MESIN TERBAIK 3.6)
# ==========================================
def create_features(df):
    df = df.copy()
    
    # Kembali menggunakan eskalator natural (1-12) yang sangat disukai Ridge
    df['Month'] = df['Date'].dt.month
    
    for col in ['Claim_Frequency', 'Total_Claim_B']:
        # Lag 6 dikembalikan! Sangat krusial untuk memori musiman.
        for i in [1, 2, 3, 6]: 
            df[f'{col}_lag_{i}'] = df[col].shift(i)
        df[f'{col}_roll_mean_3'] = df[col].shift(1).rolling(window=3).mean()
        
    return df

ts_data = create_features(monthly_data)
targets = ['Claim_Frequency', 'Total_Claim_B']

# ==========================================
# 4. THE 3.6 ORACLE HYBRID ENSEMBLE
# ==========================================
print("\nMelatih Model Rekursif (The 3.6 Oracle Hybrid)...")
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
        
        # 1. HOLT-WINTERS DAMPED (DILATIH REKURSIF 1-STEP)
        # Menjamin garis batas tidak akan meledak, namun bebas crash karena data sudah mulus!
        try:
            hw_model = ExponentialSmoothing(y_train.values, trend='add', damped_trend=True, initialization_method='estimated').fit(optimized=True)
            pred_hw = hw_model.forecast(1)[0]
        except:
            pred_hw = y_train.values[-1]
            
        # 2 & 3. BAYESIAN RIDGE & RIDGE (Penarik tren utama)
        bayes = make_pipeline(StandardScaler(), BayesianRidge()).fit(X_train, y_train)
        ridge = make_pipeline(StandardScaler(), Ridge(alpha=1.0)).fit(X_train, y_train)
        
        pred_bayes = bayes.predict(X_test)[0]
        pred_ridge = ridge.predict(X_test)[0]
        
        # BLEND SANG JUARA (40% Bayes, 30% Ridge, 30% HW Damped)
        final_val = (0.40 * pred_bayes) + (0.30 * pred_ridge) + (0.30 * pred_hw)
        
        current_ts_data.loc[current_ts_data['Date'] == pred_date, target] = final_val
        
        if target == 'Total_Claim_B':
            final_val_real = final_val * 1e9
            target_name = 'Total_Claim'
        else:
            final_val_real = final_val
            target_name = 'Claim_Frequency'
            
        month_key = pred_date[:7].replace('-', '_')
        
        if month_key in submission_months:
            final_preds[month_key][target_name] = final_val_real

# ==========================================
# 5. EXPORT HASIL
# ==========================================
submission_rows = []
print("\n--- HASIL PREDIKSI (THE 3.6 ORACLE HYBRID) ---")

for month_key in submission_months:
    freq = final_preds[month_key].get('Claim_Frequency', 0)
    total = final_preds[month_key].get('Total_Claim', 0)
    
    sev = total / freq if freq > 0 else 0
    
    print(f"{month_key} -> Freq: {freq:.1f} | Sev: {sev:,.0f} | Total: {total:,.0f}")
    
    submission_rows.append({'id': f'{month_key}_Claim_Frequency', 'value': freq})
    submission_rows.append({'id': f'{month_key}_Claim_Severity', 'value': sev})
    submission_rows.append({'id': f'{month_key}_Total_Claim', 'value': total})

submission_df = pd.DataFrame(submission_rows)
submission_df.to_csv('submission_3_6_oracle_hybrid.csv', index=False)
print("\n[LOCKED] File 'submission_3_6_oracle_hybrid.csv' siap!")
print("Mesin 3.6 + Data 4.0 berhasil digabungkan. Ini adalah wujud paling sempurna. GAS < 3.0!")    