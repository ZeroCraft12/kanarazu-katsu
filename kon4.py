import pandas as pd
import numpy as np
from sklearn.linear_model import BayesianRidge, Ridge, HuberRegressor
from sklearn.preprocessing import RobustScaler
from statsmodels.tsa.holtwinters import Holt
import warnings
warnings.filterwarnings('ignore')

print("Membaca Data Klaim...")
df_klaim = pd.read_csv('Data_Klaim.csv')

# ==========================================
# 1. PONDASI DATA (BASE 3.6 YANG SOLID)
# ==========================================
date_col = 'Tanggal Pasien Masuk RS'
df_klaim[date_col] = pd.to_datetime(df_klaim[date_col], errors='coerce')
df_klaim = df_klaim.dropna(subset=[date_col])

df_klaim_paid = df_klaim[df_klaim['Status Klaim'] == 'PAID'].copy()

# Capping di 95.0% (Lebih ketat memotong outlier ekstrem agar tren lebih mudah dibaca)
limit = df_klaim_paid['Nominal Klaim Yang Disetujui'].quantile(0.950)
df_klaim_paid['Nominal_Klaim_Capped'] = np.clip(df_klaim_paid['Nominal Klaim Yang Disetujui'], a_min=0, a_max=limit)

df_klaim_paid['YearMonth'] = df_klaim_paid[date_col].dt.to_period('M').astype(str)

monthly_data = df_klaim_paid.groupby('YearMonth').agg(
    Claim_Frequency=('Claim ID', 'count'),
    Total_Claim=('Nominal_Klaim_Capped', 'sum') 
).reset_index()

monthly_data['Date'] = pd.to_datetime(monthly_data['YearMonth'] + '-01')
monthly_data = monthly_data.sort_values('Date').reset_index(drop=True)

# Interpolasi untuk menambal data masa lalu yang mungkin bolong
monthly_data = monthly_data.drop(columns=['YearMonth']).set_index('Date').resample('MS').interpolate(method='linear').reset_index()
monthly_data['YearMonth'] = monthly_data['Date'].dt.to_period('M').astype(str)

monthly_data = monthly_data[monthly_data['Date'] >= '2022-01-01'].reset_index(drop=True)

# ==========================================
# 2. THE SOFT LANDING IMPUTATION
# ==========================================
max_idx = len(monthly_data) - 1
last_freq = monthly_data.loc[max_idx, 'Claim_Frequency']
prev_freq = monthly_data.loc[max_idx-1, 'Claim_Frequency']

# Jika Juli anjlok, JANGAN buang dan JANGAN dikali 1.4. 
# Ganti dengan rata-rata stabil dari Mei dan Juni! Ini adalah batu loncatan teraman.
if last_freq < 0.8 * prev_freq:
    print("\n[SOFT LANDING AKTIF] Bulan terakhir anjlok. Mengganti dengan rata-rata 2 bulan sebelumnya...")
    mean_freq = monthly_data['Claim_Frequency'].iloc[-3:-1].mean()
    mean_total = monthly_data['Total_Claim'].iloc[-3:-1].mean()
    
    monthly_data.loc[max_idx, 'Claim_Frequency'] = mean_freq
    monthly_data.loc[max_idx, 'Total_Claim'] = mean_total

monthly_data['Total_Claim_B'] = monthly_data['Total_Claim'] / 1e9

# ==========================================
# 3. KINEMATIC FEATURES
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
submission_months = ['2025_08', '2025_09', '2025_10', '2025_11', '2025_12']
months_to_predict = ['2025-08-01', '2025-09-01', '2025-10-01', '2025-11-01', '2025-12-01']

# ==========================================
# 4. THE ROBUST DAMPED ENSEMBLE
# ==========================================
print("\nMelatih Model Rekursif (The Robust Damped Actuary)...")
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
        # ALGORITMA 1: HOLT DAMPED TREND (Sang Pembuat Skor 3.6)
        # Menggunakan Holt murni (tanpa musiman) menjamin tidak akan pernah error
        # -------------------------------------------------------------
        try:
            holt_model = Holt(y_train.values, damped_trend=True, initialization_method='estimated').fit(optimized=True)
            pred_holt = holt_model.forecast(1)[0]
        except:
            pred_holt = y_train.values[-1]
            
        # -------------------------------------------------------------
        # ALGORITMA 2 & 3 & 4: THE ROBUST LINEAR SQUAD
        # Menggunakan RobustScaler: AI menjadi buta terhadap pencilan/outlier ekstrem!
        # -------------------------------------------------------------
        from sklearn.pipeline import make_pipeline
        
        # BayesianRidge: Sangat stabil
        bayes = make_pipeline(RobustScaler(), BayesianRidge()).fit(X_train, y_train)
        # Heavy Ridge: Alpha=10.0 memaksa garis kaku, menolak lonjakan drastis
        ridge = make_pipeline(RobustScaler(), Ridge(alpha=10.0)).fit(X_train, y_train)
        # Huber: Fokus pada median data
        huber = make_pipeline(RobustScaler(), HuberRegressor(epsilon=1.35)).fit(X_train, y_train)
        
        pred_bayes = bayes.predict(X_test)[0]
        pred_ridge = ridge.predict(X_test)[0]
        pred_huber = huber.predict(X_test)[0]
        
        # -------------------------------------------------------------
        # THE FINAL BLEND
        # 30% Holt Damped (Garis Mulus), 30% Bayes (Linier), 20% Heavy Ridge, 20% Huber
        # Kombinasi ini MENGUNCI prediksi agar tidak meledak ke atas maupun jatuh ke bawah.
        # -------------------------------------------------------------
        final_val = (0.30 * pred_holt) + (0.30 * pred_bayes) + (0.20 * pred_ridge) + (0.20 * pred_huber)
        
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
print("\n--- HASIL PREDIKSI (THE ROBUST DAMPED ACTUARY) ---")

for month_key in submission_months:
    freq = final_preds[month_key].get('Claim_Frequency', 0)
    total = final_preds[month_key].get('Total_Claim', 0)
    
    sev = total / freq if freq > 0 else 0
    
    print(f"{month_key} -> Freq: {freq:.1f} | Sev: {sev:,.0f} | Total: {total:,.0f}")
    
    submission_rows.append({'id': f'{month_key}_Claim_Frequency', 'value': freq})
    submission_rows.append({'id': f'{month_key}_Claim_Severity', 'value': sev})
    submission_rows.append({'id': f'{month_key}_Total_Claim', 'value': total})

submission_df = pd.DataFrame(submission_rows)
submission_df.to_csv('submission_robust_damped.csv', index=False)
print("\n[LOCKED] File 'submission_robust_damped.csv' siap!")
print("RobustScaler, Soft Landing, dan Holt Damped aktif. Ini adalah kestabilan absolut. GAS < 3.0!")