import pandas as pd
import numpy as np
from sklearn.linear_model import BayesianRidge, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from statsmodels.tsa.statespace.sarimax import SARIMAX
import warnings
warnings.filterwarnings('ignore')

print("Membaca Data Klaim...")
df_klaim = pd.read_csv('Data_Klaim.csv')

# ==========================================
# 1. PONDASI DATA (STRICT BASE)
# ==========================================
date_col = 'Tanggal Pasien Masuk RS'
df_klaim[date_col] = pd.to_datetime(df_klaim[date_col], errors='coerce')
df_klaim = df_klaim.dropna(subset=[date_col])

df_klaim_paid = df_klaim[df_klaim['Status Klaim'] == 'PAID'].copy()

# Capping 98.0%
limit = df_klaim_paid['Nominal Klaim Yang Disetujui'].quantile(0.980)
df_klaim_paid['Nominal_Klaim_Capped'] = np.clip(df_klaim_paid['Nominal Klaim Yang Disetujui'], a_min=0, a_max=limit)

df_klaim_paid['YearMonth'] = df_klaim_paid[date_col].dt.to_period('M').astype(str)

monthly_data = df_klaim_paid.groupby('YearMonth').agg(
    Claim_Frequency=('Claim ID', 'count'),
    Total_Claim=('Nominal_Klaim_Capped', 'sum') 
).reset_index()

monthly_data['Date'] = pd.to_datetime(monthly_data['YearMonth'] + '-01')
monthly_data = monthly_data.sort_values('Date').reset_index(drop=True)
monthly_data = monthly_data.drop(columns=['YearMonth']).set_index('Date').resample('MS').interpolate(method='linear').reset_index()

# Filter >= 2022
monthly_data = monthly_data[monthly_data['Date'] >= '2022-01-01'].reset_index(drop=True)

# ==========================================
# 2. IBNR 1.4x LAUNCHPAD
# ==========================================
max_idx = len(monthly_data) - 1
last_freq = monthly_data.loc[max_idx, 'Claim_Frequency']
prev_freq = monthly_data.loc[max_idx-1, 'Claim_Frequency']

if last_freq < 0.75 * prev_freq:
    print("\n[IBNR 1.4x AKTIF] Menyesuaikan data bulan terakhir...")
    monthly_data.loc[max_idx, 'Claim_Frequency'] *= 1.4
    monthly_data.loc[max_idx, 'Total_Claim'] *= 1.4

monthly_data['Total_Claim_B'] = monthly_data['Total_Claim'] / 1e9

# ==========================================
# 3. GLOBAL SARIMA (INDUSTRY STANDARD)
# ==========================================
print("\nMelatih Global SARIMA (1,1,1)x(0,1,1,12)...")
months_to_predict = ['2025-08-01', '2025-09-01', '2025-10-01', '2025-11-01', '2025-12-01']
submission_months = ['2025_08', '2025_09', '2025_10', '2025_11', '2025_12']

sarima_preds = {}
for target in ['Claim_Frequency', 'Total_Claim_B']:
    y_vals = monthly_data[target].values
    try:
        model_sarima = SARIMAX(
            y_vals, 
            order=(1, 1, 1), 
            seasonal_order=(0, 1, 1, 12),
            enforce_stationarity=False, 
            enforce_invertibility=False
        ).fit(disp=False)
        sarima_preds[target] = model_sarima.forecast(5)
    except Exception as e:
        print(f"SARIMA failed for {target}, using fallback. Error: {e}")
        sarima_preds[target] = np.repeat(y_vals[-1], 5)

# ==========================================
# 4. KINEMATIC FEATURES (LAG-12 INTEGRATION)
# ==========================================
def create_features(df):
    df = df.copy()
    df['Month'] = df['Date'].dt.month
    df['Time_Index'] = np.arange(1, len(df) + 1)
    
    for col in ['Claim_Frequency', 'Total_Claim_B']:
        # KUNCI UTAMA: Lag 12 secara matematis memaksa AI melihat data tahun lalu
        for i in [1, 2, 3, 12]:
            df[f'{col}_lag_{i}'] = df[col].shift(i)
    return df

ts_data = create_features(monthly_data)

print("Melatih Model ML Rekursif (BayesianRidge & Ridge)...")
current_ts_data = ts_data.copy()
final_preds = {m: {} for m in submission_months}

for i, pred_date in enumerate(months_to_predict):
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
        
        ml_pred_val = (0.50 * bayes.predict(X_test)[0]) + (0.50 * ridge.predict(X_test)[0])
        
        sarima_val = sarima_preds[target][i]
        
        # 50/50 Blend: SARIMA (Statistical Seasonality) + ML (Lag-12 anchored)
        final_val = max(0, (0.50 * sarima_val) + (0.50 * ml_pred_val))
        
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
print("\n--- HASIL PREDIKSI (THE LAG-12 SARIMA) ---")

for month_key in submission_months:
    freq = final_preds[month_key].get('Claim_Frequency', 0)
    total = final_preds[month_key].get('Total_Claim', 0)
    
    sev = total / freq if freq > 0 else 0
    
    print(f"{month_key} -> Freq: {freq:.1f} | Sev: {sev:,.0f} | Total: {total:,.0f}")
    
    submission_rows.append({'id': f'{month_key}_Claim_Frequency', 'value': freq})
    submission_rows.append({'id': f'{month_key}_Claim_Severity', 'value': sev})
    submission_rows.append({'id': f'{month_key}_Total_Claim', 'value': total})

submission_df = pd.DataFrame(submission_rows)
submission_df.to_csv('submission_lag12_sarima.csv', index=False)
print("\n[LOCKED] File 'submission_lag12_sarima.csv' siap!")