import pandas as pd
import numpy as np

# 1. Load the data
df = pd.read_csv('ICU_Preprocessed_Data.csv')

vitals = ['Heart Rate (bpm)', 'Oxygen Saturation (%)', 'Systolic BP (mmHg)', 
          'Diastolic BP (mmHg)', 'Body Temperature (°C)', 'Derived_HRV', 
          'Derived_MAP', 'Derived_Pulse_Pressure']

print(f"Initial rows: {len(df)}")

# 2. STRICT Percentile Capping (2.5% from both ends)
# Isse extreme high aur extreme low values dono hat jayengi
for col in vitals:
    lower = df[col].quantile(0.025)
    upper = df[col].quantile(0.975)
    df = df[(df[col] >= lower) & (df[col] <= upper)]

# 3. Final Stats Check
stats = pd.DataFrame()
stats['Mean'] = df[vitals].mean()
stats['Median'] = df[vitals].median()
stats['Diff'] = (stats['Mean'] - stats['Median']).abs()
stats['Skewness'] = df[vitals].skew()

print("\n--- FINAL POLISHED STATS ---")
print(stats)

# 4. Save Final Version
df.to_csv('ICU_Preprocessed_Data_v3.csv', index=False)
print(f"\nFinal rows: {len(df)}")