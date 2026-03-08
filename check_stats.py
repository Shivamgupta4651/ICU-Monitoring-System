import pandas as pd

# 1. Load the numeric dataset
df = pd.read_csv('ICU_Preprocessed_Data.csv')

# 2. Vitals to check
vitals = ['Heart Rate (bpm)', 'Oxygen Saturation (%)', 'Systolic BP (mmHg)', 
          'Diastolic BP (mmHg)', 'Body Temperature (°C)', 'Derived_HRV', 
          'Derived_MAP', 'Derived_Pulse_Pressure']

# 3. Calculation
stats = pd.DataFrame()
stats['Mean'] = df[vitals].mean()
stats['Median'] = df[vitals].median()

# Difference Calculation (Agar Mean bada hai toh positive aayega)
stats['Difference'] = stats['Mean'] - stats['Median']

# Skewness check (Zero ke paas hona chahiye)
stats['Skewness'] = df[vitals].skew()

print("--- SHANYA'S CHECKLIST ---")
print(stats)

# Logic: Agar Difference 1-2 se zyada hai, toh cleaning ki zaroorat hai