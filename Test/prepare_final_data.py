import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

# 1. Load data
df = pd.read_csv('Final_Dataset.csv')

# 2. String to Numeric (Shanya's requirement)
le = LabelEncoder()
df['Gender'] = le.fit_transform(df['Gender'])
df['Fall Detection'] = le.fit_transform(df['Fall Detection'])
df['Risk Category'] = le.fit_transform(df['Risk Category'])

# 3. Numeric Vitals for Outlier removal
vitals = ['Heart Rate (bpm)', 'Oxygen Saturation (%)', 'Systolic BP (mmHg)', 
          'Diastolic BP (mmHg)', 'Body Temperature (°C)', 'Derived_HRV', 
          'Derived_MAP', 'Derived_Pulse_Pressure']

print(f"Initial Rows: {len(df)}")

# 4. STRICT Z-SCORE CLEANING (Threshold = 2.5 instead of 3 for more tightness)
from scipy import stats
df_numeric = df[vitals]
z_scores = np.abs(stats.zscore(df_numeric))
df = df[(z_scores < 2.5).all(axis=1)]

# 5. Check Mean vs Median again
print("\n--- NEW STATS AFTER STRICT CLEANING ---")
for col in vitals:
    mean_val = df[col].mean()
    median_val = df[col].median()
    print(f"{col}: Mean={mean_val:.2f}, Median={median_val:.2f}, Diff={abs(mean_val-median_val):.2f}")

# 6. Save final numeric file
cols_to_keep = ['Age', 'Gender'] + vitals + ['Risk Category']
df[cols_to_keep].to_csv('ICU_Preprocessed_Data.csv', index=False)

print(f"\nFinal Rows: {len(df)}")
print("File Saved as ICU_Preprocessed_Data.csv")