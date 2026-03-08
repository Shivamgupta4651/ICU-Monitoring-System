import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 1. Load the dataset
df = pd.read_csv('Final_Dataset.csv')

# 2. Focus on the 11 key features
# (Excluding Gender and Fall Detection from stats as they are categorical)
vitals = ['Age', 'Heart Rate (bpm)', 'Oxygen Saturation (%)', 'Systolic BP (mmHg)', 
          'Diastolic BP (mmHg)', 'Body Temperature (°C)', 'Derived_HRV', 
          'Derived_MAP', 'Derived_Pulse_Pressure']

# 3. Statistical Calculation (Mean vs Median for Skewness)
stats = pd.DataFrame()
stats['Mean'] = df[vitals].mean()
stats['Median'] = df[vitals].median()
stats['Skewness'] = df[vitals].skew()

print("--- STATISTICAL SUMMARY ---")
print(stats)

# 4. Outlier Detection (IQR Method)
Q1 = df[vitals].quantile(0.25)
Q3 = df[vitals].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Counting outliers
outliers = ((df[vitals] < lower_bound) | (df[vitals] > upper_bound)).sum()
print("\n--- OUTLIERS DETECTED ---")
print(outliers)

# 5. Visualizing Outliers (Boxplot)
plt.figure(figsize=(12, 6))
sns.boxplot(data=df[vitals])
plt.xticks(rotation=45)
plt.title("Vitals Outlier Analysis")
plt.savefig('outlier_analysis_plot.png')

# 6. Cleaning: Removing Outliers
df_cleaned = df[~((df[vitals] < lower_bound) | (df[vitals] > upper_bound)).any(axis=1)]

# Save the polished dataset
df_cleaned.to_csv('Final_Dataset_Cleaned.csv', index=False)
print(f"\nSuccess! Cleaned data saved. Original: {len(df)} rows, Cleaned: {len(df_cleaned)} rows.")
