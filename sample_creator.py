import pandas as pd

# Define the file paths
original_data_path = r'C:\Users\Prathamesh\Downloads\Data_Science\IDS\dataset\ids_data.csv'
sample_data_path = r'C:\Users\Prathamesh\Downloads\Data_Science\IDS\dataset\new_traffic.csv'

# Read the original, large dataset
print(f"[*] Reading original data from: {original_data_path}")
df = pd.read_csv(original_data_path)

# Select the first 200 rows to create a sample
sample_df = df.head(200)

# Save the sample to a new CSV file
sample_df.to_csv(sample_data_path, index=False)

print(f"\n[+] Successfully created sample file with {len(sample_df)} rows.")
print(f"[+] Saved to: {sample_data_path}")