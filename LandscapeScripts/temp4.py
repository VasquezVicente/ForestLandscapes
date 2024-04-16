import os 
import pandas as pd

path=r"\\stri-sm01\ForestLandscapes\LandscapeProducts\Drone\2024\BCI_50ha_2024_04_03_M3E\Project\BCI_50ha_2024_04_03_M3E_reference_orig.txt"
path2=r"\\stri-sm01\ForestLandscapes\LandscapeProducts\Drone\2024\BCI_50ha_2024_04_03_M3E\Project\BCI_50ha_2024_04_03_M3E_reference_ppk.txt"

# Read in both comma delimited files
df1 = pd.read_csv(path, delimiter=',', skiprows=1)
df2 = pd.read_csv(path2, delimiter=',', skiprows=1)

# Merge the dataframes on the '#Label' column
merged_df = pd.merge(df1, df2[['#Label', 'X/Easting', 'Y/Northing', 'Z/Altitude']], on='#Label', how='left')

# Replace X/Easting, Y/Northing, and Z/Altitude values
merged_df['X/Easting'] = merged_df['X/Easting_y'].fillna(merged_df['X/Easting_x'])
merged_df['Y/Northing'] = merged_df['Y/Northing_y'].fillna(merged_df['Y/Northing_x'])
merged_df['Z/Altitude'] = merged_df['Z/Altitude_y'].fillna(merged_df['Z/Altitude_x'])

# Drop the unnecessary columns from the merge operation
merged_df.drop(columns=['X/Easting_x', 'X/Easting_y', 'Y/Northing_x', 'Y/Northing_y', 'Z/Altitude_x', 'Z/Altitude_y'], inplace=True)

# Print the resulting dataframe
print(merged_df)
merged_df.to_csv(r"\\stri-sm01\ForestLandscapes\LandscapeProducts\Drone\2024\BCI_50ha_2024_04_03_M3E\Project\BCI_50ha_2024_04_03_M3E_reference_merged.txt", index=False)