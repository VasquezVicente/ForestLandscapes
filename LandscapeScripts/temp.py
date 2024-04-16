import Metashape
import os
reference1= r"\\stri-sm01\ForestLandscapes\LandscapeRaw\Drone\2024\BCI_50ha_2024_03_06_M3E\reference.txt"
reference2=r"\\stri-sm01\ForestLandscapes\LandscapeRaw\Drone\2024\BCI_50ha_2024_03_06_M3E\reference2.txt"
output=r"\\stri-sm01\ForestLandscapes\LandscapeRaw\Drone\2024\BCI_50ha_2024_03_06_M3E\result.txt"
import pandas as pd
import numpy as np

reference1_table = pd.read_csv(reference1, sep=',', skiprows=1)
print(reference1_table.head())
reference2_table = pd.read_csv(reference2, sep=',', skiprows=1)
print(reference2_table.head())

merged_table= reference2_table
merged_table["X/Longitude"] = merged_table['#Label'].map(reference1_table.set_index('#Label')["X/Longitude"])
merged_table["Y/Latitude"] = merged_table['#Label'].map(reference1_table.set_index('#Label')["Y/Latitude"])
merged_table["Z/Altitude"] = merged_table['#Label'].map(reference1_table.set_index('#Label')["Z/Altitude"])

merged_table.to_csv(output, index=False)