import os
import pandas as pd
import geopandas as gpd

raw2022=gpd.read_file(r"D:\BCI_50ha\crownmap_datapub\BCI_50ha_2022_09_29_crownmap_raw\BCI_50ha_2022_2023_crownmap.shp")
raw2020=gpd.read_file(r"D:\BCI_50ha\crownmap_datapub\BCI_50ha_2020_08_01_crownmap_raw\Crowns_2020_08_01_MergedWithPlotData.shp")
temp=r"D:\BCI_50ha\BCI_50ha_2022_2023_crownmap_temp.shp"
#QAQC
# Create a mapping from 'mnemonic' to 'latin'
mapping = raw2022.dropna(subset=['latin']).set_index('mnemonic')['latin'].to_dict()
mapping2020 = raw2020.dropna(subset=['Latin']).set_index('Mnemonic')['Latin'].to_dict()
mapping.update(mapping2020)


raw2022['latin'] = raw2022['latin'].fillna(raw2022['mnemonic'].map(mapping))
mapping2 = raw2022.dropna(subset=['mnemonic']).set_index('latin')['mnemonic'].to_dict()
raw2022['mnemonic'] = raw2022['mnemonic'].fillna(raw2022['latin'].map(mapping2))


#if tag == -9999 then na else tag
raw2022['tag'] = raw2022['tag'].apply(lambda x: None if x == '-9999' else x)

mask = raw2022['tag'].isnull()
raw2022.loc[mask, 'latin'] = None
raw2022.loc[mask, 'mnemonic'] = None


#save a temporary file
raw2022.to_file(temp)



