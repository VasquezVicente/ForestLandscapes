
import geopandas as gpd
import pandas as pd
import numpy as np

path=r"C:\Users\VasquezV\repo\data\BCI_50ha_2022_2023_crownmap_temp.shp"
raw=gpd.read_file(path)

# i need the geometry to be in utm 17 n
raw=raw.to_crs(epsg=32617)
raw["crown_area"]=raw.area
# Fill NaN values with 0 (or another placeholder value) before conversion

raw["tag"] = raw["tag"].fillna(0)
raw["tag"] = raw["tag"].astype(int)
raw["tag"] = raw["tag"].astype(str)


# Apply formatting based on the length of the string
raw["tag"] = raw["tag"].apply(lambda x: x.zfill(6) if len(x) <= 6 else x)

raw=raw[["tag","crown_area","crown","iluminatio","lianas","flowering","latin","mnemonic","dead","note","editdate","geometry"]]

oldtree = pd.read_csv(r"D:\helene_bci_2024-03\OldTrees.txt", sep='\t', dtype={'Tag': str})
oldtree = oldtree.rename(columns={"Tag": "tag"})

oldtree['StemTag'] = pd.to_numeric(oldtree['StemTag'], errors='coerce')
oldtree = oldtree[np.isnan(oldtree['StemTag']) | (oldtree['StemTag'] == 1)]
oldtree_unique = oldtree.drop_duplicates(subset='tag', keep='first')

raw1 = pd.merge(raw, oldtree_unique[['tag', 'mnemonic']], on='tag', how='left')
raw1["mnemonic"] = np.where(raw1["mnemonic_y"].notna(), raw1["mnemonic_y"], raw1["mnemonic_x"])

for index, row in raw1.iterrows():
    if pd.isna(row['mnemonic']):
        print(f"Row {index}: tag = {row['tag']}, mnemonic is NaN")

raw1 = raw1.drop(columns=['mnemonic_x', 'mnemonic_y','latin'])

species=pd.read_csv(r"D:\BCI_50ha\aux_files\ListadoEspecies.csv")
species["latin"] = species["genus"] + " " + species["speciesname"]

species=species.rename(columns={"spp": "mnemonic"})
mnemonic_to_latin = species.set_index('mnemonic')['latin'].to_dict()

raw1['latin'] = raw1['mnemonic'].map(mnemonic_to_latin)
#now we need the dbh and the dbh_date from oldtree_unique
oldtree_unique = oldtree_unique.rename(columns={"DBH": "dbh", "exactdate": "dbh_date"})
oldtree_unique.columns

raw3 = pd.merge(raw1, oldtree_unique[['tag', 'dbh', 'dbh_date']], on='tag', how='left')

raw3.to_file(r"D:\BCI_50ha\crownmap\BCI_50ha_2022_2023_crownmap_raw.shp")
print(type(raw3))