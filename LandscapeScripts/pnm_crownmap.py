import os
import geopandas as gpd

path=r"D:\Arboles_PNM_export1\PNM_2024_crownmap.shp"
species=r"D:\BCI_50ha\aux_files\ListadoEspecies.csv"

# Load the shapefile
gdf = gpd.read_file(path)

gdf.columns

gdf= gdf[['GlobalID','CreationDa', 'Creator', 'EditDate',
       'Editor', 'Flowering', 'Tag', 'Category', 'Life_form', 'Notes',
       'Status', 'Observer', 'Species', 'Lianas', 'illuminati', 'Crown',
       'Dead_stand', 'New_leaves', 'Senecent_l', 'Fruiting', 'leafing',
       'Latin', 'geometry']]