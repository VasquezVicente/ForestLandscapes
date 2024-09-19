#phenlogy paper
import matplotlib.pyplot as plt
import os
import geopandas as gpd

timeseries=gpd.read_file(r"\\stri-sm01\ForestLandscapes\UAVSHARE\BCI_50ha_timeseries\BCI_50ha_crownmap_timeseries.shp")

species_count = timeseries.groupby('latin')['tag'].nunique().reset_index(name='count').sort_values(['count'], ascending=False)
abundant_species = species_count[species_count['count'] >= 50] #cut off point arbitrary
timeseries_abundant = timeseries[timeseries['latin'].isin(abundant_species['latin'])]
timeseries_abundant.to_file(r"\\stri-sm01\ForestLandscapes\UAVSHARE\BCI_50ha_timeseries\BCI_50ha_crownmap_timeseries_abundant.shp")
species_count=timeseries_abundant.groupby('latin')['tag'].nunique().reset_index(name='count').sort_values(['count'], ascending=False)
#determine the most abundant species in the dataset
species_count.set_index('latin')['count'].plot(kind='barh')
plt.title('Number of Individuals per Species')
plt.xlabel('Species')
plt.ylabel('Count')
plt.tight_layout()
plt.show()

#how many individual crowns to label
print('Number of individual crowns to label: ', timeseries_abundant.shape[0])

#lets see if i can plot the crown areas and see what happens
tags=timeseries_abundant['tag'].unique()
print('plotting crown: ', tags[0])
crown = timeseries_abundant[timeseries_abundant['tag'] == tags[1]]
crown['date'] = pd.to_datetime(crown['date'], format='%Y_%m_%d')

#crown['status'] = 'alive'
#crown.loc[crown['date'] > '2021-07-02', 'status'] = 'dead'

#plot area by date
plt.plot(crown['date'], crown['iou'], label='Area over Time')
scatter = plt.scatter(crown['date'], crown['iou'], c=crown['status'].astype('category').cat.codes, cmap='viridis')
handles, labels = scatter.legend_elements(prop="colors")
labels = crown['status'].astype('category').cat.categories
plt.legend(handles, labels, title="Status")
plt.title('Crown Area by Date, Tag: ' + tags[1])
plt.xlabel('Date')
plt.ylabel('Area')
plt.tight_layout()
plt.show()