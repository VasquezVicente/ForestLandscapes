import matplotlib.pyplot as plt
import os
import pandas as pd

labels=r'timeseries/50ha_timeseries_labels.csv'
labels= pd.read_csv(labels)


#plot of most labeled species
species_counts = labels["latin"].value_counts()
top_species = species_counts.head(10)
plt.figure(figsize=(8, 8))
plt.pie(top_species, labels=top_species.index, autopct='%1.1f%%', startangle=140, colors=plt.cm.Paired.colors)
plt.title("Most Labeled Species")
plt.show()

#plot individuals labeled by species
individual_counts = labels.groupby("latin")["globalId"].nunique().sort_values(ascending=False)
top_individuals = individual_counts.head(10)
plt.figure(figsize=(12, 10))
plt.bar(top_individuals.index, top_individuals.values, color='skyblue')
plt.xlabel("Species (Latin Name)")
plt.ylabel("Number of Unique Individuals Labeled")
plt.title("Top 10 Species by Labeled Individuals")
plt.xticks(rotation=15, ha="right")  # Rotate labels for better readability
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()


#flowering maybe 
labels_flowering_maybe= labels[labels['isFlowering']=="maybe"]