import os
import pandas as pd

labels_path= r"timeseries\labels"
list_files = os.listdir(labels_path)

all_csv=[]
for file_csv in list_files:
    if file_csv.endswith(".csv"):
        print(file_csv)
    label_file=pd.read_csv(labels_path + "\\" + file_csv)
    all_csv.append(label_file)
labels = pd.concat(all_csv)

labels_grouped = labels.groupby(['GlobalID','date']).agg({'leafing': lambda x: list(x)})


def determine_final_label(labels_list):
    if len(labels_list) == 1:
        return labels_list[0]
    counts = pd.Series(labels_list).value_counts()
    if counts.iloc[0] > len(labels_list) / 2:
        return counts.index[0]
    return labels_list[0]

# Apply the function to the grouped labels
labels_grouped['leafing_label'] = labels_grouped['leafing'].apply(determine_final_label)

# Optionally, drop the 'leafing' column if you only want the final label
labels_grouped = labels_grouped.drop(columns=['leafing'])

#export
labels_grouped.to_csv(r"timeseries/Labels_Log2_grouped.csv")


## open predicted df
predicted_crowns=r"C:\Users\Vicente\Downloads\predictedDf.csv"
predicted_df=pd.read_csv(predicted_crowns)
predicted_df['leafing_annotated']= predicted_df['leafing_final']
predicted_df.columns


def determine_leafing_status(mean_value):
    if mean_value <= 2:
        return "Fully Leafed"
    elif 2 < mean_value <= 2.5:
        return "Partially Leafed"
    else:
        return "Out of Leafs"

predicted_df['leafing_final'] = predicted_df['mean'].apply(determine_leafing_status)
print(predicted_df[['GlobalID','date','leafing_final']])

predicted_labels_out= predicted_df[['GlobalID','date','leafing_final']]
##summarize leafing_final
category_counts = predicted_labels_out['leafing_final'].value_counts()
print(category_counts)
predicted_labels_out.to_csv(r"C:\Users\Vicente\Downloads\predicted_labels_out.csv")


predicted_labels= predicted_df[['GlobalID','date','leafing_final','latin','mean']]


ceiba_test=predicted_labels[predicted_labels['latin']=="Ceiba pentandra"]
ceiba_test['date'] = pd.to_datetime(ceiba_test['date'], format='%Y_%m_%d')


import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(10, 6))

for name, group in ceiba_test.groupby('GlobalID'):
    ax.plot(group['date'], group['mean'], marker='o', linestyle='-')

# Add title and labels
plt.title('Mean Leafing Over Time for Ceiba pentandra by GlobalID')
plt.xlabel('Date')
plt.ylabel('Mean Leafing')

# Rotate x-axis labels for better readability
plt.xticks(rotation=45)

# Show plot
plt.tight_layout()
plt.show()


from pygam import LinearGAM, s

fig, ax = plt.subplots(figsize=(10, 6))

for name, group in ceiba_test.groupby('GlobalID'):
    group = group.sort_values('date')
    dates_numeric = (group['date'] - group['date'].min()).dt.days  # Convert dates to numeric
    gam = LinearGAM(s(0)).fit(dates_numeric[:, None], group['mean'])
    ax.plot(group['date'], gam.predict(dates_numeric[:, None]), linestyle='-', alpha=0.5)  # Add alpha for better visibility

# Add title and labels
plt.title('GAM Fit for Mean Leafing Over Time for Ceiba pentandra')
plt.xlabel('Date')
plt.ylabel('Fitted Mean Leafing')

# Rotate x-axis labels for better readability
plt.xticks(rotation=45)

# Improve layout and show the plot
plt.tight_layout()
plt.show()





import matplotlib.pyplot as plt
import pandas as pd

counts = ceiba_test.groupby(['date', 'leafing_final']).size().reset_index(name='count')
for row in counts.iterrows():
    print(row)

# Pivot the data to get counts for each category per date
pivot_counts = counts.pivot(index='date', columns='leafing_final', values='count').fillna(0)

# Sort by date to ensure chronological order
pivot_counts = pivot_counts.sort_index()

# Plot the lines for each category
fig, ax = plt.subplots(figsize=(15, 8))

for category in pivot_counts.columns:
    ax.plot(
        pivot_counts.index, 
        pivot_counts[category], 
        marker='o', 
        linestyle='-', 
        label=category
    )

# Add labels, title, and legend
ax.set_title('Counts of GlobalID by Leafing Category Over Time')
ax.set_xlabel('Date')
ax.set_ylabel('Count')
ax.legend(title="Leafing Category")
plt.xticks(rotation=45, ha='right')  # Rotate date labels for better readability

# Adjust layout and show plot
plt.tight_layout()
plt.show()