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
predicted_labels_out.to_csv(r"C:\Users\Vicente\Downloads\predicted_labels_out.csv")


predicted_labels= predicted_df[['GlobalID','date','leafing_final','latin','mean']]

