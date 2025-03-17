import labelbox as lb
import os
import pandas as pd

client = lb.Client(api_key="")
dataset = client.get_dataset("cm8bs9pgf00d40746btooascw")

data_path=r"\\stri-sm01\ForestLandscapes\UAVSHARE\BCI_50ha_timeseries"
path_out= os.path.join(data_path,'flower_dataset')

flowering_metadata=pd.read_csv(r'timeseries\dataset_corrections\check_flower1.csv')
flowering_metadata.columns

assets=[]
for index, row in flowering_metadata.iterrows():
    row_data_path = os.path.join(
        "\\\\stri-sm01\\ForestLandscapes\\UAVSHARE\\BCI_50ha_timeseries\\flower_dataset",
        f"{row['polygon_id']}.png"
    )
    dict_row = {
        'row_data': row_data_path,
        'global_key': row['globalId'],
        'media_type': 'IMAGE',
        'metadata_fields': [
            {"schema_id": "cm8bxtvk600q1074v0ad2cds7", "value": str(row['latin'])},
            {"schema_id": "cm8by8tfz08lh074lbwv5430o", "value": row['floweringIntensity']},
            {"schema_id": "cm8bydleg08dx0730cl2144z7", "value": row['isFlowering']},
            {"schema_id": "cm8bypte30abx075t61j4cvs7", "value": row['leafing']},
            {"schema_id": "cm8byrjeg08h8074mewqdf570", "value": row['date']}
        ]
    }
    assets.append(dict_row)


try:
    task = dataset.create_data_rows(assets)
    task.wait_till_done()
except Exception as err:
    print(f'Error while creating labelbox dataset -  Error: {err}')
    task.errors


#export the data
export_task = lb.ExportTask.get_task(client, "cm8cf844z0a5f075b4aqt3t31")

# Stream the export using a callback function
import ndjson
import json

classification_names = [
    'floweringIntensity', 'flowering_liana', 'isFlowering',
    'isFruiting', 'fruitingIntensity', 'segmentation',
    'newLeaves', 'leafing'
]
import pandas as pd
# Initialize list to store data rows
data_rows = []

# Open and process the NDJSON file
with open(r'timeseries\dataset_corrections\flowering_2025_03_16.ndjson') as f:
    for line in f:
        data = json.loads(line)
        data_row = data.get('data_row', {})
        global_key = data_row.get('global_key', 'N/A')

        metadata_fields = data.get('metadata_fields', [])
        date = 'N/A'
        for field in metadata_fields:
            if field.get('schema_name') == 'date_str':
                date = field.get('value')
                break

        # Initialize classification values
        classifications = {name: 'N/A' for name in classification_names}

        projects = data.get('projects', {})
        for project_details in projects.values():
            labels = project_details.get('labels', [])
            for label in labels:
                annotations = label.get('annotations', {})
                for classification in annotations.get('classifications', []):
                    name = classification.get('name')
                    if name in classification_names:
                        if 'text_answer' in classification:
                            content = classification['text_answer'].get('content')
                        elif 'radio_answer' in classification:
                            content = classification['radio_answer'].get('value')
                        elif 'checklist_answers' in classification:
                            content = [item.get('value') for item in classification['checklist_answers']]
                        else:
                            content = None
                        classifications[name] = content
        
        # Append row to the list
        data_row_dict = {'global_key': global_key, 'date': date}
        data_row_dict.update(classifications)
        data_rows.append(data_row_dict)

# Create a pandas DataFrame
import numpy as np
df = pd.DataFrame(data_rows)

df= df.rename(columns={'global_key':'globalId'})

df['floweringIntensity'] = pd.to_numeric(df['floweringIntensity'], errors='coerce')

# Convert 'leafing' to numeric, setting errors='coerce' to handle 'N/A' as NaN
df['leafing'] = pd.to_numeric(df['leafing'], errors='coerce')

# Convert 'isFlowering', 'isFruiting', 'segmentation', and 'newLeaves' to string, replacing 'N/A' with NaN
df['isFlowering'] = df['isFlowering'].replace('N/A', pd.NA).astype('string')
df['isFruiting'] = df['isFruiting'].replace('N/A', pd.NA).astype('string')
df['segmentation'] = df['segmentation'].replace('N/A', pd.NA).astype('string')
df['newLeaves'] = df['newLeaves'].replace('N/A', pd.NA).astype('string')

df= df[df['isFlowering'].notna()]
df.to_csv(r'timeseries/dataset_corrections/flower1.csv')