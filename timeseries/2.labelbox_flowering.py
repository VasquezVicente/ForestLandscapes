import labelbox
import os
import pandas as pd

client = labelbox.Client(api_key="")

#get the dataset that is currently in labelbox
dataset = client.get_dataset("cm8bs9pgf00d40746btooascw")

#export the dataset that is currently in labelbox
export_task = dataset.export()
export_task.wait_till_done()

#function to parse the labelbox object to a json
def json_stream_handler(output: labelbox.BufferedJsonConverterOutput):
  print(output.json)

#parse the labelbox object
stream=export_task.get_buffered_stream(stream_type=labelbox.StreamType.RESULT).start(stream_handler=json_stream_handler)

#get the names that are already in labelbox
export_json = [data_row.json for data_row in export_task.get_buffered_stream()]
global_keys = [item["data_row"]["external_id"] for item in export_json]

#path to the raw flower dataset hosted in ForestLandscapes server
data_path=r"\\stri-sm01\ForestLandscapes\UAVSHARE\BCI_50ha_timeseries"
path_out= os.path.join(data_path,'flower_dataset')
list_flower = [os.path.join(path_out, filename) for filename in os.listdir(path_out)]

#determine the ones the flowering trees that are in the server and not in labelbox
extra_files = [f for f in list_flower if f not in global_keys]
extra_file_ids = [os.path.basename(f) for f in extra_files]

#read in the raw flowering dataset
flowering_metadata=pd.read_csv(r'timeseries\dataset_corrections\flower.csv')
flowering_metadata['polygon_id']=flowering_metadata['polygon_id']+".png"
extra_metadata_rows = flowering_metadata[flowering_metadata['polygon_id'].isin(extra_file_ids)]
#drop doubly labeled flower
extra_metadata_rows = extra_metadata_rows.drop_duplicates(subset='polygon_id', keep=False)


# create an assets package for labelbox
assets=[]
for index, row in extra_metadata_rows.iterrows():
    row_data_path = os.path.join(
        "\\\\stri-sm01\\ForestLandscapes\\UAVSHARE\\BCI_50ha_timeseries\\flower_dataset",
        f"{row['polygon_id']}"
    )
    dict_row = {
        'row_data': row_data_path,
        'global_key': row['polygon_id'],
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

#push the data rows to labelbox
try:
    task = dataset.create_data_rows(assets)
    task.wait_till_done()
except Exception as err:
    print(f'Error while creating labelbox dataset -  Error: {err}')
    task.errors



#bring back the exports after labelling
project = client.get_project("cm8azfo2f037h074jgzid05f9")
export_task = project.export()
export_task.wait_till_done()

stream=export_task.get_buffered_stream(stream_type=labelbox.StreamType.RESULT).start(stream_handler=json_stream_handler)

export_json = [data_row.json for data_row in export_task.get_buffered_stream()]

global_keys = [item["data_row"]["external_id"] for item in export_json]



data = [] 
for row in export_json:
    polygon_id = row["data_row"]["external_id"]  # Extract polygon ID
    row_data = {"polygon_id": polygon_id}
    for project_id, project_data in row['projects'].items():  # every project is a columns
        for label in project_data['labels']:  # Access labels
            classifications = label['annotations']['classifications']
            
            for classification in classifications:
                if classification['name'] == 'newLeaves':  # Look for "newLeaves"
                    row_data['newLeaves']= classification['radio_answer']['value']
                elif classification['name'] == 'isFlowering':
                    row_data['isFlowering']= classification['radio_answer']['value']
                elif classification['name'] == 'isFruiting':
                    row_data['isFruiting']= classification['radio_answer']['value']
                elif classification['name']== 'segmentation':
                    row_data['segmentation']=classification['radio_answer']['value']
                elif classification['name'] == 'leafing':  # Fixed condition
                    row_data['leafing'] = float(classification['text_answer']['content'])# Fixed access
                elif classification['name']== 'floweringIntensity':
                    row_data['floweringIntensity']=float(classification['text_answer']['content'])
                elif classification['name'] == 'flowering_liana':
                    row_data['flowering_liana'] = [answer['name'] for answer in classification['checklist_answers']][0]
    data.append(row_data)
     
flowering_dataset= pd.DataFrame(data)
flowering_dataset['polygon_id'] = flowering_dataset['polygon_id'].apply(os.path.basename)
flowering_dataset['polygon_id'] = flowering_dataset['polygon_id'].apply(lambda x: x.split(".")[0])

flowering_dataset=flowering_dataset[~flowering_dataset['leafing'].isna()]


flowering_dataset['isFlowering']= flowering_dataset.apply(lambda x: if x.is notna()  then x if na and floweringIntesity >0 then yes)

flowering_dataset['isFlowering'] = flowering_dataset.apply(
    lambda x: x['isFlowering'] if pd.notna(x['isFlowering']) else ('yes' if x['floweringIntensity'] > 0 else 'no'),
    axis=1
)


flowering_dataset.to_csv('timeseries/dataset_corrections/flower_out.csv')
