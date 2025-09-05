import labelbox
import os
import pandas as pd

def json_stream_handler(output: labelbox.BufferedJsonConverterOutput):
  print(output.json)

export_params= {
  "metadata_fields": True,
  "data_row_details": True,
  "project_details": True,
  "label_details": True,
}

client = labelbox.Client(api_key="")
project = client.get_project("cm8azfo2f037h074jgzid05f9")
export_task = project.export(params=export_params)
export_task.wait_till_done()
stream=export_task.get_buffered_stream(stream_type=labelbox.StreamType.RESULT).start(stream_handler=json_stream_handler)
export_json = [data_row.json for data_row in export_task.get_buffered_stream()]
global_keys = [item["data_row"]["external_id"] for item in export_json]


data = [] 
for row in export_json:
    #give me all the keys of row
    row_keys = list(row.keys())
    polygon_id = row["data_row"]["external_id"]  # Extract polygon ID
    row_data = {"polygon_id": polygon_id}
    status= row["project_details"]["workflow_status"]
    row_data['status']=status
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


flowering_dataset['isFlowering'] = flowering_dataset.apply(
    lambda x: x['floweringIntensity'] if pd.notna(x['floweringIntensity']) 
              else ('yes' if x['floweringIntensity'] > 0 else 'no'),
    axis=1
)


flowering_dataset.to_csv('timeseries/dataset_corrections/flower_out.csv')