import os
import labelbox
import pandas as pd
from labelbox import Option, Classification
from labelbox import Tool, Classification, Option
from labelbox import OntologyBuilder

##list all folders starting with tile

directory_path = r"\\stri-sm01\ForestLandscapes\UAVSHARE\Forrister_Yasuni_UAV\yasuni"
tile_folders = [name for name in os.listdir(directory_path)
                if os.path.isdir(os.path.join(directory_path, name)) and name.startswith("tile")]


#loop to get all jpgs
jpg_files = []

for folder in tile_folders:
    folder_path = os.path.join(directory_path, folder)
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith('.jpg'):
                jpg_files.append(os.path.join(root, file))


df = pd.DataFrame(jpg_files, columns=['path'])
df['tile'] = df['path'].str.split('\\').str[-3]
df['filename'] = df['path'].str.split('\\').str[-1]
df['polygon_id'] = df['filename'].str.extract(r'_V_(\d+)', expand=False)
df['is_zoom'] = df['filename'].str.contains('zoom', case=False)

df= df[df['is_zoom']==True]

##open the client
client = labelbox.Client(api_key=) ##add your key

#create the dataset
dataset = client.create_dataset(
  name='Yasuni Close-Ups v3',
  description='Close ups dataset',	# optional
  iam_integration=None		# if not specified, will use default integration, set as None to not use delegated access.
)

#get the dataset id
print(dataset)

#create the payload
assets=[]
for index, row in df.iterrows():
    row_data_path = row['path']
    dict_row = {
        'row_data': row_data_path,
        'global_key': row['filename'],
        'media_type': 'IMAGE',
        'metadata_fields': [
            {"schema_id": "cmbtmolre0kos07zb4i1sgxav", "value": row['polygon_id']},
            {"schema_id": "cmctpsz9503ah07vb3tv0gpjj", "value": row['tile']}
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


#create nested ontology
#readcsv
species=pd.read_csv(r"\\stri-sm01\ForestLandscapes\UAVSHARE\Forrister_Yasuni_UAV\yasuni.spp.file(1).csv", encoding='latin1')



grouped = species.dropna(subset=["Genus", "SpeciesName"]).groupby("Genus")


# --- Genus → Species nested classification ---
genus_with_species_classification = Classification(
    class_type=Classification.Type.RADIO,
    name="genus",
    options=[
        Option(
            value=str(genus),
            label=str(genus),
            options=[
                Classification(
                    class_type=Classification.Type.RADIO,
                    name="species",
                    options=[
                        Option(value=str(species), label=str(species))
                        for species in group["SpeciesName"].dropna().unique()
                    ]
                )
            ]
        )
        for genus, group in grouped
    ]
)

# --- Family (flat radio classification) ---
family_classification = Classification(
    class_type=Classification.Type.RADIO,
    name="family",
    options=[
        Option(value=str(family), label=str(family))
        for family in species['Family'].dropna().unique()
    ]
)

# --- Mnemonic (flat radio classification) ---
mnemonic_classification = Classification(
    class_type=Classification.Type.RADIO,
    name="mnemonic",
    options=[
        Option(value=str(mnemonic), label=str(mnemonic))
        for mnemonic in species['Mnemonic'].dropna().unique()
    ]
)

# --- Liana Presence (boolean radio classification) ---
liana_presence_classification = Classification(
    class_type=Classification.Type.RADIO,
    name="liana_presence",
    options=[
        Option(value="yes", label="Yes"),
        Option(value="no", label="No")
    ]
)


segmentation_tool = Tool(
    tool=Tool.Type.RASTER_SEGMENTATION,
    name="tree-segmentation",
    color="#FF5733",
    classifications=[
        genus_with_species_classification,
        family_classification,
        mnemonic_classification,
        liana_presence_classification
    ]
)

ontology_builder = labelbox.OntologyBuilder(tools=[
    segmentation_tool
])

ontology = client.create_ontology("Yasuni ontology", ontology_builder.asdict(), media_type= labelbox.MediaType.Image)