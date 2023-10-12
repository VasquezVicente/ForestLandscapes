import os
import pandas as pd
server_dir= r'\\stri-sm01\ForestLandscapes'

filepaths = pd.read_csv(os.path.join(server_dir, "filepaths.csv"))
def find_files(filepaths,type="LandscapeRaw", source="Drone",year="2023"):
    filepaths_1 = filepaths[(filepaths["type"] == type) & (filepaths["source"] == source) & (filepaths["year"] == year)]
    copy_rename= filepaths_1.copy()
    unique_missions = filepaths_1["mission"].unique()
    # Enumerate through the unique missions
    for i, mission in enumerate(unique_missions, start=1):
        print(f"{i}. {mission}")
    return filepaths_1, copy_rename


orig, copy=find_files(filepaths,type="LandscapeProducts", source="Drone",year="2023")

# correct names of subdirs inside the product folder
copy.loc[copy["product"] == "DEM", "product"] = "DSM"
copy.loc[copy["product"] == "projects", "product"] = "Project"
copy.loc[copy["product"] == "model", "product"] = "Model"

all_orig=[]
for i in range(len(orig)):
    new=os.path.join("\\",orig.iloc[i,0], orig.iloc[i,1], orig.iloc[i,2], orig.iloc[i,3], orig.iloc[i,4], orig.iloc[i,5], orig.iloc[i,6]).replace("\\","//")
    all_orig.append(new)

all_copy=[]
for i in range(len(copy)):
    new=os.path.join("\\",copy.iloc[i,0], copy.iloc[i,1], copy.iloc[i,2], copy.iloc[i,3], copy.iloc[i,4], copy.iloc[i,5], copy.iloc[i,6]).replace("\\","//")
    all_copy.append(new)

for i in range(len(all_orig)):
    if all_orig[i]!=all_copy[i]:
        print("not equal")
        os.rename(all_orig[i], all_copy[i])
    else:
        print("equal")

##get the files paths again
maindir= r'\\stri-sm01\ForestLandscapes'
def get_all_files(directory):
    all_files = []
    for foldername, subfolders, filenames in os.walk(directory):
        for filename in filenames:
            file_path = os.path.join(foldername, filename)
            all_files.append(file_path)
    return all_files
all_files = get_all_files(maindir)
table=[]
for file in all_files:
    file_split = file.split('\\')
    print(file_split)
    table.append(file_split)
df = pd.DataFrame(table)
df = df.drop([0, 1], axis=1)
df = df[df[4] != '.git']
new_columns = ['server', 'partition', 'type','source','year', 'mission', 'product', 'file', 'column9', 'column10', 'column11', 'column12']
df.columns = new_columns
path=r'\\stri-sm01\ForestLandscapes\filepaths.csv'
df.to_csv(path, index=False)


# Correct file names inside the files subdir

filepaths = pd.read_csv(os.path.join(server_dir, "filepaths.csv"))
orig, copy=find_files(filepaths,type="LandscapeProducts", source="Drone",year="2023")

copy[copy["product"]=="Cloudpoint"]
mask = (copy["product"] == "Cloudpoint") & copy["file"].str.contains("medium")
copy.loc[mask, "file"] = copy.loc[mask, "file"].str.replace("medium", "cloud")

copy[copy["product"]=="Orthophoto"]
mask = (copy["product"] == "Orthophoto") & copy["file"].str.contains("medium")
copy.loc[mask, "file"] = copy.loc[mask, "file"].str.replace("medium", "orthomosaic")

copy[copy["product"]=="DSM"]
mask = (copy["product"] == "DSM") & copy["file"].str.contains("medium")
copy.loc[mask, "file"] = copy.loc[mask, "file"].str.replace("medium", "dsm")

all_orig=[]
for i in range(len(orig)):
    new=os.path.join("\\",orig.iloc[i,0], orig.iloc[i,1], orig.iloc[i,2], orig.iloc[i,3], orig.iloc[i,4], orig.iloc[i,5], orig.iloc[i,6],orig.iloc[i,7]).replace("\\","//")
    all_orig.append(new)

all_copy=[]
for i in range(len(copy)):
    new=os.path.join("\\",copy.iloc[i,0], copy.iloc[i,1], copy.iloc[i,2], copy.iloc[i,3], copy.iloc[i,4], copy.iloc[i,5], copy.iloc[i,6],copy.iloc[i,7]).replace("\\","//")
    all_copy.append(new)

for i in range(len(all_orig)):
    if all_orig[i]!=all_copy[i]:
        print("not equal")
        os.rename(all_orig[i], all_copy[i])
    else:
        print("equal")



#ensure raw mission folders are named correctly
filepaths = pd.read_csv(os.path.join(server_dir, "filepaths.csv"))
orig, copy=find_files(filepaths,type="LandscapeRaw", source="Drone",year="2023")

def rename_mission(row):
    if row['mission'].startswith(row['year']):
        parts = row['mission'].split("_")
        year = parts[0]
        month = parts[1]
        day = parts[2]
        site = parts[3]
        plot = parts[4]
        drone = parts[5]
        newname = f"{site}_{plot}_{year}_{month}_{day}_{drone}"
        return newname
    else:
        return row['mission']

copy['mission'] = copy.apply(rename_mission, axis=1)

all_orig=[]
for i in range(len(orig)):
    new=os.path.join("\\",orig.iloc[i,0], orig.iloc[i,1], orig.iloc[i,2], orig.iloc[i,3], orig.iloc[i,4], orig.iloc[i,5]).replace("\\","//")
    all_orig.append(new)

all_copy=[]
for i in range(len(copy)):
    new=os.path.join("\\",copy.iloc[i,0], copy.iloc[i,1], copy.iloc[i,2], copy.iloc[i,3], copy.iloc[i,4], copy.iloc[i,5]).replace("\\","//")
    all_copy.append(new)


for i in range(len(all_orig)):
    if all_orig[i]!=all_copy[i]:
        print("not equal")
        os.rename(all_orig[i], all_copy[i])
    else:
        print("equal")

