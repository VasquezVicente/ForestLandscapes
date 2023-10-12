import os
import pandas as pd

maindir= r'\\stri-sm01\ForestLandscapes'

def get_all_files(directory):
    all_files = []
    for foldername, subfolders, filenames in os.walk(directory):
        for filename in filenames:
            file_path = os.path.join(foldername, filename)
            all_files.append(file_path)
    return all_files

# Get all files in the specified directory
all_files = get_all_files(maindir)

table=[]
for file in all_files:
    file_split = file.split('\\')
    print(file_split)
    table.append(file_split)
    
#make table a df
df = pd.DataFrame(table)
df = df.drop([0, 1], axis=1)

#filter df by column 4 == .git
df = df[df[4] != '.git']
#change the column names , column name to server 
new_columns = ['server', 'partition', 'type','source','year', 'mission', 'product', 'file', 'column9', 'column10', 'column11', 'column12']
df.columns = new_columns
path=r'\\stri-sm01\ForestLandscapes\filepaths.csv'
df.to_csv(path, index=False)
out= input("File paths generated in the ForestLandscape NAS server. Press Enter to exit")

