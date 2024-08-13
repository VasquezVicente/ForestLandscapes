# script to copy every ith photo in a folder to a new folder
# to be executed from within a folder
import os
import sys
import shutil

pathname= input("Enter the path of the folder: ")
pathlist = os.listdir(pathname)  # Make a list of what is in that path

pathlist = [elem for elem in pathlist if elem.lower().endswith('.jpg')]  # List comprehension filters

print("This script will copy the ith image within")
print("the cwd into a new folder\n")
print("CWD: " + pathname + "\n")

# Input handling
try:
    ith_image = int(input("Integer of every ith photo to copy: "))
except ValueError:
    print("Please enter a valid integer.")
    sys.exit(1)

print("")

# Determine the suffix for the folder name
if ith_image >= 4:
    suffix = "th"
elif ith_image == 3:
    suffix = "rd"
elif ith_image == 2:
    suffix = "nd"
else:
    suffix = "st"

output_folder = f"{pathname.split(os.sep)[-1]}_{ith_image}{suffix}_image"
output_path = os.path.join(pathname, output_folder)

# Create the output directory
try:
    os.mkdir(output_path)
except FileExistsError:
   print("not created, already exists")
# Move every ith image from the cwd to the new folder
for index, file in enumerate(pathlist):
    if (index + 1) % ith_image == 0:
        src = os.path.join(pathname, file)
        dst = os.path.join(output_path, file)
        shutil.move(src, dst)
        print(f"Moved {file} to {output_folder}")