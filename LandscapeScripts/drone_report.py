import os
import pandas as pd

while True:
    year_to_check = input("Enter year to check (type 'exit' to quit): ")

    if year_to_check.lower() == 'exit':
        print("Exiting the program...")
        break

    server_dir = r"\\stri-sm01\ForestLandscapes\LandscapeRaw\Drone"
    server_dir=os.path.join(server_dir, year_to_check)
    missions= os.listdir(server_dir)

    #check if missions in raw folder are in the product folder
    not_allright_flag_raw_product = False #flag to check if all missions are in product folder

    server_dir_product = r"\\stri-sm01\ForestLandscapes\LandscapeProducts\Drone"
    server_dir_product=os.path.join(server_dir_product, year_to_check)
    missions_product= os.listdir(server_dir_product)

    for mission in missions:
        if mission not in missions_product:
            print(f"{mission} is not in product folder")
            if not os.path.exists(os.path.join(server_dir_product,mission)):
                os.makedirs(os.path.join(server_dir_product,mission))  #if not there make it
            not_allright_flag_raw_product = True

    #print the flag
    if not_allright_flag_raw_product:
        print("Some missions are not in product folder.")
    else:
        print("All missions in the raw folder are in product folder.")

    #check file structure of raw missions
    not_allright_flag_raw_structure = False
    for mission in missions:
        subdir = os.listdir(os.path.join(server_dir, mission))
        if not (subdir == ['Images', 'Images_extra'] or subdir == ['Images', 'Images_extra', 'Videos']):
            print(subdir)
            print(f"{mission} raw structure is not alright")
            not_allright_flag_raw_structure = True
        #fixing case: if there is no Images_extra folder, create it
        if not os.path.exists(os.path.join(server_dir, mission, 'Images_extra')):
            os.mkdir(os.path.join(server_dir, mission, 'Images_extra'))

    # Check flag and return message
    if not_allright_flag_raw_structure:
        print("Some missions do not have a correct raw structure")
    else:
        print("All missions raw structure is correct.")

    #check Images folder for empty missions or missions with no images
    not_allright_flag_raw_empty = False
    for mission in missions:
        if len(os.listdir(os.path.join(server_dir, mission, 'Images'))) == 0:
            print(f"{mission} has no images")
            not_allright_flag_raw_empty = True

    if not_allright_flag_raw_empty:
        print("Some missions have no images.")
    else:
        print("All missions have images.")


    #check landscapeproducts
    server_dir= r'\\stri-sm01\ForestLandscapes\LandscapeProducts\Drone'
    server_dir=os.path.join(server_dir, year_to_check)
    missions= os.listdir(server_dir)

    #check file structure of raw missions
    not_allright_flag_product_structure = False
    for mission in missions:
        subdir = os.listdir(os.path.join(server_dir, mission))
        if not (subdir == ['Cloudpoint', 'DSM', 'Orthophoto', 'Project'] or subdir == ['Cloudpoint', 'DSM', 'Model', 'Orthophoto', 'Project']):
            print(f"{mission} products structure is not alright")
            not_allright_flag_product_structure= True
            if not os.path.exists(os.path.join(server_dir, mission, 'Orthophoto')):
                os.mkdir(os.path.join(server_dir, mission, 'Orthophoto'))
            if not os.path.exists(os.path.join(server_dir, mission, 'Project')):
                os.mkdir(os.path.join(server_dir, mission, 'Project'))
            if not os.path.exists(os.path.join(server_dir, mission, 'DSM')):
                os.mkdir(os.path.join(server_dir, mission, 'DSM'))
            if not os.path.exists(os.path.join(server_dir, mission, 'Cloudpoint')):
                os.mkdir(os.path.join(server_dir, mission, 'Cloudpoint'))

    #print flag
    if not_allright_flag_product_structure:
        print("Some missions product structure are not alright.")
    else:
        print("All missions product structure are alright.")

    #check for orthomosaics
    not_allright_flag_ortho= False
    for mission in missions:
        #orthoname= os.path.join('_'.join(mission.split('_')[0:5])+"_orthomosaic.tif")
        if 'P4P' in mission:
            orthoname= mission.replace('P4P','orthomosaic')+".tif"
        elif 'EBEE' in mission:
            orthoname= mission.replace('EBEE','orthomosaic')+".tif"
        elif 'INSPIRE' in mission:
            orthoname= mission.replace('INSPIRE','orthomosaic')+".tif"
        elif 'SOLO' in mission:
            orthoname= mission.replace('SOLO','orthomosaic')+".tif"
        if not os.path.exists(os.path.join(server_dir, mission, 'Orthophoto', orthoname)):
            print(f"{mission} has no orthomosaic")
            not_allright_flag_ortho = True

    #print flag
    if not_allright_flag_ortho:
        print("Some missions have no orthomosaic.")
    else:
        print("All missions have orthomosaics.")

    #check for DSMs
    not_allright_flag = False
    for mission in missions:
        if 'P4P' in mission:
            dsmname= mission.replace('P4P','dsm')+".tif"
        elif 'EBEE' in mission:
            dsmname= mission.replace('EBEE','dsm')+".tif"
        elif 'INSPIRE' in mission:
            dsmname= mission.replace('INSPIRE','dsm')+".tif"
        elif 'SOLO' in mission:
            dsmname= mission.replace('SOLO','dsm')+".tif"
        if not os.path.exists(os.path.join(server_dir, mission, 'DSM', dsmname)):
            print(f"{mission} has no DSM")
            not_allright_flag = True

    #print flag
    if not_allright_flag:
        print("Some missions have no DSM.")
    else:
        print("All missions have DSMs.")

    #check for point clouds
    not_allright_flag_report = False
    for mission in missions:
        if 'P4P' in mission:
            cloudname= mission.replace('P4P','cloud')+".las"
        elif 'EBEE' in mission:
            cloudname= mission.replace('EBEE','cloud')+".las"
        elif 'INSPIRE' in mission:
            cloudname= mission.replace('INSPIRE','cloud')+".las"
        elif 'SOLO' in mission:
            cloudname= mission.replace('SOLO','cloud')+".las"
        if not os.path.exists(os.path.join(server_dir, mission, 'Cloudpoint', cloudname)):
            print(f"{mission} has no point cloud")
            not_allright_flag_report = True

    #print flag
    if not_allright_flag_report:
        print("Some missions have no point cloud.")
    else:
        print("All missions have point clouds.")

    #check for project report
    not_allright_flag_report = False
    for mission in missions:
        if 'P4P' in mission:
            reportname= mission.replace('P4P','report')+".pdf"
        elif 'EBEE' in mission:
            reportname= mission.replace('EBEE','report')+".pdf"
        elif 'INSPIRE' in mission:
            reportname= mission.replace('INSPIRE','report')+".pdf"
        elif 'SOLO' in mission:
            reportname= mission.replace('SOLO','report')+".pdf"
        if not os.path.exists(os.path.join(server_dir, mission, 'Project', reportname)):
            print(f"{mission} has no project report")
            not_allright_flag_report = True
            orig= os.path.join(server_dir, mission, 'Project', reportname.replace('_report.pdf', '_medium.pdf'))
            if os.path.exists(orig):
                os.rename(orig, os.path.join(server_dir, mission, 'Project', reportname))
            else:
                continue
            

    #print flag
    if not_allright_flag_report:
        print("Some missions have no project report.")
    else:
        print("All missions have project reports.")


    #if all flags false then all checks passed
    if not_allright_flag_raw_structure or not_allright_flag_raw_empty or not_allright_flag_product_structure or not_allright_flag_ortho or not_allright_flag or not_allright_flag_report:
        print("The folder did not pass all checks.")
    else:
        print("The folder passed all checks.Celebrate!")


    press = input("Press 'exit' to quit or press Enter to check another year")
    if press.lower() == 'exit':
        print("Exiting the program...")
        break  # stops the loop