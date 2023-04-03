# -*- coding: utf-8 -*-
import IdKit.utils
import IdKit.interface
import os
import sys
import wget
import zipfile
import logging

#Setup log file
logging.basicConfig(filename='Idkit.log', level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s : %(message)s')
logging.getLogger('PIL').setLevel(logging.WARNING) # Thanks PIL for destroying log files

env_path = os.path.dirname(os.path.realpath(__file__))

if len(sys.argv) == 2 :
    if sys.argv[1] == "uninstall" :
        logging.info('Uninstalling environement')
        IdKit.utils.remove_env_prog(env_path)
        print("- Successfully uninstall idkit")
        logging.info(' - Done')
        sys.exit()

# Generate environement of the program

if not os.path.exists(os.path.join(env_path, "Idkit")):
    
    print("- Generating environement")
    logging.info('Creating environement')
    
    IdKit.utils.create_folders(env_path)
    
    # idkit.db database of program
    print(" - Downloading database")
    path = IdKit.utils.get_path(env_path, "Database")
    add_path = os.path.join(path, "idkit.db")
    url = "https://filesender.renater.fr/download.php?token=e08cc673-d83a-45c8-a7ed-cd924c3f92e5&files_ids=23283443"
    logging.info('Download from %s' %(url))
    wget.download(url, add_path)
    
    # idkit.png for program logo 
    path2 = IdKit.utils.get_path(env_path, "Interface")
    url2 = "https://filesender.renater.fr/download.php?token=e08cc673-d83a-45c8-a7ed-cd924c3f92e5&files_ids=23283444"
    add_path2 = os.path.join(path2, "idkit.png")
    logging.info('Download from %s' %(url2))
    wget.download(url2, out=add_path2)
    
    # new_dataset.zip for images of database
    print(" - Downloading image dataset")
    path3 = IdKit.utils.get_path(env_path, "Img_base")
    add_path3 = os.path.join(path3, "new_dataset.zip")
    url3 = "https://filesender.renater.fr/download.php?token=e08cc673-d83a-45c8-a7ed-cd924c3f92e5&files_ids=23283445"
    logging.info('Download from %s' %(url3))
    wget.download(url3, out=add_path3)
    logging.info('Exctract new_dataset.zip')
    with zipfile.ZipFile(add_path3, 'r') as zip_ref:
        zip_ref.extractall(path3)
    
    
    # model.pt for trained auto encodeur
    print(" - Downloading trained model")
    path4 = IdKit.utils.get_path(env_path, "Encoder")
    add_path4 = os.path.join(path4, "model.pt")
    url4 = "https://filesender.renater.fr/download.php?token=376da0cb-3715-41be-9851-fb27af1aba89&files_ids=23539804"
    wget.download(url4, out=add_path4)
    logging.info(' - Done')   
    
    print("- Environenement generated in %s" %(os.path.join(env_path, "Idkit")))

path = IdKit.utils.get_path(env_path, "Database")
add_path = os.path.join(path, "idkit.db")
path2 = IdKit.utils.get_path(env_path, "Interface")
add_path2 = os.path.join(path2, "idkit.png")
path3 = IdKit.utils.get_path(env_path, "Img_base")
add_path3 = os.path.join(path3, "new_dataset.zip")
path4 = IdKit.utils.get_path(env_path, "Encoder")
add_path4 = os.path.join(path4, "model.pt")

if not os.path.exists(add_path) or not os.path.exists(add_path2) or not os.path.exists(add_path3) or not os.path.exists(add_path4):
    logging.error('Files missings')
    raise(Exception("Files missings, please reinstall the program"))

print("- Launching program")
logging.info('Launching program')
IdKit.interface.f1(env_path)




