# -*- coding: utf-8 -*-
import IdKit.utils
import IdKit.database
#import IdKit.autoencoder
import IdKit.main_f
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
    
    # idkit.db database du programme
    print(" - Downloading database")
    path = IdKit.utils.get_path(env_path, "Database")
    add_path = os.path.join(path, "idkit.db")
    url = "https://filesender.renater.fr/download.php?token=e08cc673-d83a-45c8-a7ed-cd924c3f92e5&files_ids=23283443"
    logging.info('Download from %s' %(url))
    wget.download(url, add_path)
    
    # idkit.png pour le logo du programme 
    path = IdKit.utils.get_path(env_path, "Interface")
    url2 = "https://filesender.renater.fr/download.php?token=e08cc673-d83a-45c8-a7ed-cd924c3f92e5&files_ids=23283444"
    add_path = os.path.join(path, "idkit.png")
    logging.info('Download from %s' %(url2))
    wget.download(url2, out=add_path)
    
    # new_dataset.zip pour les images de la base de données
    print(" - Downloading image dataset")
    path = IdKit.utils.get_path(env_path, "Img_base")
    add_path = os.path.join(path, "new_dataset.zip")
    url3 = "https://filesender.renater.fr/download.php?token=e08cc673-d83a-45c8-a7ed-cd924c3f92e5&files_ids=23283445"
    logging.info('Download from %s' %(url3))
    wget.download(url3, out=add_path)
    logging.info('Exctract new_dataset.zip')
    with zipfile.ZipFile(add_path, 'r') as zip_ref:
        zip_ref.extractall(path)
    logging.info(' - Done')
        

    #IdKit.autoencoder.launch_encoder(env_path)
    
    print("- Environenement generated in %s" %(os.path.join(env_path, "Idkit")))


print("- Launching program")
logging.info('Launching program')
IdKit.main_f.f1(env_path)




