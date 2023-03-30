# -*- coding: utf-8 -*-
import IdKit.utils
import IdKit.database
#import IdKit.autoencoder
import IdKit.main_f
import os
import sys
import wget
import zipfile

env_path = os.path.dirname(os.path.realpath(__file__))

if len(sys.argv) == 2 :
    if sys.argv[1] == "uninstall" :
        IdKit.utils.remove_env_prog(env_path)
        print("- Successfully uninstall idkit")
        sys.exit()

# Generate environement of the program

if not os.path.exists(os.path.join(env_path, "Idkit")):
    
    print("- Generating environement")
    
    IdKit.utils.create_folders(env_path)
    
    # idkit.db database du programme
    print(" - Downloading database")    
    path = IdKit.utils.get_path(env_path, "Database")
    add_path = os.path.join(path, "idkit.db")
    url = "https://filesender.renater.fr/download.php?token=e08cc673-d83a-45c8-a7ed-cd924c3f92e5&files_ids=23283443"
    wget.download(url, add_path)
    
    # idkit.png pour le logo du programme 
    path = IdKit.utils.get_path(env_path, "Interface")
    url2 = "https://filesender.renater.fr/download.php?token=e08cc673-d83a-45c8-a7ed-cd924c3f92e5&files_ids=23283444"
    add_path = os.path.join(path, "idkit.png")
    wget.download(url2, out=add_path)
    
    # new_dataset.zip pour les images de la base de données
    print(" - Downloading image dataset")
    path = IdKit.utils.get_path(env_path, "Img_base")
    add_path = os.path.join(path, "new_dataset.zip")
    url3 = "https://filesender.renater.fr/download.php?token=e08cc673-d83a-45c8-a7ed-cd924c3f92e5&files_ids=23283445"
    wget.download(url3, out=add_path)
    with zipfile.ZipFile(add_path, 'r') as zip_ref:
        zip_ref.extractall(path)
        

    #IdKit.autoencoder.launch_encoder(env_path)
    
    print("- Environenement generated in %s" %(os.path.join(env_path, "Idkit")))


print("- Launching program")
IdKit.main_f.f1(env_path)




