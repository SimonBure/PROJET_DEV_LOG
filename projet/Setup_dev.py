# -*- coding: utf-8 -*-
import src_deploy.interface as main
import utils
import shutil
import sys
import os
import wget
import zipfile
import logging

# Setup log file
logging.basicConfig(filename='Idkit.log', level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s : %(message)s')
logging.getLogger('PIL').setLevel(logging.WARNING) # Merci PIL ...


env_path = os.path.dirname(os.path.realpath(__file__))

test = input("First time launching setup ? (Y/N)")
if test == "Y" or test == "y":
    sys.path.append(env_path)
    add_path = os.path.join(env_path, "src")
    sys.path.append(add_path)


# Generate environement of the program

test = input("Créer environnement ? (Y/N)")


if test == "Y" or test == "y":
    logging.info('Creating environement')
    utils.create_folders(env_path, dev = True)
    
    # idkit.db database of program
    path = utils.get_path(env_path, "Database")
    add_path = os.path.join(path, "idkit.db")
    url = "https://filesender.renater.fr/download.php?token=e08cc673-d83a-45c8-a7ed-cd924c3f92e5&files_ids=23283443"
    wget.download(url, add_path)
    
    # idkit.png for program logo
    path = utils.get_path(env_path, "Interface")
    url2 = "https://filesender.renater.fr/download.php?token=e08cc673-d83a-45c8-a7ed-cd924c3f92e5&files_ids=23283444"
    add_path = os.path.join(path, "idkit.png")
    wget.download(url2, out=add_path)
    
    # new_dataset.zip for images of database
    path = utils.get_path(env_path, "Img_base")
    add_path = os.path.join(path, "new_dataset.zip")
    url3 = "https://filesender.renater.fr/download.php?token=e08cc673-d83a-45c8-a7ed-cd924c3f92e5&files_ids=23283445"
    wget.download(url3, out=add_path)
    with zipfile.ZipFile(add_path, 'r') as zip_ref:
        zip_ref.extractall(path)
    
    # model.pt for trained auto encodeur
    path = utils.get_path(env_path, "Encoder")
    add_path = os.path.join(path, "model.pt")
    url4 = "https://filesender.renater.fr/download.php?token=376da0cb-3715-41be-9851-fb27af1aba89&files_ids=23539804"
    wget.download(url4, out=add_path)
    
test = input("Lancer programme ? (Y/N)")

if test == "Y" or test == "y":
    logging.info('Program launched')
    main.f1(env_path)


test2 = input("Détruire environnemnt ? (Y/N)")

if test2 == "Y" or test2 == "y":
    logging.info('Destructing environement')
    utils.remove_env_prog(env_path)
