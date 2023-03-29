# -*- coding: utf-8 -*-
import src.database as db
import src.main_f as main
import src.autoencoder as auto
import utils
import shutil
import sys
import os
import wget
import zipfile
import logging

#Setup log file
logging.basicConfig(filename='Idkit.log', encoding='utf-8', level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s : %(message)s')


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
    
    # idkit.db database du programme
    path = utils.get_path(env_path, "Database")
    add_path = os.path.join(path, "idkit.db")
    url = "https://filesender.renater.fr/download.php?token=e08cc673-d83a-45c8-a7ed-cd924c3f92e5&files_ids=23283443"
    wget.download(url, add_path)
    
    # idkit.png pour le logo du programme
    path = utils.get_path(env_path, "Interface")
    url2 = "https://filesender.renater.fr/download.php?token=e08cc673-d83a-45c8-a7ed-cd924c3f92e5&files_ids=23283444"
    add_path = os.path.join(path, "idkit.png")
    wget.download(url2, out=add_path)
    
    # new_dataset.zip pour les images de la base de données
    path = utils.get_path(path, "Img_base")
    add_path = os.path.join(path, "new_dataset.zip")
    url3 = "https://filesender.renater.fr/download.php?token=e08cc673-d83a-45c8-a7ed-cd924c3f92e5&files_ids=23283445"
    wget.download(url3, out=add_path)
    with zipfile.ZipFile(add_path, 'r') as zip_ref:
        zip_ref.extractall(path)
    
    
    # Temp as no option for downloading dataset exist
    path = os.path.join(env_path, "temp", "list_attr_celeba.txt")
    dst = utils.get_path(env_path, "Img_base")
    dst = os.path.join(dst, "celeba", "list_attr_celeba.txt")
    shutil.copy(path, dst)
    path = os.path.join(env_path, "temp", "img_align_celeba.zip")
    dst = utils.get_path(env_path, "Img_base")
    dst = os.path.join(dst, "celeba")
    with zipfile.ZipFile(path, 'r') as zip_ref:
        zip_ref.extractall(dst)

    path = os.path.join(env_path, "temp", "idkit.png")
    dst = utils.get_path(env_path, "Interface")
    dst = os.path.join(dst, "idkit.png")
    shutil.copy(path, dst)

    test = input("Créer database ? (Y/N)")
    if test == "Y" or test == "y":
        db.create_database(env_path, "Project")
        logging.info('Creating database')

        img = db.request_data_by_id(env_path, 1)
        print(img)
    
    test = input("Lancer Autoencodeur ? (Y/N)")
    if test == "Y" or test == "y":
        auto.launch_encoder(env_path)

test = input("Lancer programme ? (Y/N)")

if test == "Y" or test == "y":
    logging.info('Program launched')
    main.f1(env_path)


test2 = input("Détruire environnemnt ? (Y/N)")

if test2 == "Y" or test == "y":
    logging.info('Destructing environement')
    utils.remove_env_prog(env_path)
