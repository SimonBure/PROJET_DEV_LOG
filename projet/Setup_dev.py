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

env_path = os.path.dirname(os.path.realpath(__file__))


test = input("Need to download dataset ? (Y/N)")
if test == "Y" or test == "y":
    add_path = os.path.join(env_path, "temp", "list_attr_celeba.txt")
    url = "https://filesender.renater.fr/download.php?token=80050e2e-f52b-44ed-8bad-ff4d77649cb3&files_ids=22772324"
    wget.download(url, add_path)
    url2 = "https://filesender.renater.fr/download.php?token=80050e2e-f52b-44ed-8bad-ff4d77649cb3&files_ids=22772325"
    add_path = os.path.join(env_path, "temp", "img_align_celeba.zip")
    wget.download(url2, out=add_path)


test = input("First time launching setup ? (Y/N)")
if test == "Y" or test == "y":
    sys.path.append(env_path)
    add_path = os.path.join(env_path, "src")
    sys.path.append(add_path)


# Generate environement of the program

test = input("Créer environnement ? (Y/N)")

if test == "Y" or test == "y":
    utils.create_folders(env_path)

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
        db.create_database(env_path)

        img = db.request_data_by_id(env_path, 1)
        print(img)
    
    test = input("Lancer Autoencodeur ? (Y/N)")
    if test == "Y" or test == "y":
        auto.launch_encoder(env_path)

test = input("Lancer programme ? (Y/N)")

if test == "Y" or test == "y":
    main.f1(env_path)


test2 = input("Détruire environnemnt ? (Y/N)")

if test2 == "Y" or test == "y":
    utils.remove_env_prog(env_path)
