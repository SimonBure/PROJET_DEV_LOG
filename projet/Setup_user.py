# -*- coding: utf-8 -*-
import IdKit.utils
import IdKit.database
import IdKit.autoencoder
import IdKit.main_f
import shutil
import sys
import os
import wget
import zipfile

env_path = os.path.dirname(os.path.realpath(__file__))

test = input("Nécessité de télécharger les fichiers du programme ? (Y/N)")
if test == "Y":
    create_path = os.path.join(env_path, "temp")
    os.makedirs(create_path)
    add_path = os.path.join(env_path, "temp", "list_attr_celeba.txt")
    url = "https://filesender.renater.fr/download.php?token=80050e2e-f52b-44ed-8bad-ff4d77649cb3&files_ids=22772324"
    wget.download(url, add_path)
    url2 = "https://filesender.renater.fr/download.php?token=80050e2e-f52b-44ed-8bad-ff4d77649cb3&files_ids=22772325"
    add_path = os.path.join(env_path, "temp", "img_align_celeba.zip")
    wget.download(url2, out=add_path)
    url3 = "https://filesender.renater.fr/download.php?token=c11040a4-e2dd-4198-baa7-f40bf9ae7d96&files_ids=22873415"
    add_path = os.path.join(env_path, "temp", "idkit.png")
    wget.download(url3, out=add_path)


# Generate environement of the program

test = input("Créer l'environnement du programme ? (Y/N)")

if test == "Y":
    IdKit.utils.create_folders(env_path)

    # Temp as no option for downloading dataset exist
    path = os.path.join(env_path, "temp", "list_attr_celeba.txt")
    dst = IdKit.utils.get_path(env_path, "Img")
    dst = os.path.join(dst, "celeba", "list_attr_celeba.txt")
    shutil.copy(path, dst)
    path = os.path.join(env_path, "temp", "img_align_celeba.zip")
    dst = IdKit.utils.get_path(env_path, "Img")
    dst = os.path.join(dst, "celeba")
    with zipfile.ZipFile(path, 'r') as zip_ref:
        zip_ref.extractall(dst)

    path = os.path.join(env_path, "temp", "idkit.png")
    dst = IdKit.utils.get_path(env_path, "Interface")
    dst = os.path.join(dst, "idkit.png")
    shutil.copy(path, dst)
        
    IdKit.create_db.create_database(env_path)

    IdKit.autoencoder.launch_encoder(env_path)

test3 = input("Lancer programme ? (Y/N)")

if test3 == "Y" or test3 == "y":
    IdKit.main_f.f1(env_path)


test2 = input("Détruire environnemnt ? (Y/N)")

if test2 == "Y" or test2 == "y":
    IdKit.utils.remove_env_prog(env_path)
