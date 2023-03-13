# -*- coding: utf-8 -*-
import src.create_db as db
import utils
import shutil
import sys
import os
import wget
import zipfile


test = input("Need to download file ? (Y/N)")
if test == "Y":
    add_path = utils.get_path("Other")
    add_path = os.path.join(add_path, "temp", "list_attr_celeba.txt")
    url = "https://filesender.renater.fr/download.php?token=80050e2e-f52b-44ed-8bad-ff4d77649cb3&files_ids=22772324"
    wget.download(url, add_path)
    url2 = "https://filesender.renater.fr/download.php?token=80050e2e-f52b-44ed-8bad-ff4d77649cb3&files_ids=22772325"
    add_path = utils.get_path("Other")
    add_path = os.path.join(add_path, "temp", "img_align_celeba.zip")
    wget.download(url2, out=add_path)


test = input("First time launching setup ? (Y/N)")
if test == "Y":
    sys.path.append(utils.get_path("Other"))
    add_path = utils.get_path("Other")
    add_path = os.path.join(add_path, "src")
    sys.path.append(add_path)


# Generate environement of the program

test = input("Créer environnement ? (Y/N)")

if test == "Y":
    utils.create_folders()

    # Temp as no option for downloading dataset exist
    path = utils.get_path("Other")
    path = os.path.join(path, "temp", "list_attr_celeba.txt")
    dst = utils.get_path("Img")
    dst = os.path.join(dst, "celeba", "list_attr_celeba.txt")
    shutil.copy(path, dst)

    path = utils.get_path("Other")
    path = os.path.join(path, "temp", "img_align_celeba.zip")
    dst = utils.get_path("Img")
    dst = os.path.join(dst, "celeba")
    with zipfile.ZipFile(path, 'r') as zip_ref:
        zip_ref.extractall(dst)

    test = input("Créer database ? (Y/N)")
    if test == "Y":
        db.create_database()

        img = db.request_data_by_id(1)
        print(img)


test2 = input("Détruire environnemnt ? (Y/N)")

if test2 == "Y":
    utils.remove_env_prog()
