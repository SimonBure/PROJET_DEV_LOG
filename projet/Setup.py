# -*- coding: utf-8 -*-
import src.create_db as db
import utils
import shutil
import sys
import os


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
    path = os.path.join(path, "temp", "img_align_celeba")
    dst = utils.get_path("Img")
    dst = os.path.join(dst, "celeba", "img_align_celeba")
    shutil.copytree(path, dst)

    test = input("Créer database ? (Y/N)")
    if test == "Y":
        db.create_database()
        
        img = db.request_data_by_id(1)
        print(img)


test2 = input("Détruire environnemnt ? (Y/N)")

if test2 == "Y":
    utils.remove_env_prog()
