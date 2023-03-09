# -*- coding: utf-8 -*-
import os
import shutil
import utils

# Generate environement of the program


def create_folders(path):
    """
    Create folders needed for the program
    """

    """
    Database Folder, will contain :
        database
        img_dataset folder
            img + attribute selection
    """
    create_path = os.path.join(path, "Database", "img_dataset")
    os.makedirs(create_path)

    """
    Result Folder, will contain :
        Output of the program
    """
    create_path = os.path.join(path, "Result")
    os.makedirs(create_path)

    """
    Interface Folder, will contain :
        logo.png
    """
    create_path = os.path.join(path, "Interface")
    os.makedirs(create_path)

    """
    Auto-encoder Folder, will contain :
        gen_img
            .img
    """
    create_path = os.path.join(path, "Auto_encoder", "gen_img")
    os.makedirs(create_path)


create_folders(utils.get_path())


def remove_env_prog(path):

    shutil.rmtree(path)


# remove_env_prog(utils.get_path())
