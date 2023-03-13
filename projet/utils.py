# -*- coding: utf-8 -*-
import os
import platform
import shutil


def get_sub_sys():
    """
    As pathway are different in each computer, search for sub-sytem type

    Returns
    -------
    sys[0] : str
        Name of sub-system.

    """
    sys = platform.uname()  # Collect system data
    return sys[0]


def get_path(where):
    """
    As pathway are different in each computer, compute actual pathway to store data in
    a known path

    Returns
    -------
    path : str
        Path of the dataset download.

    """
    path = os.path.dirname(os.path.realpath(__file__))
    if where == "Env":
        path = os.path.join(path, "env")
    elif where == "Database":
        path = os.path.join(path, "env", "Database")
    elif where == "Img":
        path = os.path.join(path, "env", "Database", "img_dataset")
    elif where == "Interface":
        path = os.path.join(path, "env", "Inteface")
    elif where == "Encoder":
        path = os.path.join(path, "env", "Auto_encoder")
    elif where == "Result":
        path = os.path.join(path, "env", "Result")

    return path  # Collect the path


def create_folders():
    """
    Create folders needed for the program
    """

    path = get_path("Env")

    """
    Database Folder, will contain :
        database
        img_dataset folder
            img + attribute selection
    """
    create_path = os.path.join(path, "Database", "img_dataset", "celeba")
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


def remove_env_prog():
    shutil.rmtree(get_path("Env"))
