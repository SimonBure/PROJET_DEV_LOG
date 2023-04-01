# -*- coding: utf-8 -*-
import os
import platform
import shutil
from numpy import save, load
from torch import Tensor
from importlib import resources


def save_tensor_to_disk_numpy(tensor: Tensor, path_to_save: str):
    numpy_version = tensor.numpy()  # Convert tensor to numpy array
    save(path_to_save, numpy_version)  # Save numpy array to .npy format


def load_tensor(path_to_load: str) -> Tensor:
    np_arr = load(path_to_load)  # Load numpy array from .npy binary file
    return Tensor(np_arr)  # Convert to a tensor


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


def get_path(env_path, where):
    """
    As pathway are different in each computer, compute actual pathway to store data in
    a known path

    Returns
    -------
    path : str
        Path of the dataset download.

    """
    path = ''
    if where == "Env":
        path = os.path.join(env_path, "env")
    elif where == "Database":
        path = os.path.join(env_path, "env", "Database")
    elif where == "Db_poject":
        path = os.path.join(env_path, "env", "Database", "project.db")
    elif where == "Img":
        path = os.path.join(env_path, "env", "Database", "img_dataset","celeba" ,"img_align_celeba")
    elif where == "Img_base":
        path = os.path.join(env_path, "env", "Database", "img_dataset")
    elif where == "Interface":
        path = os.path.join(env_path, "env", "Interface")
    elif where == "Encoder":
        path = os.path.join(env_path, "env", "Auto_encoder")
    elif where == "Result":
        path = os.path.join(env_path, "env", "Result")
    elif where == "Temp":
        path = os.path.join(env_path, "temp")
    elif where == "Auto_encoder":
        path = os.path.join(env_path, "env", "Auto_encoder")
    elif where == "gen_img":
        path = os.path.join(env_path, "env", "Auto_encoder", "gen_img")
    return path  # Collect the path


def create_folders(env_path, dev = False):
    """
    Create folders needed for the program
    """

    path = get_path(env_path, "Env")

    """
    Database Folder, will contain :
        database
        img_dataset folder
            img + attribute selection
    """
    if dev == True :
        create_path = os.path.join(path, "Database", "img_dataset", "celeba")
        os.makedirs(create_path)
    else :
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
    
    """
    Data Folder, will contain :
        idkit.db
        idkit.png
        new_dataset.zip
    """
    create_path = os.path.join(path, "Data")
    os.makedirs(create_path)


def remove_env_prog(env_path):
    shutil.rmtree(get_path(env_path, "Env"))
    
