import sqlite3
import numpy as np
import os

import torchvision.datasets
import torchvision.transforms as transforms


def get_dataset_path() :
    """
    As pathway are different in each computer, compute actual pathway to store data in
    a known path
    
    Returns
    -------
    path : str
        Path of the dataset donload.
        
    """
    path = os.getcwd()
    path += "\img_dataset"
    return path


path = get_dataset_path()

first_data = torchvision.datasets.CelebA(root = path, transform = transforms.PILToTensor(), download = True)

path_atr = path + "/celeba/list_attr_celeba.txt"

# Uniquement le fichier attribut :
data = np.loadtxt(path_atr, dtype = str, skiprows = 2)
