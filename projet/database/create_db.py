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

def metadata_pull(path) :
    """
    Get the metadata's name for creating the table in the database

    Take
    -------
    path : str
        Path of the downloaded dataset.
    
    Returns
    -------
    metadata : list
        List of the header's name of the metadata.
        
    """
    path_meta = path + "/celeba/list_attr_celeba.txt"
    with open(path_meta, "r") as file:
        print(file.readline())
        metadata = file.readline()

    metadata = metadata.split(" ")
    return metadata

def create_meta_table(cursor, path_of_metadata) :
    """
    Create the table of metadata in the database

    Take
    -------
    cursor : database.cursor
        Cursor for the database
    path : str
        Path of the downloaded dataset.
    
    """
    # Retrieve metadata :
    metadata = metadata_pull(path_of_metadata)

    # Start the create table line :
    table_str = "[id] INTEGER PRIMARY KEY, "
    for el in metadata :
        table_str += " [%s] TEXT, " %(el)
    table_str = table_str[:-2] # We retrieve the last 2 as there is no more data to append

    com_line = "CREATE TABLE IF NOT EXISTS portrait (%s)" %(table_str)
    cursor.execute(comm)

def create_database(path_of_metadata) :
    """
    Create the database needed for the project. Insert dataset CelebA from : <insert url ?>

    Take
    -------
    path : str
        Path of the downloaded dataset.
    
    Returns
    -------
    cursor : database.cursor
        Cursor for communicating with the database
    """
    sqlite3.connect('project_db')

    con = sqlite3.connect('project_db')
    cursor = con.cursor()

    create_meta_table(cursor, path_of_metadata)

    return cursor


path = get_dataset_path()

first_data = torchvision.datasets.CelebA(root = path, transform = transforms.PILToTensor(), download = True)

path_atr = path + "/celeba/list_attr_celeba.txt"

# Uniquement le fichier attribut :
data = np.loadtxt(path_atr, dtype = str, skiprows = 2)

data = data[0:10]

db_cursor = create_database(path_of_metadata)

