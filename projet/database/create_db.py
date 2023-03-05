import sqlite3
import numpy as np
import os
import platform

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
    path = os.getcwd()  # Collect the path
    sys = platform.uname() # Collect system data
    if sys[0] == "Windows" :
        path += "\img_dataset" # Windows style path
    else :
        path += "/img_dataset" # Linux/Mac style path
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
    data : numpy array
        Array of metadata value for each image.
        
    """
    path_data = path + "/celeba/list_attr_celeba.txt"
    with open(path_data, "r") as file:
        print(file.readline())
        metadata = file.readline()

    metadata = metadata.split(" ")
    
    data = np.loadtxt(path_data, dtype = str, skiprows = 2)
    
    #Test choice
    data = data[0:10]
    
    return metadata, data

def create_meta_table(cursor, metadata) :
    """
    Create the table of metadata in the database

    Take
    -------
    cursor : database.cursor
        Cursor for the database
    path : str
        Path of the downloaded dataset.
    
    """
    # Start the create table line :
    table_str = "[id] INTEGER PRIMARY KEY, "
    for el in metadata :
        table_str += " [%s] TEXT, " %(el)
    table_str = table_str[:-2] # We retrieve the last 2 as there is no more data to append

    com_line = "CREATE TABLE IF NOT EXISTS portrait (%s)" %(table_str)
    cursor.execute(com_line)

def insert_data(cursor, dataset) :
    """
    Insert data in the table

    Take
    -------
    cursor : database.cursor
        Cursor for the database
    dataset : numpy.array
        Contain thevalue of metadatas of the dataset
    
    """
    # Add the index column to the dataset
    dataset = np.insert(dataset, 0, list(range(0,len(dataset))), axis = 1)
    
    # Transform Dataset into tuple to provide for the cursor
    list_data = []
    for i in range(len(dataset)) :
        list_data.append(tuple(dataset[i]))

    # Insert data in the table
    cursor.executemany("INSERT INTO portrait VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", list_data)
    print('We have inserted', cursor.rowcount, 'records to the table.') # Testing possibility ?

def create_database(path_of_metadata) :
    """
    Create the database needed for the project. Insert dataset CelebA from : <insert url ?>

    Take
    -------
    path : str
        Path of the downloaded dataset.
    dataset : numpy.array
        Contain the metadatas of the dataset
    
    Returns
    -------
    cursor : database.cursor
        Cursor for communicating with the database
    """
    sqlite3.connect('project_db')

    con = sqlite3.connect('project_db')
    cursor = con.cursor()
    
    # Retrieve datas :
    metadata, dataset = metadata_pull(path_of_metadata)

    create_meta_table(cursor, metadata)

    insert_data(cursor, dataset)

    return cursor


path = get_dataset_path()

#Download dataset
#first_data = torchvision.datasets.CelebA(root = path, transform = transforms.PILToTensor(), download = True)

db_cursor = create_database(path)

res = db_cursor.execute("SELECT * FROM portrait")
test = res.fetchall()
