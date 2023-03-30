# -*- coding: utf-8 -*-

import database
import utils
import os
import numpy as np
import shutil
import sqlite3

env_path = "../"

dataset_path = utils.get_path(env_path, "Img_base")
dataset_path = os.path.join(dataset_path, "new_dataset")

if not os.path.exists(dataset_path) :
    os.makedirs(dataset_path)
    
def get_X_img(env_path, array, x, who):
    """
    Return the path of 5 image. If array is given, try to have img at max
    considering the attributes, else choose randomly.

    Take
    -------
    env_path : str
        Path of the environement.
    array : 1D array
        metadata array of 0 and 1

    Returns
    -------
    path_img_list : str
        Absolute path of the 5 images.

    """
    path_img_list = database.request_data_by_metadata(env_path, array, who)
    if len(path_img_list) == 0 :
        return 0 # Do nothing
    elif len(path_img_list) > x:
        path_img_list_temp = []
        numbers = np.random.randint(1, len(path_img_list), size= x)
        for i in numbers:
            path_img_list_temp.append(path_img_list[i])
        path_img_list = path_img_list_temp

    return path_img_list

possibilities = ["-1", "1", "0"]

meta = ["0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "-1", "0", "0", "0", "0", "0", "0", "0", "0",
                   "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0"]

# All possibilities that exists for each attributes :
for i in [1,2] :
    for b in possibilities :
        if i == 1 :
            meta[4] = b
        else :
            meta[0] = b
        for c in possibilities : 
            if i == 1 :
                meta[5] = c
            else :
                meta[6] = c
            for d in possibilities : 
                if i == 1 :
                    meta[7] = d
                else :
                    meta[8] = d
                for e in possibilities :  
                    if i == 1 :
                        meta[9] = e
                    else :
                        meta[11] = e
                    for f in possibilities :
                        if i == 1 :
                            meta[13] = f
                        else :
                            meta[15] = f
                        for g in possibilities :
                            if i == 1 :
                                meta[16] = g
                            else :
                                meta[17] = g
                            for h in possibilities :
                                if i == 1 :
                                    meta[20] = h
                                else :
                                    meta[22] = h
                                for i in possibilities :
                                    if i == 1 :
                                        meta[24] = i
                                    else :
                                        meta[23] = i
                                    for j in possibilities :
                                        if i == 1 :
                                            meta[25] = j
                                        else :
                                            meta[28] = j
                                        for k in possibilities :
                                            if i == 1 :
                                                meta[27] = k
                                            else :
                                                meta[32] = k
                                            for l in possibilities :
                                                if i == 1 :
                                                    meta[33] = l
                                                else :
                                                    meta[30] = l
                                                for m in possibilities :
                                                    if i == 1 :
                                                        meta[35] = m
                                                    else :
                                                        meta[39] = m
                                    
                                                    path = get_X_img(env_path, meta, 4, "Project")
                                                    if type(path) == list :
                                                        for img in path :
                                                            shutil.copy(img, dataset_path)
                                                    elif type(path) == str :
                                                        shutil.copy(path, dataset_path)
    print("Moitié faite !")

def get_database_path(env_path):
    """
    Retrieve access to database to query her

    Take
    -------
    env_path : str
        Path of the environement.

    Returns
    -------
    path : str
        Path of the dataset download.

    """
    path = utils.get_path(env_path, "Database")
    data_loc = os.path.join(path, "autoencode.db")
    return data_loc

def metadata_pull(env_path):
    """
    Get the metadata's name for creating the table in the database

    Take
    -------
    env_path : str
        Path of the environement.

    Returns
    -------
    metadata : list
        List of the header's name of the metadata.
    data : numpy array
        Array of metadata value for each image.

    """
    path = utils.get_path(env_path, "Img_base")
    path_data = os.path.join(path, "celeba", "list_attr_celeba.txt")

    with open(path_data, "r") as file:
        file.readline()
        metadata_raw = file.readline()

    metadata = metadata_raw.split(" ")

    data = np.loadtxt(path_data, dtype=str, skiprows=2)
    
    real_data = []
    path_img = os.path.join(path, "new_dataset")
    img_name = os.listdir(path_img)
    for name in img_name :
        name = int(name[:-4])
        real_data.append(data[name-1])
    
    path_data = os.path.join(path, "list_attr_celeba.txt")
    with open(path_data, "w") as file:
        file.write(metadata_raw)
        for line in real_data :
            line_str = ""
            for i in range(len(line)) :
                line_str += str(line[i]) + " "
            file.write(line_str + "\n")
        

    # Split for testing
    #data = data[0:10]

    return metadata, real_data


def create_meta_table(cursor, metadata):
    """
    Create the table of metadata in the database

    Take
    -------
    cursor : database.cursor
        Cursor for the database
    metadata : list
        List of the header's name of the metadata.

    """
    # Start the create table line :
    table_str = "[id] INTEGER PRIMARY KEY, [name] TEXT,"
    for el in metadata[:-1]:  # Because last = empty, img name ?
        table_str += " [%s] TEXT, " % (el)
    # We retrieve the last 2 as there is no more data to append
    table_str = table_str[:-2]

    com_line = "CREATE TABLE IF NOT EXISTS portrait (%s);" % (table_str)
    cursor.execute(com_line)

def insert_data(cursor, connect, dataset):
    """
    Insert data in the table

    Take
    -------
    cursor : database.cursor
        Cursor for the database
    connect : database.connector
        Connector of the database
    dataset : numpy.array
        Contain thevalue of metadatas of the dataset

    """
    # Add the index column to the dataset
    dataset = np.insert(dataset, 0, list(range(0, len(dataset))), axis=1)

    # Transform Dataset into tuple to provide for the cursor
    list_data = []
    for i in range(len(dataset)):
        list_data.append(tuple(dataset[i]))

    # Insert data in the table
    cursor.executemany(
        "INSERT INTO portrait VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", list_data)
    print('We have inserted', cursor.rowcount,
          'records to the table.')  # Testing possibility ?
    connect.commit()
    
def create_database(env_path):
    """
    Create the database needed for the project. Insert CelebA dataset from personal link

    Take
    -------
    env_path : str
        Path of the environement.
    """
    con = sqlite3.connect(r"%s" % (get_database_path(env_path)))
    cursor = con.cursor()

    # Retrieve datas :
    metadata, dataset = metadata_pull(env_path)

    create_meta_table(cursor, metadata)

    insert_data(cursor, con, dataset)
    
   
create_database(env_path)


def get_database_cursor(env_path):
    """
    Create the database's cursor

    Take
    -------
    env_path : str
        Path of the environement.

    Returns
    -------
    cursor : database.cursor
        Cursor for communicating with the database
    connect : database.connector
        Connector of the database
    """
    connect = sqlite3.connect(get_database_path(env_path))
    cursor = connect.cursor()
    return cursor, connect

def request_data_by_metadata(env_path, array):
    """
    Made a request that pull data according to metadatas

    Take
    -------
    env_path : str
        Path of the environement.
    array : 1D array
        metadata array of 0 and 1

    Returns
    -------
    querry : str, list of str
        filename of possible img according to metadata gave

    """
    cursor, con = get_database_cursor(env_path)
    path = utils.get_path(env_path, "Database")

    metadata, data = metadata_pull(env_path)

    where_str = ""
    querry_array = []
    for i in range(40):  # 40 attributes
        if array[i] != "0":
            where_str += "[%s] = ? AND " % (metadata[i])
            querry_array.append(array[i])

    where_str = where_str[:-4]

    res = cursor.execute("SELECT [name] FROM portrait WHERE %s" %
                         (where_str), tuple(querry_array))
    querry = res.fetchall()
    response = []
    for elem in querry:
        response.append(img_name_to_path(path, elem))
    return response


def img_name_to_path(path, name):
    """
    Convert image name into image path to display

    Take
    -------
    path : str
        Relative path of the image
    name : str
        name of the file choosen

    Returns
    -------
    path : str
        Absolute path of the image.

    """
    return os.path.join(path, "img_dataset", "celeba", "img_align_celeba", "%s" % (name))

def create_querry_array(genre = 0, age = 0, hair_col = 0, 
                        beard = 0, mustache = 0, glasses = 0) :
    """
    Create the metadata array from users choice

    Parameters
    ----------
    genre : int, optional
        Attibute selected by the user for var_name. The default is 0.
        1 = Women, 2 = Men, 0 = Not mentioned
    age : int, optional
        Attibute selected by the user for var_name. The default is 0.
        1 = Young, 2 = Aged, 0 = Not mentioned
    hair_col : int, optional
        Attibute selected by the user for var_name. The default is 0.
        1 = Black, 2 = Black, 3 = Brown, 4 = Grey, 5 = Bold, 
        6 = Other, 0 = Not mentioned
    beard : int, optional
        Attibute selected by the user for var_name. The default is 0.
        1 = With, 2 = Without, 0 = Not mentioned
    mustache : int, optional
        Attibute selected by the user for var_name. The default is 0.
        1 = With, 2 = Without, 0 = Not mentioned
    glasses : int, optional
        Attibute selected by the user for var_name. The default is 0.
        1 = With, 2 = Without, 0 = Not mentioned
    hat : int, optional
        Attibute selected by the user for var_name. The default is 0.
        1 = With, 2 = Without, 0 = Not mentioned

    Returns
    -------
    array : 1D array
        metadata array of 0 and 1

    """
    
    array = ['0'] * 40
    
    array[10] = "-1" # No blurry images
    
    if genre == 1 : # Femme
        array[20] = "-1"
    elif genre == 2 :
        array[20] = "1"
        
    if age == 1 : # Cas jeune
        array[39] = "1"
    elif age == 2 :
        array[39] = "-1"
        
    if hair_col == 1 : # Noirs
        array[8] = "1"
        array[4] = "-1"
    elif hair_col == 2 : # Blonds
        array[9] = "1"
        array[4] = "-1"
    elif hair_col == 3 : # Marron
        array[11] = "1"
        array[4] = "-1"
    elif hair_col == 4 : # Gris
        array[17] = "1"
        array[4] = "-1"
    elif hair_col == 5 : # Chauve
        array[4] = "1"
        array[9] = "-1"
        array[8] = "-1"
        array[11] = "-1"
        array[17] = "-1"
    elif hair_col == 6 : # Autres
        array[4] = "-1"
        array[9] = "-1"
        array[8] = "-1"
        array[11] = "-1"
        array[17] = "-1"
        
        
    if beard == 1 : # Barbu
        array[24] = "-1"
    elif beard == 2 :
        array[24] = "1"
        
    if mustache == 1 : # Avec moustaches
        array[22] = "1"
    elif mustache == 2 :
        array[22] = "-1"
        
    if glasses == 1 : # Avec lunettes
        array[15] = "1"
    elif glasses == 2 :
        array[15] = "-1"
    
    return array

def get_numb_response(env_path, array) :
    """
    Generate number of img that correspond to the querry

    Parameters
    ----------
    env_path : str
        Path of the environement.
    array : 1D array
        metadata array of 0 and 1

    Returns
    -------
    len(resp)
        Number of image

    """
    
    resp = request_data_by_metadata(env_path, array)
    return len(resp)

print(get_numb_response(env_path, create_querry_array(2,1,6,1,1,1)))


cursor, con = get_database_cursor(env_path)

res = cursor.execute("SELECT [name] FROM portrait")
querry = res.fetchall()
print(len(querry))
con.close()

