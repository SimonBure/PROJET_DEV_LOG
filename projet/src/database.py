import sqlite3
import numpy as np
import os
import utils


def get_database_path(env_path, who = "idkit"):
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
    if who == "Project" :
        data_loc = os.path.join(path, "project.db")
    elif who == "Auto" :
        data_loc = os.path.join(path, "autoencode.db")
    else :
        data_loc = os.path.join(path, "idkit.db")
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
        metadata = file.readline()

    metadata = metadata.split(" ")

    data = np.loadtxt(path_data, dtype=str, skiprows=2)

    # Split for testing
    #data = data[0:10]

    return metadata, data


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


def create_database(env_path, who = "Project"):
    """
    Create the database needed for the project. Insert CelebA dataset from personal link

    Take
    -------
    env_path : str
        Path of the environement.
    """
    con = sqlite3.connect(r"%s" % (get_database_path(env_path,"Project")))
    cursor = con.cursor()

    # Retrieve datas :
    metadata, dataset = metadata_pull(env_path)

    create_meta_table(cursor, metadata)

    insert_data(cursor, con, dataset)


def get_database_cursor(env_path, who = "idkit"):
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
    
    if os.path.exists(get_database_path(env_path, who)) :
        connect = sqlite3.connect(get_database_path(env_path, who))
        cursor = connect.cursor()
    else :
        raise(Exception("Database do not exist"))
    return cursor, connect


def request_data_by_id(env_path, numbers, who = "Project"):
    """
    Made a request that pull numbers id asked

    Take
    -------
    env_path : str
        Path of the environement.
    numbers : int, list, tuple or 1D array
        id's image of database to pull

    Returns
    -------
    querry : str, list of str
        filename of the selected id number

    """
    cursor, con = get_database_cursor(env_path, who)
    path = utils.get_path(env_path, "Database")

    if type(numbers) == int:
        res = cursor.execute(
            "SELECT [name] FROM portrait WHERE id = %s" % (numbers))
        name = res.fetchall()[0]
        querry = img_name_to_path(path, name)
    else:
        querry = []
        for id in numbers:
            res = cursor.execute(
                "SELECT [name] FROM portrait WHERE id = %s" % (id))
            name = res.fetchall()[0]
            querry.append(img_name_to_path(path, name))

    return querry


def request_data_by_metadata(env_path, array, who = "idkit"):
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
    cursor, con = get_database_cursor(env_path, who)
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


def get_5_img(env_path, array=[], who = "idkit"):
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
    if array == []:
        numbers = np.random.randint(1, 634, size=5)
        path_img_list = request_data_by_id(env_path, numbers)
    else:
        path_img_list = request_data_by_metadata(env_path, array, who)
        if len(path_img_list) > 5:
            path_img_list_temp = []
            numbers = np.random.choice(list(range(len(path_img_list))), size=5, replace = False )
            for i in numbers:
                path_img_list_temp.append(path_img_list[i])
            path_img_list = path_img_list_temp
        elif len(path_img_list) < 5:
            print("Not enough img in database")
            return 0

    return path_img_list


def print_database(env_path, who = "idkit"):
    """
    Debug function see what is inside database

    Take
    -------
    env_path : str
        Path of the environement.

    Returns
    -------
    querry : list of str
        All rows and lines of the dataset

    """
    cursor, con = get_database_cursor(env_path, who)

    res = cursor.execute("SELECT * FROM portrait")
    querry = res.fetchall()
    return querry

def create_querry_array(genre = 0, age = 0, hair_col = 0, 
                        glasses = 0, mustache = 0, beard = 0):
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
    
    

if __name__ == '__main__':

    env_path = "../"

    if os.path.exists(utils.get_path(env_path, "Database")):
        if os.path.exists(get_database_path(env_path)):
            print("Database already exist")
        else:
            create_database(env_path)

    numbers = [1, 3, 6]

    meta = ["-1", "-1", "-1", "1", "-1", "-1", "-1", "1", "-1", "-1", "-1", "1", "-1", "-1", "-1", "-1", "-1", "-1", "-1",
            "1", "-1", "1", "-1", "-1", "1", "-1", "-1", "-1", "-1", "-1", "-1", "1", "-1", "-1", "-1", "-1", "-1", "-1", "-1", "1"]

    print(request_data_by_metadata(env_path, meta))

    print(request_data_by_id(env_path, 720))

    meta_incomplete = ["0", "-1", "-1", "1", "-1", "-1", "-1", "1", "-1", "-1", "-1", "1", "-1", "-1", "-1", "-1", "-1", "-1", "-1",
                       "1", "-1", "1", "-1", "-1", "1", "-1", "-1", "-1", "-1", "-1", "-1", "1", "-1", "-1", "-1", "-1", "-1", "-1", "-1", "1"]
    print(request_data_by_metadata(env_path, meta_incomplete))

    print(get_5_img(env_path))

    print(get_5_img(env_path, meta))

    meta_incomplete = ["0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "1", "0", "0", "0", "0", "0", "0", "0",
                       "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "1"]

    print(get_5_img(env_path, meta_incomplete))

    print(request_data_by_id(env_path, numbers))

    print(len(print_database(env_path, "Auto")))
    
    print(create_querry_array())
    
    print(get_numb_response(env_path, create_querry_array(0)))
    
