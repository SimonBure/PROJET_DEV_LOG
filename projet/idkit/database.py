import sqlite3
import numpy as np
import os
import utils
import logging


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
    data_loc = os.path.join(path, "idkit.db")
    return data_loc


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
    if os.path.exists(get_database_path(env_path)):
        connect = sqlite3.connect(get_database_path(env_path))
        cursor = connect.cursor()
    else:
        raise (Exception("Database do not exist"))
    return cursor, connect


def request_data_by_id(env_path, numbers):
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
    query : str, list of str
        filename of the selected id number
        
    
    >> request_data_by_id("../", 1)[-10:]
    001239.jpg
    """
    cursor, con = get_database_cursor(env_path)
    path = utils.get_path(env_path, "Database")

    if type(numbers) == int:
        res = cursor.execute(
            "SELECT [name] FROM portrait WHERE id = %s" % (numbers))
        name = res.fetchall()[0]
        query = img_name_to_path(path, name)
    else:
        query = []
        for id in numbers:
            res = cursor.execute(
                "SELECT [name] FROM portrait WHERE id = %s" % (id))
            name = res.fetchall()[0]
            query.append(img_name_to_path(path, name))

    return query


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
    query : str, list of str
        filename of possible img according to metadata gave
        
    >> request_data_by_metadata("../", 
                                ["0", "0", "0", "0", "0", "0", "0", "0",
                                 "0", "0", "0", "1", "0", "0", "0", "0",
                                 "0", "0", "0","0", "0", "0", "0", "0", 
                                 "0", "0", "0", "0", "0", "0", "0", "0", 
                                 "0", "0", "0", "0", "0", "0", "0", "1"])[0][-10:]
    007612.jpg
    """
    cursor, con = get_database_cursor(env_path)
    path = utils.get_path(env_path, "Database")

    metadata = ["5_o_Clock_Shadow", "Arched_Eyebrows", "Attractive",
                "Bags_Under_Eyes", "Bald", "Bangs", "Big_Lips", "Big_Nose",
                "Black_Hair", "Blond_Hair", "Blurry", "Brown_Hair",
                "Bushy_Eyebrows", "Chubby", "Double_Chin", "Eyeglasses",
                "Goatee", "Gray_Hair", "Heavy_Makeup", "High_Cheekbones",
                "Male", "Mouth_Slightly_Open", "Mustache", "Narrow_Eyes",
                "No_Beard", "Oval_Face", "Pale_Skin", "Pointy_Nose",
                "Receding_Hairline", "Rosy_Cheeks", "Sideburns", "Smiling",
                "Straight_Hair", "Wavy_Hair", "Wearing_Earrings",
                "Wearing_Hat", "Wearing_Lipstick", "Wearing_Necklace",
                "Wearing_Necktie", "Young"]

    where_str = ""
    querry_array = []
    for i in range(40):  # 40 attributes
        if array[i] != "0":
            where_str += "[%s] = ? AND " % (metadata[i])
            querry_array.append(array[i])

    where_str = where_str[:-4]

    res = cursor.execute("SELECT [name] FROM portrait WHERE %s" %
                         (where_str), tuple(querry_array))
    query = res.fetchall()
    response = []
    for elem in query:
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
    return os.path.join(path, "img_dataset", "%s" % (name))


def get_5_img(env_path, array=[]):
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
    logging.info('Database - Retrieving 5 images')
    if array == []:
        numbers = np.random.randint(1, 634, size=5)
        path_img_list = request_data_by_id(env_path, numbers)
    else:
        path_img_list = request_data_by_metadata(env_path, array)
        if len(path_img_list) > 5:
            path_img_list_temp = []
            numbers = np.random.choice(
                list(range(len(path_img_list))), size=5, replace=False)
            for i in numbers:
                path_img_list_temp.append(path_img_list[i])
            path_img_list = path_img_list_temp
        elif len(path_img_list) < 5:
            print("Not enough img in database")
            logging.warning('Not enough img in database for selected querry')
            return 0

    return path_img_list


def print_database(env_path):
    """
    Debug function see what is inside database

    Take
    -------
    env_path : str
        Path of the environement.

    Returns
    -------
    query : list of str
        All rows and lines of the dataset
        
    >> len(print_database("../"))
    634
    """
    cursor, con = get_database_cursor(env_path)

    res = cursor.execute("SELECT * FROM portrait")
    query = res.fetchall()
    return query


def create_query_array(genre = 0, age = 0, hair_col = 0,
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
        
    >> create_querry_array()
    ["0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "-1", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0"]
    """

    array = ['0'] * 40

    array[10] = "-1"  # No blurry images

    if genre == 1:  # Femme
        array[20] = "-1"
    elif genre == 2:
        array[20] = "1"

    if age == 1:  # Cas jeune
        array[39] = "1"
    elif age == 2:
        array[39] = "-1"

    if hair_col == 1:  # Noirs
        array[8] = "1"
        array[4] = "-1"
    elif hair_col == 2:  # Blonds
        array[9] = "1"
        array[4] = "-1"
    elif hair_col == 3:  # Marron
        array[11] = "1"
        array[4] = "-1"
    elif hair_col == 4:  # Gris
        array[17] = "1"
        array[4] = "-1"
    elif hair_col == 5:  # Chauve
        array[4] = "1"
        array[9] = "-1"
        array[8] = "-1"
        array[11] = "-1"
        array[17] = "-1"
    elif hair_col == 6:  # Autres
        array[4] = "-1"
        array[9] = "-1"
        array[8] = "-1"
        array[11] = "-1"
        array[17] = "-1"

    if beard == 1:  # Barbu
        array[24] = "-1"
    elif beard == 2:
        array[24] = "1"

    if mustache == 1:  # Avec moustaches
        array[22] = "1"
    elif mustache == 2:
        array[22] = "-1"

    if glasses == 1:  # Avec lunettes
        array[15] = "1"
    elif glasses == 2:
        array[15] = "-1"

    return array


def get_numb_response(env_path, array):
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
        
    >> get_numb_response("../", ["0", "0", "0", "0", "0", "0", "0", "0",
                                 "0", "0", "0", "1", "0", "0", "0", "0",
                                 "0", "0", "0","0", "0", "0", "0", "0", 
                                 "0", "0", "0", "0", "0", "0", "0", "0", 
                                 "0", "0", "0", "0", "0", "0", "0", "1"])
    59
    """

    resp = request_data_by_metadata(env_path, array)
    return len(resp)


if __name__ == '__main__':

    env_path = "../"

    numbers = [1, 3, 6]

    print(request_data_by_id(env_path, 2))

    print(get_5_img(env_path))

    meta_incomplete = ["0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0",
                       "1", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0",
                       "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0",
                       "0", "0", "0", "0", "0", "0", "1"]

    print(get_5_img(env_path, meta_incomplete))

    print(request_data_by_id(env_path, numbers))

    print(len(print_database(env_path)))

    print(create_query_array())

    print(get_numb_response(env_path, create_query_array(0)))
