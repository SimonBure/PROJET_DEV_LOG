import sqlite3
import numpy as np
import os
import utils


def get_database_path():
    """
    Retrieve access to database to query her

    Returns
    -------
    path : str
        Path of the dataset download.

    """
    path = utils.get_path("Database")
    data_loc = os.path.join(path, "project.db")
    return data_loc


def metadata_pull():
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
    path = utils.get_path("Img")
    path_data = os.path.join(path, "celeba", "list_attr_celeba.txt")

    with open(path_data, "r") as file:
        file.readline()
        metadata = file.readline()

    metadata = metadata.split(" ")

    data = np.loadtxt(path_data, dtype=str, skiprows=2)

    # Split for testing
    data = data[0:10]

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


def create_database():
    """
    Create the database needed for the project. Insert CelebA dataset from personal link

    """
    con = sqlite3.connect(r"%s" % (get_database_path()))
    cursor = con.cursor()

    # Retrieve datas :
    metadata, dataset = metadata_pull()

    create_meta_table(cursor, metadata)

    insert_data(cursor, con, dataset)


def get_database_cursor():
    """
    Create the database's cursor

    Returns
    -------
    cursor : database.cursor
        Cursor for communicating with the database
    connect : database.connector
        Connector of the database
    """
    connect = sqlite3.connect(get_database_path())
    cursor = connect.cursor()
    return cursor, connect


def request_data_by_id(numbers):
    """
    Made a request that pull numbers id asked

    Take
    -------
    numbers : int, list, tuple or 1D array
        id's image of database to pull

    Returns
    -------
    querry : str, list of str
        filename of the selected id number

    """
    cursor, con = get_database_cursor()
    path = utils.get_path("Database")

    if type(numbers) == int:
        res = cursor.execute(
            "SELECT [name] FROM portrait WHERE id = %s" % (numbers))
        name = str(res.fetchall()[0])[2:-3]
        querry = img_name_to_path(path, name)
    else:
        querry = []
        for id in numbers:
            res = cursor.execute(
                "SELECT [name] FROM portrait WHERE id = %s" % (id))
            name = str(res.fetchall()[0])[2:-3]
            querry.append(img_name_to_path(path, name))

    return querry


def request_data_by_metadata(array):
    """
    Made a request that pull data according to metadatas

    Take
    -------
    array : 1D array
        metadata array of 0 and 1
    path : str
        path of metadata

    Returns
    -------
    querry : str, list of str
        filename of possible img according to metadata gave

    """
    cursor, con = get_database_cursor()
    path = utils.get_path("Database")

    metadata, data = metadata_pull()

    where_str = ""
    for data in metadata[:-1]:  # Because last = empty, img name ?
        where_str += "[%s] = ? AND " % (data)
    where_str = where_str[:-4]

    res = cursor.execute("SELECT [name] FROM portrait WHERE %s" %
                         (where_str), tuple(array))
    querry = str(res.fetchall()[0])[2:-3]
    querry = img_name_to_path(path, querry)
    return querry


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


def print_database():
    """
    Debug function see what is inside database

    Returns
    -------
    querry : list of str
        All rows and lines of the dataset

    """
    cursor, con = get_database_cursor()

    res = cursor.execute("SELECT * FROM portrait")
    querry = res.fetchall()
    return querry


if __name__ == '__main__':
    
    if os.path.exists(utils.get_path("Database")) :
        if os.path.exists(get_database_path()) :
            print("Database already exist")
        else :
            create_database()   

    numbers = [1, 3, 6]

    meta = ["-1", "-1", "-1", "1", "-1", "-1", "-1", "1", "-1", "-1", "-1", "1", "-1", "-1", "-1", "-1", "-1", "-1", "-1",
            "1", "-1", "1", "-1", "-1", "1", "-1", "-1", "-1", "-1", "-1", "-1", "1", "-1", "-1", "-1", "-1", "-1", "-1", "-1", "1"]

    print(request_data_by_metadata(meta))
    
    print(request_data_by_id(2))

    print(request_data_by_id(numbers))

    print(print_database())