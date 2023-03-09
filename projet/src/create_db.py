import sqlite3
import numpy as np
import os
import platform

import torchvision.datasets
import torchvision.transforms as transforms


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


def get_dataset_path():
    """
    As pathway are different in each computer, compute actual pathway to store data in
    a known path

    Returns
    -------
    path : str
        Path of the dataset download.

    """
    path = os.getcwd()  # Collect the path
    sys = get_sub_sys()  # Collect system data
    if sys == "Windows":
        path += "\img_dataset"  # Windows style path
    else:
        path += "/img_dataset"  # Linux/Mac style path
    return path


def metadata_pull(path):
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
    sys = get_sub_sys()  # Collect system data
    if sys == "Windows":
        path_data = path + "\celeba\list_attr_celeba.txt"  # Windows style path
    else:
        path_data = path + "/celeba/list_attr_celeba.txt"  # Linux/Mac style path

    with open(path_data, "r") as file:
        file.readline()
        metadata = file.readline()

    metadata = metadata.split(" ")

    data = np.loadtxt(path_data, dtype=str, skiprows=2)

    # Test choice
    data = data[0:10]

    return metadata, data


def create_meta_table(cursor, metadata):
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


def create_database(path_of_metadata):
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
    connect = sqlite3.connect('project_db')
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
    path = get_dataset_path()

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


def request_data_by_metadata(array, path):
    """
    Made a request that pull numbers id asked

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

    metadata, data = metadata_pull(path)

    where_str = ""
    for data in metadata[:-1]:  # Because last = empty, img name ?
        where_str += "[%s] = ? AND " % (data)
    where_str = where_str[:-4]

    res = cursor.execute("SELECT * FROM portrait WHERE %s" %
                         (where_str), tuple(array))
    #querry = str(res.fetchall()[0])[2:-3]
    querry = res.fetchall()
    return querry


def img_name_to_path(path, name):
    """
    Convert image name into image path to display

    Returns
    -------
    path : str
        Path of the image.

    """
    return path + "/celeba/img_align_celeba/" + name


def print_database():
    """
    Debug function see what is inside database

    Returns
    -------
    path : str
        Path of the image.

    """
    cursor, con = get_database_cursor()

    res = cursor.execute("SELECT * FROM portrait")
    querry = res.fetchall()
    return querry


if __name__ == '__main__':

    path = get_dataset_path()

    # Download dataset
    if not os.path.exists(path + "/celeba"):  # Prevent from requesting again
        first_data = torchvision.datasets.CelebA(
            root=path, transform=transforms.PILToTensor(), download=True)

    create_database(path)

    numbers = [1, 3, 6]

    meta = ["-1", "-1", "-1", "1", "-1", "-1", "-1", "1", "-1", "-1", "-1", "1", "-1", "-1", "-1", "-1", "-1", "-1", "-1",
            "1", "-1", "1", "-1", "-1", "1", "-1", "-1", "-1", "-1", "-1", "-1", "1", "-1", "-1", "-1", "-1", "-1", "-1", "-1", "1"]

    print(request_data_by_metadata(meta, path))

    print(request_data_by_id(numbers))

    db_cursor, con = get_database_cursor()

    querry = "SELECT * FROM portrait"
    res = db_cursor.execute(querry)
    test = res.fetchall()
    print(test)
    print(print_database())
