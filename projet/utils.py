# -*- coding: utf-8 -*-
import os
import platform


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


def get_path():
    """
    As pathway are different in each computer, compute actual pathway to store data in
    a known path

    Returns
    -------
    path : str
        Path of the dataset download.

    """
    path = os.getcwd()
    path = os.path.join(path, "env")

    return path  # Collect the path
