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
for a in possibilities :
    meta[0] = a
    for b in possibilities :
        meta[4] = b
        for c in possibilities : 
            meta[5] = c
            for d in possibilities : 
                meta[7] = d
                for e in possibilities :  
                    meta[8] = e
                    for f in possibilities :
                        meta[9] = f
                        for g in possibilities :
                            meta[11] = g
                            for h in possibilities :
                                meta[13] = h
                                for i in possibilities :
                                    meta[15] = i
                                    for j in possibilities :
                                        meta[16] = j
                                        for k in possibilities :
                                            meta[17] = k
                                            for l in possibilities :
                                                meta[20] = l
                                                for m in possibilities :
                                                    meta[22] = m
                                                    for n in possibilities :
                                                        meta[23] = n
                                                        for o in possibilities :
                                                            meta[24] = o
                                                            for p in possibilities :
                                                                meta[25] = p
                                                                for q in possibilities :
                                                                    meta[28] = q
                                                                    for r in possibilities :
                                                                        meta[27] = r
                                                                        for s in possibilities :
                                                                            meta[32] = s
                                                                            for t in possibilities :
                                                                                meta[33] = t
                                                                                for u in possibilities :
                                                                                    meta[30] = u
                                                                                    for v in possibilities :
                                                                                        meta[35] = v
                                                                                        for w in possibilities :
                                                                                            meta[39] = w
                                    
                                                                                            path = get_X_img(env_path, meta, 3, "Project")
                                                                                            if type(path) == list :
                                                                                                for img in path :
                                                                                                    shutil.copy(img, dataset_path)
                                                                                            elif type(path) == str :
                                                                                                shutil.copy(path, dataset_path)
    print("Moitié faite !")


