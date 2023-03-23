from typing import Any

import numpy as np
import torch.nn as nn
import torch
from PIL import Image
from numpy import ndarray
from torch import Tensor

from create_db import request_data_by_id
from torchvision import transforms
import autoencoder as ae


# TODO Fusionner les 2 fonctions en 1 seule avec un paramètre type="numpy" ou "tensor"
def flatten_img(img_path: str | list[str], img_type="tensor", encode=False) -> Tensor | ndarray | Any:
    if img_type == "tensor":
        # To transform a numpy array or a PIL image to a torch Tensor
        to_tensor = transforms.ToTensor()
        # To flatten a torch tensor to a tensor with one dimension x number of color channels
        flatten = nn.Flatten(1, 2)

        if type(img_path) is list:
            temp_img = Image.open(img_path[0])  # Temporary img to get it size
            temp_tensor = to_tensor(temp_img)
            size = temp_tensor.shape

            # Global Tensor containing all the images
            flat_img_tensor = torch.zeros((len(img_path),
                                           size[2] * size[1],
                                           size[0]))
            for i, path in enumerate(img_path):
                if type(path) is str:
                    img = Image.open(path)  # PIL picture

                    if encode:
                        # TODO Aller chercher un autoencoder entraîné pour encoder les photos
                        img = ae.encode()

                    img_tensor = to_tensor(img)  # Transform PIL to torch Tensor
                    flat_img_tensor[i] = flatten(img_tensor)  #

                else:
                    raise TypeError("List should contain paths (str)")
            return flat_img_tensor

        elif type(img_path) is str:
            img = Image.open(img_path)  # PIL picture

            if encode:
                # TODO Aller chercher un autoencoder entraîné
                img = ae.encode()

            img_tensor = to_tensor(img)
            return flatten(img_tensor)

        else:
            raise TypeError("Input should either be a path (str)\
                or a list of paths")

    elif img_type == "numpy":
        if type(img_path) is list:
            flat_img_list = [0] * len(img_path)
            for i, path in enumerate(img_path):
                if type(path) is str:
                    img = Image.open(path)  # PIL picture

                    if encode:
                        # TODO Aller chercher un autoencoder entraîné
                        img = ae.encode()

                    img_arr = np.array(img)
                    flat_img_list[i] = np.concatenate(img_arr)

                else:
                    raise TypeError("List should contain paths (str)")
            return np.array(flat_img_list)

        elif type(img_path) is str:
            img = Image.open(img_path)  # PIL picture

            if encode:
                # TODO Aller chercher un autoencoder entraîné
                img = ae.encode()
                img = img.numpy()

            img_arr = np.array(img)
            return np.concatenate(img_arr)

        else:
            raise TypeError("Input should either be a path (str)\
                or a list of paths")

    else:
        raise ValueError("Wrong parameter img_type value. Should either\
                         be tensor or numpy")


def mutate_img(img_encoded: np.ndarray | torch.Tensor, noise: float, modif="random") -> np.ndarray:
    """

    Parameters
    ----------
    img_encoded
    noise
    modif

    Returns
    -------

    """
    # TODO Tester différentes transformations sur les images:
    # Modifier les pixels aléatoirement, uniformément
    if modif == "random":
        if type(img_encoded) is np.ndarray:
            img_encoded: np.ndarray
            # Adding white noise to the numpy array
            img_mut = img_encoded + noise \
                * np.random.normal(size=img_encoded.shape)

        elif type(img_encoded) is torch.Tensor:
            img_encoded: torch.Tensor
            # Adding white noise to a torch Tensor
            img_mut = img_encoded + noise \
                * torch.randn(img_encoded.size())

        else:
            raise TypeError(f"Input should either be of type np.ndarray \
                or torch.Tensor and not a {type(img_encoded)}")
        return img_mut

    elif modif == "uniform":
        if type(img_encoded) is np.ndarray:
            img_mut = img_encoded  # TODO Transformation du tableau

        elif type(img_encoded) is torch.Tensor:
            img_mut = img_encoded  # TODO Transformation avec nn.Linear ?

        else:
            raise TypeError(f"Input should either be of type np.ndarray \
                or torch.Tensor and not a {type(img_encoded)}")

    else:
        raise ValueError("Chose a valid value for the modif parameter")


def crossing_over(images_encoded: np.ndarray or torch.Tensor) -> np.ndarray:
    # TODO images_encoded est une liste des images choisies qu'il faut "fusionner"
    # TODO Diviser les différentes images au hasard, les regrouper
    pass


if __name__ == "__main__":
    env_path = "./projet"

    id_nb = 20
    # Path of the 20th image
    pic_path = request_data_by_id(env_path, id_nb)
    print(f"Path for the picture(s): {pic_path}")

    # Path of the first 20 images
    id_array = np.arange(start=0, stop=20, step=1)
    pic_path_list = request_data_by_id(env_path, id_array)

    # Open the image with PIL
    pic = Image.open(pic_path)
    print(f"Type of the picture: {type(pic)}")

    # Convert the image into a ndarray
    pic_array = np.array(pic)
    # print(f"Array of the pixels: {pic_array}")
    print(f"Shape of the pic: {pic_array.shape}")

    # Testing flatten func for ndarray
    print(f"Flat ndarray shape: {flatten_img(pic_path, 'numpy').shape}")
    print(f"Flat ndarray list shape: {flatten_img(pic_path_list, 'numpy').shape}")

    # Transform the image into a torch Tensor object
    to_tensor = transforms.ToTensor()
    pic_tensor = to_tensor(pic_array)
    print(f"Base Tensor dim: {pic_tensor.shape}")

    # Testing flatten func for Tensor
    flat_pic = flatten_img(pic_path)
    print(f"Tensor dim after flatten func: {flat_pic.shape}")

    # Trying with oliveti dataset
    oliveti_faces = ae.faces.images  # ndarray of all the pictures
    fst_face = oliveti_faces[0]
    a_tensor = to_tensor(fst_face)
    print(f"Olivetti shape: {a_tensor.shape}")

    # Create an autoencoder
    autoencoder = ae.Autoencoder()

    # Encoding an image
    pic_encoded = ae.encode(autoencoder, fst_face)
    print(f"Shape of the encoded tensor: {pic_encoded.shape}")

    # Testing mutation func
    some_tensor = torch.randn(size=(5, 5))
    print(f"Base tensor: {some_tensor}")
    print(f"Mutated tensor: {mutate_img(some_tensor, noise=1)}")

    some_array = np.random.randn(5, 5)
    print(f"Base array: {some_array}")
    print(f"Mutated array: {mutate_img(some_array, 1)}")
