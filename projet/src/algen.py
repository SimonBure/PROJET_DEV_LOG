from typing import Any

import numpy as np
import torch.nn as nn
import torch
from PIL import Image
from numpy import ndarray
from torch import Tensor

from database import request_data_by_id
from torchvision import transforms
import autoencoder as ae


# TODO Fusionner les 2 fonctions en 1 seule avec un paramètre type="numpy" ou "tensor"
def flatten_img(img_path: str | list[str], img_type="tensor", encode=False) -> Tensor | ndarray:
    """

    Parameters
    ----------
    img_path
    img_type
    encode

    Returns
    -------

    """
    if img_type == "tensor":
        # To transform a numpy array or a PIL image to a torch Tensor
        to_tensor = transforms.ToTensor()
        # To flatten a torch tensor to a tensor with one dimension x number of color channels
        flatten = nn.Flatten(1, 2)

        if type(img_path) is list:
            temp_img = Image.open(img_path[0])  # Temporary img to get its size
            temp_tensor = to_tensor(temp_img)
            size = temp_tensor.shape

            # Global Tensor containing all the images
            flat_img_tensor = torch.zeros((len(img_path), size[0],
                                           size[1] * size[2]))
            # print(f"Global tensor shape: {flat_img_tensor.shape}")
            for i, path in enumerate(img_path):
                if type(path) is str:
                    img = Image.open(path)  # PIL picture

                    if encode:
                        # TODO Aller chercher un autoencoder entraîné pour encoder les photos
                        img = ae.encode()

                    img_tensor = to_tensor(img)  # Transform PIL to torch Tensor
                    # print(f"Image tensor shape: {img_tensor.shape}")
                    flat_img_tensor[i] = flatten(img_tensor)

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

            img_arr = np.transpose(np.array(img), (2, 0, 1))  # Uniformisation of the data dimension
            print(f"Array of the img is: {img_arr.shape}")
            return img_arr.reshape(img_arr.shape[0], -1)

        else:
            raise TypeError("Input should either be a path (str)\
                or a list of paths")

    else:
        raise ValueError("Wrong parameter img_type value. Should either\
                         be tensor or numpy")


def mutate_img(img_encoded: ndarray | Tensor, mutation_rate: float = 0.2, noise: float = 1, mut_type="random")\
        -> ndarray | Tensor:
    """

    Parameters
    ----------
    img_encoded
    mutation_rate
    noise
    mut_type

    Returns
    -------

    """
    # Randomly selects the pixels to be modified
    if mut_type == "random":
        if type(img_encoded) is ndarray:
            img_encoded: np.ndarray
            mut_proba_arr = np.random.random(size=img_encoded.shape)  # Random draw for each pixel of img_encoded
            img_mut = img_encoded
            noise_arr = noise * np.random.normal(size=img_encoded.shape)
            # Adding noise only on pixels where mut_proba_arr is lower than mutation_rate
            img_mut[mut_proba_arr < mutation_rate] += noise_arr[mut_proba_arr < mutation_rate]

        elif type(img_encoded) is Tensor:
            img_encoded: torch.Tensor
            mut_proba_tensor = torch.randn(size=img_encoded.size())
            img_mut = img_encoded
            noise_tensor = noise * torch.randn(size=img_encoded.size())
            img_mut[mut_proba_tensor < mutation_rate] += noise_tensor[mut_proba_tensor < mutation_rate]

        else:
            raise TypeError(f"Input should either be of type np.ndarray \
                or torch.Tensor and not a {type(img_encoded)}")

        return img_mut

    # Modify every pixel with a random noise
    elif mut_type == "uniform":
        # Add random noise on each pixel
        if type(img_encoded) is np.ndarray:
            img_encoded: np.ndarray
            # Adding white noise to the numpy array
            img_mut = img_encoded + noise \
                * np.random.normal(size=img_encoded.shape)

        elif type(img_encoded) is torch.Tensor:
            img_encoded: torch.Tensor
            # Adding white noise to a torch Tensor
            img_mut = img_encoded + noise \
                * torch.randn(size=img_encoded.size())

        else:
            raise TypeError(f"Input should either be of type np.ndarray \
                or torch.Tensor and not a {type(img_encoded)}")

        return img_mut

    else:
        raise ValueError("Chose a valid value for the modif parameter")


def crossing_over(images_encoded: ndarray | Tensor, crossing_rate: float) -> ndarray | Tensor:
    # TODO images_encoded est un array / tensor des images choisies qu'il faut "fusionner"
    # TODO traverser chaque pixel et l'échanger avec un autre d'une image différente

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
    print(f"Shape of first element: {flatten_img(pic_path_list, 'numpy')[0].shape}")

    # Transform the image into a torch Tensor object
    to_tensor = transforms.ToTensor()
    pic_tensor = to_tensor(pic_array)
    print(f"Base Tensor dim: {pic_tensor.shape}")

    # Testing flatten func for Tensor
    flat_pic = flatten_img(pic_path)
    print(f"Tensor dim after flatten func: {flat_pic.shape}")
    flat_pics = flatten_img(pic_path_list)
    print(f"Tensor list dim after flatten: {flat_pics.shape}")

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

    # Testing mutation on all pixels
    some_tensor = torch.randn(size=(5, 5))
    print(f"Base tensor: {some_tensor}")
    # print(f"Mutated tensor: {mutate_img(some_tensor, noise=1, mut_type='uniform')}")

    some_array = np.random.randn(5, 5)
    print(f"Base array: {some_array}")
    # print(f"Mutated array: {mutate_img(some_array, noise=1, mut_type='uniform')}")

    # Testing mutation on random pixels
    mut_tensor_rdm = mutate_img(some_tensor, mutation_rate=0.2)
    print(f"Mutated tensor (random): {mut_tensor_rdm}")
    mut_arr_rdm = mutate_img(some_array, mutation_rate=0.2)
    print(f"Mutated array (random): {mut_arr_rdm}")


