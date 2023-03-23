import numpy as np
import torch.nn as nn
import torch
from PIL import Image
from create_db import request_data_by_id
from torchvision import transforms
# import utils
import autoencoder as ae


def flatten_img_tensor(img_path: str or list[str], flatten_dim: int, encode=False) -> torch.Tensor:
    """_summary_

    Args:
        img_path (strorlist[str]): _description_
        flatten_dim (int): _description_
        encode (bool, optional): _description_. Defaults to False.

    Raises:
        TypeError: _description_
        TypeError: _description_

    Returns:
        torch.Tensor: _description_
    """
    # To transform a numpy array or a PIL image to a torch Tensor
    tensor_transfo = transforms.ToTensor()
    # To flatten a torch tensor to a tensor with only one dimension
    flat_fct = nn.Flatten(0, 2)

    if type(img_path) is list:
        flat_img_tensor = torch.zeros(size=(len(img_path), flatten_dim))
        for i, path in enumerate(img_path):
            if type(path) is str:
                img = Image.open(img_path)  # PIL picture

                if encode:
                    # TODO Aller chercher un autoencoder entraîné
                    img = ae.encode()

                img_tensor = tensor_transfo(img)
                flat_img_tensor[i] = flat_fct(img_tensor)

            else:
                raise TypeError("List should contain paths (str)")
        return flat_img_tensor

    elif type(img_path) is str:
        img = Image.open(img_path)  # PIL picture

        if encode:
            # TODO Aller chercher un autoencoder entraîné
            img = ae.encode()

        img_tensor = tensor_transfo(img)
        return flat_fct(img_tensor)

    else:
        raise TypeError("Input should either be a path (str)\
            or a list of paths")


def flatten_img_numpy(img_path: str or list[str], encode=False) -> np.ndarray:
    """_summary_

    Args:
        img_path (strorlist[str]): _description_
        encode (bool, optional): _description_. Defaults to False.

    Raises:
        TypeError: _description_
        TypeError: _description_

    Returns:
        np.ndarray: _description_
    """
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

        img_arr = np.array(img)
        return np.concatenate(img_arr)

    else:
        raise TypeError("Input should either be a path (str)\
            or a list of paths")


def mutate_img(img_encoded: np.ndarray or torch.Tensor, modif="random") -> np.ndarray:
    pass
    # TODO Tester différentes transformations sur les images:
    # Modifier les pixels aléatoirement, uniformément
    if modif == "random":
        noise_intensity = 0.4
        if type(img_encoded) is np.ndarray:
            # Adding white noise to the numpy array
            img_mut = img_encoded + noise_intensity \
                * np.random.normal(0, 1, img_encoded.shape)

        elif type(img_encoded) is torch.Tensor:
            # Adding white noise to a torch Tensor
            img_mut = img_encoded + noise_intensity \
                * torch.randn(img_encoded.size)

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
    # Diviser les différentes images au hasard, les regrouper
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

    # print(f"Flatten img: {flatten_img_numpy(pic_path)}")
    print(f"Flatten shape: {flatten_img_numpy(pic_path).shape}")

    print(f"Flatten list shape: {flatten_img_numpy(pic_path_list).shape}")

    # Open the image with PIL
    pic = Image.open(pic_path)
    print(f"Type of the picture: {type(pic)}")

    # Convert the image into a ndarray
    pic_array = np.array(pic)
    # print(f"Array of the pixels: {pic_array}")
    print(f"Shape of the pic: {pic_array.shape}")

    # Transform the image into a torch Tensor object
    tensor_transfo = transforms.ToTensor()
    pic_tensor = tensor_transfo(pic_array)
    print(f"Type of the tensor: {type(pic_tensor)}")
    print(f"Shape of the tensor: {pic_tensor.shape}")
    # print(f"Picture in tensor form: {pic_tensor}")

    # Trying with oliveti dataset
    oliveti_faces = ae.faces.images  # ndarray of all the pictures
    fst_face = oliveti_faces[0]
    a_tensor = tensor_transfo(fst_face)
    print(f"Olivetti shape: {a_tensor.shape}")

    # Create an autoencoder
    autoencoder = ae.Autoencoder()

    # Encoding an image
    pic_encoded = ae.encode(autoencoder, fst_face)
    print(f"Shape of the encoded tensor: {pic_encoded.shape}")
    flat = nn.Flatten(0, 2)
    pic_flatten = flat(pic_encoded)
    print(f"Type after Flatten ?: {type(pic_flatten)}")
    print(f"Flatten shape {pic_flatten.size()}")

    zero_tensor = torch.zeros(size=(50, 2304))
    print(zero_tensor[39])
