import os.path

import numpy as np
import torch.nn as nn
import torch
from PIL import Image
from copy import deepcopy
from numpy import ndarray
from torch import Tensor
from torchvision import transforms
from torchsummary import summary

from database import request_data_by_id
import autoencoder as ae
import autoencoder_v2 as ae2
from projet import utils


def flatten_img(img_path: str | list[str]) -> Tensor:
    # TODO Update doc
    """Uses the path stored in img_path to create a Tensor or a ndarray
    in a convenient shape for all the future modifications.
    This function can also encode the image if needed


    Parameters
    ----------
    img_path: str or list[str]
        Path or paths of the images to be retrieved

    Returns
    -------
    flat_img: torch.Tensor
        Contains the values for all the pixels of the images found in the
        given path. If several paths are given, all the images are stored
        in one single Tensor

    >>>path = "./projet/env/Database/img_dataset/celeba/img_align_celeba/000021.jpg"
    >>>flatten_img(path).size()
    torch.Size([3, 38804])

    >>>path_list = ['./projet/env/Database/img_dataset/celeba/img_align_celeba/000001.jpg',
    >>>             './projet/env/Database/img_dataset/celeba/img_align_celeba/000002.jpg',
    >>>             './projet/env/Database/img_dataset/celeba/img_align_celeba/000003.jpg']
    >>>flatten_img(path_list).size()
    torch.Size([3, 3, 38804])
    """
    # Path where to find a trained autoencoder
    model_path = os.path.join(utils.get_path(env_path, "Encoder"), "model.pt")
    autoencoder = ae.load_model(model_path)  # Loading the trained autoencoder

    # To transform a numpy array or a PIL image to a torch Tensor
    to_tensor = transforms.ToTensor()
    # To flatten a torch tensor to a tensor with two dimensions only
    flatten = nn.Flatten(1, 2)

    if type(img_path) is list:
        temp_img = Image.open(img_path[0])  # Temporary img to get its size
        temp_tensor = to_tensor(temp_img)
        temp_tensor = ae2.crop_image_tensor(temp_tensor)
        temp_tensor = ae2.encode(autoencoder, temp_tensor)
        size = temp_tensor.shape

        # Global Tensor containing all the images
        flat_img_tensor = torch.zeros((len(img_path), size[0],
                                       size[1] * size[2]))
        print(f"Global tensor shape: {flat_img_tensor.shape}")

        for i, path in enumerate(img_path):
            if type(path) is str:
                img = Image.open(path)  # PIL Image
                img_tensor = to_tensor(img)  # Image -> Tensor

                # Cropping the tensor to 160x160 pixels
                img_tensor_crop = ae2.crop_image_tensor(img_tensor)

                encoded_tensor = ae2.encode(autoencoder, img_tensor_crop)

                # print(f"Image tensor shape: {img_tensor.shape}")
                flat_img_tensor[i] = flatten(encoded_tensor)

            else:
                raise TypeError("List should contain paths (str)")

        return flat_img_tensor

    elif type(img_path) is str:
        img = Image.open(img_path)  # PIL Image
        img_tensor = to_tensor(img)  # Image -> Tensor

        # Cropping the tensor to 160x160 pixels
        img_tensor_crop = ae2.crop_image_tensor(img_tensor)

        encoded_tensor = ae2.encode(autoencoder, img_tensor_crop)

        return flatten(encoded_tensor)

    else:
        raise TypeError("Input should either be a path (str)\
            or a list of paths")


def deflatten_img(flat_tensor: Tensor, base_encoded_dim: torch.Size) -> Image:
    # TODO Docstring here
    # Path where to find a trained autoencoder
    model_path = os.path.join(utils.get_path(env_path, "Encoder"), "model.pt")
    autoencoder = ae.load_model(model_path)  # Loading the trained autoencoder

    # Reform the original encoded tensor
    unflat_tensor = flat_tensor.reshape(base_encoded_dim)

    decoded_img = ae.decode(autoencoder, unflat_tensor)

    return decoded_img


def mutate_img(img_encoded: Tensor, mutation_rate: float = 0.2,
               noise: float = 1, mut_type="random") -> Tensor:
    """Slightly modifies a or several images given in a ndarray or a
    Tensor with random noise.

    Parameters
    ----------
    img_encoded: torch.Tensor
        Tensor containing one or several images pixels values
    mutation_rate: float
        Probability for a pixel to be modified
    noise: float
        Strength of the random noise, coefficient multiplying the noise
    mut_type: str
        Either random or uniform. If uniform every pixel is perturbed
        with a Gaussian random noise. If random, the pixels to be
        modified are randomly chosen according to mutation_rate

    Returns
    -------
    img_mut: torch.Tensor
        Image or images built on img_encoded with Gaussian random noise
        added to it

    >>>tensor = torch.randn((3, 3))
    tensor([[-3.3558,  1.5579, -0.2904],
        [-0.2572, -0.7410, -0.8748],
        [ 1.2381, -0.4762,  0.3762]])
    >>>mutate_img(tensor, mut_type='uniform')
    tensor([[-3.6974,  1.9027, -0.6431],
        [-0.0740,  0.5979, -1.3189],
        [ 0.7544, -1.8443,  0.2005]])
    >>>mutate_img(tensor, mutation_rate=0.4)
    tensor([[-3.3558,  1.5579, -0.2904],
        [-0.2572, -0.7410, -0.8748],
        [ 1.8747, -0.3136,  0.4488]])
    """
    if type(img_encoded) is Tensor:
        # Add random noise to random pixels
        if mut_type == "random":
            # Randomly selects the pixels to be modified
            mut_proba_tensor = torch.rand(size=img_encoded.size())
            img_mut = img_encoded
            noise_tensor = noise * torch.randn(size=img_encoded.size())
            img_mut[mut_proba_tensor < mutation_rate] += noise_tensor[mut_proba_tensor < mutation_rate]
            return img_mut

        # Add random noise on each pixel
        elif mut_type == "uniform":
            # Adding white noise to a torch Tensor
            img_mut = img_encoded + noise \
                      * torch.randn(size=img_encoded.size())
            return img_mut

    else:
        raise TypeError(f"Input should either be of type or torch.Tensor \
                        and not a {type(img_encoded)}")


def crossing_over(img_encoded: Tensor, crossing_rate: float) -> Tensor:
    """Swaps pixels between the given input images. Swaps are made
    randomly for each pixels.

    Parameters
    ----------
    img_encoded: torch.Tensor
        Tensor containing one or several images pixels values. The image
        where the pixel are drawn is chosen randomly between all the
        input images, with a uniform distribution
    crossing_rate: float
        Probability for a pixel to be swapped between images

    Returns
    -------
    new_img: torch.Tensor
        Image or images on which the crossing-overs were performed

    # TODO Test de code pour les crossing over
    """
    if type(img_encoded) is Tensor:
        for i, img in enumerate(img_encoded):
            crossing_tensor = torch.rand(size=img.size())

            # Randomly choosing which image to swap pixels with
            other_ind = [k for k in range(img_encoded.size()[0]) if k != i]
            chosen_ind = np.random.choice(other_ind)

            new_img = deepcopy(img)
            # Swapping
            new_img[crossing_tensor < crossing_rate] = img_encoded[chosen_ind][crossing_tensor < crossing_rate]
            return new_img

    else:
        raise TypeError(f"Input should either be of type or torch.Tensor \
                        and not a {type(img_encoded)}")


if __name__ == "__main__":
    env_path = "./projet"

    id_nb = 20
    # Path of the 20th image
    pic_path = request_data_by_id(env_path, id_nb)
    print(f"Path for the picture(s): {pic_path}")

    # Path of the first 3 images
    id_array = np.arange(start=0, stop=3, step=1)
    pic_path_list = request_data_by_id(env_path, id_array)
    print(f"Path list: {pic_path_list}")

    # Open the image with PIL
    pic = Image.open(pic_path)

    # Converting the image to a Tensor
    tf_tensor = transforms.ToTensor()
    pic_tensor = tf_tensor(pic)

    # Testing mutation on all pixels
    # some_tensor = torch.randn(size=(3, 3))
    # print(f"Base tensor: {some_tensor}")
    # print(f"Mutated tensor: {mutate_img(some_tensor, mut_type='uniform')}")

    # Testing mutation on random pixels
    # mut_tensor_rdm = mutate_img(some_tensor, mutation_rate=0.2)
    # print(f"Mutated tensor (random): {mut_tensor_rdm}")
    # mut_arr_rdm = mutate_img(some_array, mutation_rate=0.2)
    # print(f"Mutated array (random): {mut_arr_rdm}")

    # Load an autoencoder and encode an img
    # pic.show("Image de base")
    model_path = os.path.join(utils.get_path(env_path, "Encoder"), "model.pt")
    pic_cropped = ae2.crop_image_tensor(pic_tensor)
    print(f"Base size: {pic_tensor.size()}")
    print(f"Cropped size: {pic_cropped.size()}")
    autoencoder = ae.load_model(model_path)  # Loading the trained model
    encoded_img = ae2.encode(autoencoder, pic_cropped)  # Encoding the tensor
    print(f"Image tensor : {encoded_img.size()}")
    decoded = ae.decode(autoencoder, encoded_img)  # Decoding the tensor
    # decoded.show("Image décodée")

    # Testing flatten on an image
    flat_encoded_tensor = flatten_img(pic_path)
    print(f"Flat encoded size: {flat_encoded_tensor.size()}")

    # Testing flatten on several images
    flat_several = flatten_img(pic_path_list)
    print(f"Several image tensor size: {flat_several.size()}")

    # Testing resize
    deflat_img = deflatten_img(flat_encoded_tensor, encoded_img.size())
    deflat_img.show()

    # Testing mutation on flat encoded image
    mut_img = mutate_img(flat_encoded_tensor, mut_type="uniform")
    deflat_img = deflatten_img(mut_img, encoded_img.size())
    deflat_img.show()

    mut_img = mutate_img(flat_encoded_tensor, mut_type="random")
    deflat_img = deflatten_img(mut_img, encoded_img.size())
    deflat_img.show()
