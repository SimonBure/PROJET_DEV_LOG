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


def flatten_img(img_path: str | list[str], img_type="tensor", encode=True)\
        -> Tensor | ndarray:
    """Uses the path stored in img_path to create a Tensor or a ndarray
    in a convenient shape for all the future modifications.
    This function can also encode the image if needed

    Parameters
    ----------
    img_path: str or list[str]
        Path or paths of the images to be retrieved
    img_type: str
        Specifies the type of the object to be returned
    encode: bool
        True if the images need to be encoded by the autoencoder,
        False otherwise

    Returns
    -------
    flat_img: torch.Tensor or numpy.ndarray
        Contains the values for all the pixels of the images found in the
        given path. If several paths are given, all the images are stored
        in one single object

    >>>path = "./projet/env/Database/img_dataset/celeba/img_align_celeba/000021.jpg"
    >>>flatten_img(path, img_type="numpy").shape
    (3, 38804)

    >>>flatten_img(path).size()
    torch.Size([3, 38804])

    >>>path_list = ['./projet/env/Database/img_dataset/celeba/img_align_celeba/000001.jpg',
    >>>             './projet/env/Database/img_dataset/celeba/img_align_celeba/000002.jpg',
    >>>             './projet/env/Database/img_dataset/celeba/img_align_celeba/000003.jpg']
    >>>flatten_img(path_list).size()
    torch.Size([3, 3, 38804])
    """
    model_path = os.path.join(utils.get_path(env_path, "Encoder"), "model.pt")
    if img_type == "tensor":
        # To transform a numpy array or a PIL image to a torch Tensor
        to_tensor = transforms.ToTensor()
        # To flatten a torch tensor to a tensor with two dimensions only
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
            img_tensor = to_tensor(img)
            img_tensor = ae2.crop_image_tensor(img_tensor)

            if encode:
                model = ae.load_model(model_path)
                img_tensor = ae2.encode(model, img_tensor)

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

            # Uniformization of the data dimension
            img_arr = np.transpose(np.array(img), (2, 0, 1))
            print(f"Array of the img is: {img_arr.shape}")
            return img_arr.reshape(img_arr.shape[0], -1)

        else:
            raise TypeError("Input should either be a path (str)\
                or a list of paths")

    else:
        raise ValueError("Wrong parameter img_type value. Should either\
                         be tensor or numpy")


def deflatten_img(flat_tensor: Tensor, base_width: int = 18,
                  base_length: int = 18, decode=True) -> Image:
    base_dim = (base_width, base_length, 3)
    if decode:
        model_path = os.path.join(utils.get_path(env_path, "Encoder"), "model.pt")
        model = ae.load_model(model_path)
        unflat_img = flat_tensor.reshape((64, 18, 18))
        decoded_img = ae.decode(model, unflat_img)
        return decoded_img

    else:
        rev_tf = transforms.ToPILImage()
        unflat_tensor = flat_tensor.reshape(base_dim)
        return rev_tf(unflat_tensor)


def mutate_img(img_encoded: ndarray | Tensor, mutation_rate: float = 0.2,
               noise: float = 1, mut_type="random") -> ndarray | Tensor:
    """Slightly modifies a or several images given in a ndarray or a
    Tensor with random noise.

    Parameters
    ----------
    img_encoded: numpy.ndarray or torch.Tensor
        Object containing an or several images pixels values
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
    img_mut: numpy.ndarray or torch.Tensor
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
    # Randomly selects the pixels to be modified
    if mut_type == "random":
        if type(img_encoded) is ndarray:
            img_encoded: ndarray
            # Random draw for each pixel of img_encoded
            mut_proba_arr = np.random.random(size=img_encoded.shape)
            img_mut = img_encoded
            noise_arr = noise * np.random.normal(size=img_encoded.shape)
            # Adding noise only on pixels where mut_proba_arr is lower than mutation_rate
            img_mut[mut_proba_arr < mutation_rate] += noise_arr[mut_proba_arr < mutation_rate]

        elif type(img_encoded) is Tensor:
            img_encoded: Tensor
            mut_proba_tensor = torch.rand(size=img_encoded.size())
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
            img_encoded: ndarray
            # Adding white noise to the numpy array
            img_mut = img_encoded + noise \
                * np.random.normal(size=img_encoded.shape)

        elif type(img_encoded) is torch.Tensor:
            img_encoded: Tensor
            # Adding white noise to a torch Tensor
            img_mut = img_encoded + noise \
                * torch.randn(size=img_encoded.size())

        else:
            raise TypeError(f"Input should either be of type np.ndarray \
                or torch.Tensor and not a {type(img_encoded)}")

        return img_mut

    else:
        raise ValueError("Chose a valid value for the modif parameter")


def crossing_over(images_encoded: ndarray | Tensor,
                  crossing_rate: float) -> ndarray | Tensor:
    """Swaps pixels between the given input images. Swaps are made
    randomly for each pixels.

    Parameters
    ----------
    images_encoded: numpy.ndarray or torch.Tensor
        Object containing one or several images pixels values. The images
        where the pixel are drawn is chosen randomly between all the
        input images, with a uniform distribution
    crossing_rate: float
    Probability for a pixel to be swapped between images

    Returns
    -------
    new_img: numpy.ndarray or torch.Tensor
        Image or images on which the crossing-overs are performed

    # TODO Test de code pour les crossing over
    """
    if type(images_encoded) is ndarray:
        images_encoded: ndarray
        for i, img in enumerate(images_encoded):
            crossing_arr = np.random.random(size=img.shape)
            # Randomly choosing which image to swap pixels with
            other_ind = [j for j in range(images_encoded.shape[0]) if j != i]
            chosen_ind = np.random.choice(other_ind)

            new_img = deepcopy(img)
            # Swapping
            new_img[crossing_arr < crossing_rate] = images_encoded[chosen_ind][crossing_arr < crossing_rate]
            return new_img

    elif type(images_encoded) is Tensor:
        images_encoded: Tensor
        for i, img in enumerate(images_encoded):
            crossing_tensor = torch.rand(size=img.size())

            # Randomly choosing which image to swap pixels with
            other_ind = [k for k in range(images_encoded.size()[0]) if k != i]
            chosen_ind = np.random.choice(other_ind)

            new_img = deepcopy(img)
            # Swapping
            new_img[crossing_tensor < crossing_rate] = images_encoded[chosen_ind][crossing_tensor < crossing_rate]
            return new_img

    else:
        raise TypeError(f"Input should either be of type np.ndarray \
            or torch.Tensor and not a {type(images_encoded)}")


if __name__ == "__main__":
    env_path = "./projet"

    id_nb = 20
    # Path of the 20th image
    pic_path = request_data_by_id(env_path, id_nb)
    print(f"Path for the picture(s): {pic_path}")

    # Path of the first 4 images
    id_array = np.arange(start=0, stop=4, step=1)
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

    # some_array = np.random.randn(3, 3)
    # print(f"Base array: {some_array}")
    # print(f"Mutated array: {mutate_img(some_array, mut_type='uniform')}")

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

    # Testing flatten on encoded image
    flat_encoded_tensor = flatten_img(pic_path)
    print(f"Flat encoded shape: {flat_encoded_tensor.size()}")

    # Testing resize
    deflat_img = deflatten_img(flat_encoded_tensor)
    deflat_img.show()
    # deflatten_img.show("Image après le processus de flatten - deflatten")

    # Testing mutation on flat encoded image
    # mut_img = mutate_img(flat_encoded_img)
