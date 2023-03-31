import os.path

import numpy as np
import torch.nn as nn
import torch
from PIL import Image
from PIL import ImageFilter
from torch import Tensor
from torchvision import transforms

from database import request_data_by_id
import autoencoder as ae
import autoencoder_v2 as ae2
from projet import utils


def flatten_img(img_path: str | list[str]) -> Tensor:
    """Uses the path stored in img_path to create a Tensor in a smaller
    dimension for the future mutations and crossing-overs.
    This function also encodes the image retrieved from the given path.

    Parameters
    ----------
    img_path: str or list[str]
        Path or paths of the images to retrieve

    Returns
    -------
    flat_img: torch.Tensor
        Contains the values for all the pixels of the images found in the
        given path. If several paths are given, all the images are stored
        in one single Tensor

    >>>path = "./projet/env/Database/img_dataset/celeba/img_align_celeba/000021.jpg"
    >>>Image.open(path).size
    >>>flatten_img(path).size()
    torch.Size([64, 324])

    >>>path_list = ['./projet/env/Database/img_dataset/celeba/img_align_celeba/000001.jpg',
    >>>             './projet/env/Database/img_dataset/celeba/img_align_celeba/000002.jpg',
    >>>             './projet/env/Database/img_dataset/celeba/img_align_celeba/000003.jpg']
    >>>flatten_img(path_list).size()
    torch.Size([3, 64, 324])
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

        for i, path in enumerate(img_path):
            if type(path) is str:
                img = Image.open(path)  # PIL Image
                img_tensor: Tensor = to_tensor(img)  # Image -> Tensor

                # Cropping the tensor to 160x160 pixels
                img_tensor_crop = ae2.crop_image_tensor(img_tensor)

                encoded_tensor = ae2.encode(autoencoder, img_tensor_crop)

                # print(f"Image tensor shape: {img_tensor.shape}")
                flat_img_tensor[i]: Tensor = flatten(encoded_tensor)

            else:
                raise TypeError("List should contain paths (str)")

        return flat_img_tensor

    elif type(img_path) is str:
        img = Image.open(img_path)  # PIL Image
        img_tensor = to_tensor(img)  # Image -> Tensor

        # Cropping the tensor to 160x160 pixels
        img_tensor_crop = ae2.crop_image_tensor(img_tensor)

        encoded_tensor: Tensor = ae2.encode(autoencoder, img_tensor_crop)
        flat_tensor: Tensor = flatten(encoded_tensor)

        return flat_tensor

    else:
        raise TypeError("Input should either be a path (str)\
            or a list of paths")


def deflatten_img(flat_tensor: Tensor, base_encoded_dim: torch.Size)\
        -> Image.Image | list[Image.Image]:
    """Resize a Tensor containing one or several encoded images into the
    specified size then decode it using the trained model.

    Parameters
    ----------
    flat_tensor: torch.Tensor
        Tensor containing the encoded version of one or several images.
    base_encoded_dim: torch.Size
        Size of the tensor before going into flatten_img function.

    Returns
    -------
    decoded_img: PIL.Image
        Decoded version of the image
    or
    img_list: list[PIL.Image]
        Decoded versions of the images

    >>>tensor = torch.randn((64, 324))
    >>>deflatten_img(tensor, torch.Size((64, 18, 18))).size
    (160, 160)
    """
    # Path where to find a trained autoencoder
    model_path = os.path.join(utils.get_path(env_path, "Encoder"), "model.pt")
    autoencoder = ae.load_model(model_path)  # Loading the trained autoencoder

    if flat_tensor.dim() == 2:
        # Reform the original encoded tensor
        unflat_tensor = flat_tensor.reshape(base_encoded_dim)

        decoded_img = ae.decode(autoencoder, unflat_tensor)

        return decoded_img

    elif flat_tensor.dim() == 3:
        # Creating a list for the images
        img_list = [0] * flat_tensor.size()[0]

        for i, flat_img in enumerate(flat_tensor):
            # Reform the original encoded tensor
            unflat_img = flat_img.reshape(base_encoded_dim)

            decoded_img = ae.decode(autoencoder, unflat_img)

            # Enhancing the image
            filter = ImageFilter.SMOOTH  # SHARPEN, SMOOTH, EDGE_ENHANCE are quite good
            enhanced_img = decoded_img.filter(filter)

            img_list[i] = enhanced_img

        return img_list

    else:
        raise Exception(f"Wrong tensor dimension. Expected: 2 or 3, got\
                        {flat_tensor.dim()}")


def mutate_img(tensor_encoded: Tensor, mutation_rate: float = 0.05,
               noise: float = 1, mode: str = 'add',
               scale: str = 'partial') -> Tensor:
    """Slightly modifies a or several images given in a Tensor.
    The modifications can be classic random noise or specific pixel
    reconstruction according to the tensor mean and standard deviation.

    Parameters
    ----------
    tensor_encoded: torch.Tensor
        Tensor containing one or several images pixels values.
    mutation_rate: float
        Probability for a pixel to be modified.
    noise: float
        Strength of the random noise, coefficient multiplying the noise.
    scale: str
        Either 'partial' or 'total'. If 'total' than every pixel is
        perturbed with a Gaussian random noise. If 'partial' than the
        modified pixels are randomly chosen according to mutation_rate.
        Default to 'partial'.
    mode: str
        Either 'add' or 'reconstruct'.
        Specifies the type of modifications to perform. If 'add' than
        the noise is added to chosen tensor values. The noise is a
        random number drawn from a gaussian distribution with mean 0
        and a standard deviation of 1. If 'reconstruct' than the tensor
        is rebuilt

    Returns
    -------
    img_mut: torch.Tensor
        Image or images built on img_encoded with Gaussian random noise
        added to it.

    >>>a = torch.randn((3, 3))
    tensor([[-0.2015,  0.4780,  1.1307],
        [ 1.6018,  0.4684,  0.2649],
        [ 1.0546,  1.3608,  0.6707]])
    >>>mutate_img(a, mutation_rate=0.3, mode='add', scale='partial')
    tensor([[-0.6087,  0.4780,  0.3646],
        [ 1.6018, -0.0372,  0.2649],
        [ 1.0546,  1.3608,  0.6707]])
    >>>mutate_img(a, mode='add', scale='total')
    tensor([[-3.3558,  1.5579, -0.2904],
        [-0.2572, -0.7410, -0.8748],
        [ 1.8747, -0.3136,  0.4488]])
    >>>b = torch.zeros((3, 3)) + 2
    tensor([[ 3.5582,  4.3696,  1.3062],
        [-0.6573,  3.4122,  1.5021],
        [ 5.4977,  2.1273,  2.0095]])
    >>>mutate_img(a, mode='reconstruct', scale='partial')
    tensor([[ 4.7168,  4.3696,  1.3062],
        [-0.6573,  3.4122,  1.5021],
        [ 6.6994,  1.3877, -0.0979]])
    >>>mutate_img(a, mode='reconstruct', scale='total')
    tensor([[ 2.6608,  4.7125,  5.6431],
        [ 1.2003,  0.7265, -0.4394],
        [ 5.1246,  0.6733,  0.6792]])
    """
    if type(tensor_encoded) is Tensor:
        if tensor_encoded.dim() == 2:
            img_mut = tensor_encoded
            if mode == 'add':
                # Add random noise to random pixels
                if scale == 'partial':
                    # Randomly selects the pixels to be modified
                    mut_proba_tensor = torch.rand(size=tensor_encoded.size())
                    noise_tensor = noise * torch.randn(size=tensor_encoded.size())
                    img_mut[mut_proba_tensor < mutation_rate] += noise_tensor[mut_proba_tensor < mutation_rate]

                # Add random noise on each pixel
                elif scale == 'total':
                    # Adding white noise to all the tensor values
                    img_mut += noise * torch.randn(size=tensor_encoded.size())

                else:
                    raise ValueError(f"Wrong value for the scale parameter. \
                    Expected 'partial' or 'total' got {scale} instead")

            # Building a new tensor based on the mean and deviation
            # of the input tensor
            elif mode == 'reconstruct':
                mu = tensor_encoded.mean()
                std = tensor_encoded.std()

                if scale == 'partial':
                    # Randomly selects the pixels to be modified
                    mut_proba_tensor = torch.rand(size=tensor_encoded.size())
                    # Size of the selected region
                    selected_size = img_mut[mut_proba_tensor < mutation_rate].size()
                    # Reconstruction of the selected region
                    img_mut[mut_proba_tensor < mutation_rate] = mu \
                        + torch.randn(selected_size) * std

                elif scale == 'total':
                    # Reconstruction of the whole tensor
                    img_mut = mu + torch.randn(tensor_encoded.size()) * std

                else:
                    raise ValueError(f"Wrong value for the scale parameter. \
                    Expected 'partial' or 'total' got {scale} instead")

            else:
                raise ValueError(f"Wrong value for the modif parameter. \
                Expected 'add' or 'reconstruct' got {mode} instead")

            return img_mut

        elif tensor_encoded.dim() == 3:
            global_tensor = torch.zeros(tensor_encoded.size())
            for i, tensor in enumerate(tensor_encoded):
                img_mut = tensor
                # Adding gaussian noise to the tensors
                if mode == 'add':
                    # Act on random values
                    if scale == 'partial':
                        # Randomly selects the pixels to be modified
                        mut_proba_tensor = torch.rand(size=tensor.size())
                        # Creating the gaussian noise
                        noise_tensor = noise * torch.randn(size=tensor.size()) + img_mut

                        img_mut = torch.where(mut_proba_tensor < mutation_rate,
                                              noise_tensor, img_mut)
                    # Act on every value
                    elif scale == 'total':
                        # Adding white noise to all the tensor values
                        img_mut = img_mut + noise * torch.randn(size=tensor.size())

                    else:
                        raise ValueError(f"Wrong value for the scale parameter. \
                        Expected 'partial' or 'total' got {scale} instead")

                # Building a new tensor based on the mean and deviation
                # of the input tensor
                elif mode == 'reconstruct':
                    mu = tensor.mean()
                    std = tensor.std()

                    if scale == 'partial':
                        # Randomly selects the pixels to be modified
                        mut_proba_tensor = torch.rand(size=tensor.size())
                        # Size of the selected region
                        selected_size = img_mut[mut_proba_tensor < mutation_rate].size()
                        # Reconstruction of the selected region
                        img_mut[mut_proba_tensor < mutation_rate] = mu \
                            + torch.randn(selected_size) * std

                    elif scale == 'total':
                        # Reconstruction of the whole tensor
                        img_mut = mu + torch.randn(tensor.size()) * std

                else:
                    raise ValueError(f"Wrong value for the modif parameter. \
                    Expected 'add' or 'reconstruct' got {mode} instead")

                global_tensor[i] = img_mut

            return global_tensor

        else:
            raise TypeError(f"Wrong Tensor dimension, expected 2 or 3, \
                            having {tensor_encoded.dim()}")

    else:
        raise TypeError(f"Input should be of type or torch.Tensor \
                        and not a {type(tensor_encoded)}")


def crossing_over(tensor_encoded: Tensor, crossing_rate: float) -> Tensor:
    """Swaps pixels between the given input images. Swaps are made
    randomly for each pixels.

    Parameters
    ----------
    tensor_encoded: torch.Tensor
        Tensor containing several images pixels values. The image
        where the pixel are drawn is chosen randomly between all the
        input images, with a uniform distribution
    crossing_rate: float
        Probability for a pixel to be swapped between images

    Returns
    -------
    new_tensor: torch.Tensor
        Image or images on which the crossing-overs were performed

    >>>a = torch.randn((2, 3, 3))
    tensor([[[-0.1689, -1.0904, -0.6837],
         [ 0.7088,  1.5056, -1.0236],
         [ 0.9674, -1.3691,  0.3151]],
        [[-1.3492,  1.3077, -0.3023],
         [-0.2509,  0.4455, -1.4012],
         [ 0.2429, -0.4856, -1.4789]]])
    >>>crossing_over(a, crossing_rate=0.3)
    tensor([[[-1.3492, -1.0904, -0.6837],
         [ 0.7088,  1.5056, -1.4012],
         [ 0.9674, -1.3691,  0.3151]],

        [[-1.3492,  1.3077, -0.3023],
         [-0.2509,  1.5056, -1.4012],
         [ 0.9674, -1.3691,  0.3151]]])
    """
    if type(tensor_encoded) is Tensor:
        global_tensor = torch.zeros(tensor_encoded.size())
        for i, tensor in enumerate(tensor_encoded):
            crossing_tensor = torch.rand(size=tensor.size())

            # Randomly choosing which image to swap pixels with
            other_ind = [k for k in range(tensor_encoded.size()[0]) if k != i]
            chosen_ind = np.random.choice(other_ind)

            chosen_tensor = tensor_encoded[chosen_ind]
            # Swapping pixels between tensors
            new_tensor = torch.where(crossing_tensor < crossing_rate,
                                     chosen_tensor, tensor)

            global_tensor[i] = new_tensor

        return global_tensor

    else:
        raise TypeError(f"Input should be of type or torch.Tensor \
                        and not a {type(tensor_encoded)}")


if __name__ == "__main__":
    env_path = "./projet"

    id_nb = 20
    # Path of the 20th image
    pic_path = request_data_by_id(env_path, id_nb)
    # print(f"Path for the picture(s): {pic_path}")

    # Path of the 3 images
    id_array = np.arange(start=3, stop=6, step=1)
    pic_path_list = request_data_by_id(env_path, id_array)
    # print(f"Path list: {pic_path_list}")

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
    print(f"Dim of the tensor: {flat_several.dim()}")

    # Testing deflatten
    deflat_img = deflatten_img(flat_encoded_tensor, encoded_img.size())
    # deflat_img.show()

    # Testing deflatten on several images
    deflat_several = deflatten_img(flat_several, encoded_img.size())
    for img in deflat_several:
        img.show()

    # Testing mutations on one flat encoded tensor
    # Adding white noise on every pixel
    # mut_img = mutate_img(flat_encoded_tensor, mutation_rate=0.01, noise=0.5, scale='total', mode='add')
    # deflat_img = deflatten_img(mut_img, encoded_img.size())
    # deflat_img.show()
    # mut_img = mutate_img(flat_encoded_tensor, mutation_rate=0.7, noise=0.5, scale='total', mode='add')
    # deflat_img = deflatten_img(mut_img, encoded_img.size())
    # deflat_img.show()

    # Adding white noise on some pixel
    # mut_img = mutate_img(flat_encoded_tensor, mutation_rate=0.01, noise=0.8, scale='partial', mode='add')
    # deflat_img = deflatten_img(mut_img, encoded_img.size())
    # deflat_img.show()
    # mut_img = mutate_img(flat_encoded_tensor, mutation_rate=0.5, noise=0.8, scale='partial', mode='add')
    # deflat_img = deflatten_img(mut_img, encoded_img.size())
    # deflat_img.show()

    # Reconstructing tensor
    # Totally
    # mut_img = mutate_img(flat_encoded_tensor, mode='reconstruct', scale='total')
    # deflat_img = deflatten_img(mut_img, encoded_img.size())
    # deflat_img.show()

    # Partially
    # mut_img = mutate_img(flat_encoded_tensor, mode='reconstruct', scale='partial')
    # deflat_img = deflatten_img(mut_img, encoded_img.size())
    # deflat_img.show()

    # Testing mutations on several flat encoded tensors
    # Adding white noise on some pixel
    # mut_several = mutate_img(flat_several, mutation_rate=0.2, mode='add', scale='partial')
    # deflat_sev = deflatten_img(mut_several, encoded_img.size())
    # for img in deflat_sev:
    #     img.show()

    # Adding white noise on some pixel
    # mut_several = mutate_img(flat_several, mutation_rate=0.2, mode='add', scale='total')
    # deflat_sev = deflatten_img(mut_several, encoded_img.size())
    # for img in deflat_sev:
    #     img.show()

    # Reconstructing tensor
    # Totally
    # mut_several = mutate_img(flat_several, mode='reconstruct', scale='total')
    # deflat_sev = deflatten_img(mut_several, encoded_img.size())
    # for img in deflat_sev:
    #     img.show()

    # Partially
    # mut_several = mutate_img(flat_several, mutation_rate=0.2, mode='add', scale='total')
    # deflat_sev = deflatten_img(mut_several, encoded_img.size())
    # for img in deflat_sev:
    #     img.show()

    # Testing crossing-overs
    crossed_tensors = crossing_over(flat_several, crossing_rate=0.6)
    deflat_sev = deflatten_img(crossed_tensors, encoded_img.size())
    for img in deflat_sev:
        img.show()

