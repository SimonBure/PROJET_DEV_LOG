import os.path

import numpy as np
import torch.nn as nn
import torch
from PIL import Image
from PIL import ImageFilter
from torch import Tensor
from torchvision import transforms

from database import request_data_by_id
from projet.src_deploy import autoencoder_deploy as ae
from projet import utils


def flatten_img(img_path, env_path):
    """Uses the path stored in img_path to create a Tensor in a smaller
    dimension for the future mutations and crossing-overs.
    This function also encodes the image retrieved from the given path.

    Parameters
    ----------
    img_path: list of str
        Path or paths of the images to retrieve
    env_path: str
        Path of the current environment, used to retrieve the
        autoencoder trained model

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
    model_path = os.path.join(utils.get_path(env_path, "Encoder"),
                              "model40k.pt")
    autoencoder = ae.load_model(model_path)  # Loading the trained autoencoder

    # To transform a PIL image to a torch Tensor
    to_tensor = transforms.ToTensor()
    # To flatten a torch tensor to a tensor with two dimensions only
    flatten = nn.Flatten(1, 2)

    if type(img_path) is list:
        temp_img = Image.open(img_path[0])  # Temporary img to get its size
        temp_tensor = to_tensor(temp_img)
        temp_tensor = ae.crop_image_tensor(temp_tensor)
        temp_tensor = ae.encode(autoencoder, temp_tensor)
        size = temp_tensor.shape

        # Global Tensor containing all the images
        flat_img_tensor = torch.zeros((len(img_path), size[0],
                                       size[1] * size[2]))

        for i, path in enumerate(img_path):
            if type(path) is str:
                img = Image.open(path)  # PIL Image
                img_tensor: Tensor = to_tensor(img)  # Image -> Tensor

                if img_tensor.size() == torch.Size((3, 218, 178)):
                    # Cropping the tensor to 160x160 pixels
                    img_tensor_crop = ae.crop_image_tensor(img_tensor)

                else:
                    img_tensor_crop = img_tensor

                # print(img_tensor_crop.size())
                encoded_tensor = ae.encode(autoencoder, img_tensor_crop)

                flat_img_tensor[i]: Tensor = flatten(encoded_tensor)

            else:
                raise TypeError("List should contain paths (str)")

        return flat_img_tensor

    elif type(img_path) is str:
        img = Image.open(img_path)  # PIL Image
        img_tensor = to_tensor(img)  # Image -> Tensor

        # Cropping the tensor to 160x160 pixels
        img_tensor_crop = ae.crop_image_tensor(img_tensor)

        encoded_tensor: Tensor = ae.encode(autoencoder, img_tensor_crop)
        flat_tensor: Tensor = flatten(encoded_tensor)

        return flat_tensor

    else:
        raise TypeError("Input should either be a path (str)\
            or a list of paths")


def deflatten_img(flat_tensor, base_encoded_dim, env_path):
    """Resize a Tensor containing one or several encoded images into the
    specified size then decode it using the trained model.

    Parameters
    ----------
    flat_tensor: torch.Tensor
        Tensor containing the encoded version of one or several images.
    base_encoded_dim: tuple of int
        Size of the tensor before going into flatten_img function.
    env_path: str
        Path of the current environment, used to retrieve the
        autoencoder trained model

    Returns
    -------
    decoded_img: PIL.Image.Image
        Decoded version of the image
    or
    img_list: list of PIL.Image.Image
        Decoded versions of the images

    >>>tensor = torch.randn((64, 324))
    >>>deflatten_img(tensor, torch.Size((64, 18, 18))).size
    (160, 160)
    """
    # Path where to find a trained autoencoder
    model_path = os.path.join(utils.get_path(env_path, "Encoder"), "model40k.pt")
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

            # Reducing the noise with a filter
            filter = ImageFilter.SMOOTH_MORE
            enhanced_img = decoded_img.filter(filter)
            filter = ImageFilter.SHARPEN
            enhanced_img = enhanced_img.filter(filter)
            img_list[i] = enhanced_img

        return img_list

    else:
        raise Exception(f"Wrong tensor dimension. Expected: 2 or 3, got\
                        {flat_tensor.dim()}")


def mutate_img(tensor_encoded, mutation_rate=0.05, mode='add', scale='partial'):
    """Slightly modifies a or several images given in a Tensor.
    The modifications can be classic random noise or specific pixel
    reconstruction according to the tensor mean and standard deviation.

    Parameters
    ----------
    tensor_encoded: Tensor
        Tensor containing one or several images pixels values.
    mutation_rate: float
        Probability for a pixel to be modified.
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
    """
    if type(tensor_encoded) is Tensor:
        if tensor_encoded.dim() == 2:
            img_mut = tensor_encoded
            if mode == 'add':
                # Add random noise to random pixels
                if scale == 'partial':
                    # Randomly selects the pixels to be modified
                    mut_proba_tensor = torch.rand(size=tensor_encoded.size())
                    noise_tensor = torch.randn(size=tensor_encoded.size())
                    img_mut[mut_proba_tensor < mutation_rate] += noise_tensor[mut_proba_tensor < mutation_rate]

                # Add random noise on each pixel
                elif scale == 'total':
                    # Adding white noise to all the tensor values
                    img_mut += torch.randn(size=tensor_encoded.size())

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
                        noise_tensor = torch.randn(size=tensor.size()) + img_mut

                        img_mut = torch.where(mut_proba_tensor < mutation_rate,
                                              noise_tensor, img_mut)
                    # Act on every value
                    elif scale == 'total':
                        # Adding white noise to all the tensor values
                        img_mut = img_mut + torch.randn(size=tensor.size())

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
                        # Reconstructed tensor
                        recons_tensor = mu + torch.randn(tensor.size()) * std
                        # Reconstruction of the selected region
                        img_mut = torch.where(mut_proba_tensor < mutation_rate,
                                              recons_tensor, img_mut)

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


def chose_closest_tensor(input_tensor, other_tensors):
    """Returns a sub-tensor inside other_tensors that is the closest, in
    terms of euclidian distance of its value, to input_tensor.

    Parameters
    ----------
    input_tensor: torch.Tensor
        Tensor object.
    other_tensors: torch.Tensor
        Bigger tensor containing two tensors of the same dimension as
        input_tensor.

    Returns
    -------
    closest_tensor: torch.Tensor
        Closest Tensor to input_tensor, same size as input_tensor

    >>>a = torch.randn((3, 3)) + 2
    tensor([[-1.1437, -0.4471,  0.8246],
        [-0.8996,  0.2804,  0.7916],
        [-0.0613,  0.1011, -1.1687]])
    >>>b = torch.randn(3, 3) + 3
    tensor([[3.3959, 2.4724, 2.4523],
        [3.1876, 3.6320, 4.1872],
        [3.4574, 2.3627, 3.7312]])
    >>>c = torch.randn(3, 3) + 7
    tensor([[6.9013, 6.2670, 6.6347],
        [7.2425, 6.0343, 6.7958],
        [7.7789, 8.6575, 8.1256]])
    >>>d = torch.cat([b.unsqueeze(0), c.unsqueeze(0)], 0)
    >>>chose_closest_tensor(a, d)
    tensor([[3.3959, 2.4724, 2.4523],
            [3.1876, 3.6320, 4.1872],
            [3.4574, 2.3627, 3.7312]])
    """
    # Computing euclidian distance between input and the other tensors
    dist_list = [torch.dist(input_tensor, t, p=2) for t in other_tensors]
    dist_tensor = Tensor(dist_list)
    # Choosing the tensor that has the closest distance to input_tensor
    closest_tensor = other_tensors[torch.argmin(dist_tensor)]
    return closest_tensor


def crossing_over(tensor_encoded, crossing_rate):
    """Swaps pixels between the given input images. Swaps are made
    randomly for each pixel but the choice of which images are to be
    swapped is made based on a mean distance criterion.

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

        crossing_tensor = torch.rand(size=tensor_encoded[0].size())
        fst_tensor = tensor_encoded[0]
        chosen_tensor = chose_closest_tensor(fst_tensor, tensor_encoded[1:])

        # List of bool, True if the tensor has been used before
        is_tensor_used = [torch.equal(fst_tensor, t) or torch.equal(chosen_tensor, t) for t in tensor_encoded]
        # Swapping the values between the tensors
        new_tensor = torch.where(crossing_tensor < crossing_rate, chosen_tensor, fst_tensor)
        chosen_tensor = torch.where(crossing_tensor < crossing_rate, fst_tensor, chosen_tensor)

        # Boolean tensor, True if the tensor has not been used yet
        bool_tensor = ~Tensor(is_tensor_used).bool()
        # Only tensor remaining to be swapped
        remaining_tensor = tensor_encoded[bool_tensor][0]
        last_cross_tens = torch.rand(size=remaining_tensor.size())
        last_chosen_tens = chose_closest_tensor(remaining_tensor,
                                                tensor_encoded[~bool_tensor])
        # Swapping the values for the remaining tensor
        final_tensor = torch.where(last_cross_tens < crossing_rate,
                                   last_chosen_tens, remaining_tensor)

        # Feeling the global tensor
        global_tensor[0] = new_tensor
        global_tensor[1] = chosen_tensor
        global_tensor[2] = final_tensor

        # for i, tensor in enumerate(tensor_encoded):
        #     crossing_tensor = torch.rand(size=tensor.size())
        #     # Swap with the closest image between the two others
        #     other_ind = [k for k in range(tensor_encoded.size()[0]) if k != i]
        #     chosen_tensor = chose_closest_tensor(tensor, tensor_encoded[other_ind])
        #
        #     equal_tensor_list = [torch.equal(chosen_tensor, t) for t in tensor_encoded]
        #     bool_tensor = Tensor(equal_tensor_list).bool()
        #     print(bool_tensor)
        #     tensor_encoded = tensor_encoded[bool_tensor]
        #     print(tensor_encoded.size())
        #     # Swapping pixels between tensors
        #     new_tensor = torch.where(crossing_tensor < crossing_rate,
        #                              chosen_tensor, tensor)
        #
        #     global_tensor[i] = new_tensor

        return global_tensor

    else:
        raise TypeError(f"Input should be of type or torch.Tensor \
                        and not a {type(tensor_encoded)}")


def remove_worst_tensor(input_tensor):
    """Returns a truncated version of input_tensor where the tensor with
    the biggest euclidian distances with the other is removed.

    Parameters
    ----------
    input_tensor: torch.Tensor
        Tensor object of dimension 3

    Returns
    -------
    best_tensors: torch.Tensor
        Tensors with the lowest euclidian distance between each other
    >>>a = torch.randn((3, 3)) + 2.3
    tensor([[3.0174, 2.3837, 2.4347],
        [3.2638, 4.1361, 1.5455],
        [2.2102, 4.6333, 1.7257]])
    >>>b = torch.randn((3, 3)) + 12.9
    tensor([[11.9757, 12.4747, 12.2593],
        [14.0547, 12.2040, 10.5324],
        [11.1283, 12.4093, 11.5780]])
    >>>c = torch.randn((3, 3)) + 5.43
    tensor([[6.1780, 5.0044, 5.2440],
        [4.8488, 5.7857, 6.1550],
        [5.9140, 5.0360, 3.1640]])
    >>>cat = torch.cat((a.unsqueeze(0), b.unsqueeze(0), c.unsqueeze(0)), 0)
    >>>remove_worst_tensor(cat)
    tensor([[[3.0174, 2.3837, 2.4347],
         [3.2638, 4.1361, 1.5455],
         [2.2102, 4.6333, 1.7257]],

        [[6.1780, 5.0044, 5.2440],
         [4.8488, 5.7857, 6.1550],
         [5.9140, 5.0360, 3.1640]]])
    """
    sum_dist_tensor = torch.zeros(input_tensor.size()[0])
    for i, tensor in enumerate(input_tensor):
        is_other_tensor = [False if torch.equal(tensor, t)
                           else True for t in input_tensor]
        other_tensor = input_tensor[Tensor(is_other_tensor).bool()]
        dist_to_other = torch.dist(tensor, other_tensor)
        sum_dist_tensor[i] = torch.sum(dist_to_other)

    # Create a list of bool containing False if the parsing index is the
    # one of the tensor with the greatest total distance to the others
    to_select = [False if i == torch.argmax(sum_dist_tensor)
                 else True for i in range(input_tensor.size()[0])]
    # Converting to boolean Tensor
    bool_tensor = Tensor(to_select).bool()
    # Selecting the tensors with the lowest standard deviation
    best_tensors = input_tensor[bool_tensor]
    return best_tensors


def create_new_images(img_path, env_path):
    """Generates 5 new images based on the 3 given with img_path.
    It uses the crossing_over function to generate 6 new images by
    swapping the values between the encoded tensors and then remove the
    worst one in terms of standard deviation

    Parameters
    ----------
    img_path: list of str
        Path where to retrieve the selected images. These images are
        used to create 5 new images.
    env_path: str
        Path of the current environment, used to save the generated
        images at the correct place

    Returns
    -------
    True to indicate to the interface that the image are generated
    """
    img_encoded_tensor = flatten_img(img_path, env_path)
    # Create 3 new images with crossing-overs
    crossed_img = crossing_over(img_encoded_tensor, crossing_rate=0.35)
    # Create 3 more images with crossing-overs
    more_crossing = crossing_over(img_encoded_tensor, crossing_rate=0.35)
    # Concatenation of the 6 tensors
    new_tensors = torch.cat((crossed_img, more_crossing), dim=0)
    # Keep the 5 best images in terms of deviation
    good_5_tensors = remove_worst_tensor(new_tensors)
    # Decoding and converting to PIL Images
    new_images = deflatten_img(good_5_tensors, (64, 18, 18), env_path)
    # Saving the generated pictures
    for i, img in enumerate(new_images):
        name = f"image{i}.png"
        path = utils.get_path(env_path, 'gen_img') + "/" + name
        img.save(path)

    return True


if __name__ == "__main__":
    env_path = "./projet"

    id_nb = 20
    # Path of the 20th image
    pic_path = request_data_by_id(env_path, id_nb, who='idkit')
    # print(f"Path for the picture(s): {pic_path}")

    # Path of the 3 images
    id_array = np.arange(start=3, stop=6, step=1)
    pic_path_list = request_data_by_id(env_path, id_array, who='idkit')
    # print(f"Path list: {pic_path_list}")

    # Open the image with PIL
    pic = Image.open(pic_path)

    # Converting the image to a Tensor
    tf_tensor = transforms.ToTensor()
    pic_tensor = tf_tensor(pic)

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
    # print("Partial mutation on images")
    # mut_several = mutate_img(flat_several, mutation_rate=0.1, mode='add', scale='partial')
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

    # Choosing 3 random images in the database
    random_id = np.random.randint(low=0, high=600, size=(3,))
    # print(random_id)
    random_img_path = request_data_by_id(env_path, random_id, who='idkit')
    print(random_img_path)
    random_img_tensor = flatten_img(random_img_path, env_path)

    # chose_closest_tensor(random_img_tensor[0], random_img_tensor[1:])

    # Showing the 3 selected images
    for path in random_img_path:
        img = Image.open(path)
        img.show()

    # Mutating the images
    mut_rand_tensor = mutate_img(random_img_tensor, mutation_rate=0.1,
                                 mode='reconstruct', scale='partial')

    # Testing crossing-overs
    # crossed_tensors = crossing_over(random_img_tensor, crossing_rate=0.3)
    # crossed_tensors_mut = crossing_over(mut_rand_tensor, crossing_rate=0.4)
    # deflat_sev = deflatten_img(crossed_tensors, encoded_img.size())
    # deflat_sev_mut = deflatten_img(crossed_tensors_mut, encoded_img.size())
    # for img in deflat_sev:
    #     img.show()


    # Testing to remove the worst tensor
    # good_tensors = remove_worst_tensor(crossed_tensors)
    # print(f"Size after removing the worst tensor: {good_tensors.size()}")

    # for img in deflat_sev_mut:
    #     img.show()

    create_new_images(random_img_path, env_path)

    # Regenerating based on already generated images
    # path_gen_img = utils.get_path(env_path, 'gen_img')
    # path_img = [path_gen_img + "/" + f"image{i}.png" for i in range(3)]
    # tensor_regen = flatten_img(path_img, env_path)
    # crossing_over(tensor_regen, crossing_rate=0.5)
    # img_regen = deflatten_img(tensor_regen, encoded_img.size(), env_path)
    # for img in img_regen:
    #     img.show()
