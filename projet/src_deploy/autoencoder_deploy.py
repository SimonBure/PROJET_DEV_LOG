import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms
from torch.utils.data import Dataset
import os
from PIL import Image
import database
import torchvision.transforms as T
import utils


class Autoencoder(nn.Module):
    """
    A convolutional autoencoder neural network.

    The autoencoder has two main components: an encoder and a decoder. The encoder consists of a sequence of
    convolutional layers that encode the input image into a lower-dimensional latent space. The decoder takes the
    encoded representation and reconstructs the original image.

    Args:
        None

    Attributes:
        encoder (nn.Sequential): A sequence of convolutional layers that make up the encoder.
        decoder (nn.Sequential): A sequence of convolutional layers that make up the decoder.

    Methods:
        forward(x: torch.Tensor): Computes the forward pass of the autoencoder given an input tensor `x`. Returns the
        reconstructed output tensor.
    """
    def __init__(self):
        super(Autoencoder, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 7, stride=2, padding=1)
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 7, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 3, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


class MyDataset(Dataset):
    """
    A PyTorch dataset that represents a collection of samples.

    Args:
        samples (torch.Tensor): A 4D tensor of shape (batch_size, height, width, channels) representing the samples.

    Attributes:
        samples (torch.Tensor): A 4D tensor of shape (batch_size, height, width, channels) representing the samples.

    Methods:
        __len__(): Returns the number of samples in the dataset.
        __getitem__(idx: int): Returns the sample at the given index.

    """
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        return sample


def load_dataset(width, height, nb_samples=-1, crop_images=False):
    """
    Loads image data from a folder and returns a PyTorch dataset object.

    parameters:
        width (int): The desired width of the images in pixels.
        height (int): The desired height of the images in pixels.
        nb_samples (int, optional): The maximum number of images to load from the folder. Defaults to -1, which loads all images.
        crop_images (bool, optional): Whether to crop the images to a fixed size. Defaults to False.

    Returns:
        MyDataset: A PyTorch dataset object containing the loaded image data.

    Raises:
        IOError: If the folder path is invalid or cannot be accessed.

    """
    # define crop parameters
    top = 40
    left = 18
    crop_height = 160
    crop_width = 160

    # Get a list of all image files in the input directory
    file_list = database.request_data_by_id(env_path, range(nb_samples))

    # Save the total number of images in the folder
    total_nb_images = len(file_list)

    # Create a numpy array to hold the image data
    samples = np.zeros((len(file_list), height, width, 3), dtype=np.uint8)

    # Loop through the list of image files and load each image into the samples array
    for i, filename in enumerate(file_list):
        # Print a progress indicator every 10% of the way through the loop
        if i % (len(file_list)//10) == 0:
            print('-', end='', flush=True)
        # Load the image from disk
        im = Image.open(filename)
        # Convert the image data to a numpy array and add it to the samples array
        samples[i] = np.asarray(im)

    # Convert the numpy array to a PyTorch tensor, and normalize the pixel values to [0, 1]
    samples = torch.from_numpy(samples.astype(np.float32) / 255.0)

    # Crop the images if requested
    if crop_images:
        cropped_samples = torch.empty(samples.size(0), crop_height, crop_width, 3)
        for i, tensor in enumerate(samples):
            img = tensor
            img = img.permute(2, 0, 1)
            img = T.functional.crop(img, top, left, crop_height, crop_width)
            img = img.permute(1, 2, 0)
            cropped_samples[i] = img
        samples = cropped_samples

    # Remove the extra dimension from the tensor to make it a single tensor instead of a tuple
    samples = samples.squeeze()

    # Create a dataset object from the samples tensor
    dataset = MyDataset(samples)

    print(f"Number loaded images: {len(dataset)}/{total_nb_images}\n")

    return dataset


def crop_image_tensor(tensor):
    """
    Crops an image tensor to a fixed size.

    Parameters:
        tensor (torch.Tensor): A 3D tensor of shape (height, width, channels) representing an RGB image.

    Returns:
        torch.Tensor: A 3D tensor of shape (crop_height, crop_width, channels) representing the cropped image.

    """
    top = 40
    left = 18
    crop_height = 160
    crop_width = 160
    img = T.functional.crop(tensor, top, left, crop_height, crop_width)
    return img


def load_model(name_file):
    """
    Loads the state dictionary of a PyTorch model from a file with the given name
    and returns the corresponding model object.

    Parameters:
        name_file: str
            Filename to load the model from.

    Returns:
        model: torch.nn.Module
            The loaded PyTorch model.
    """
    model = Autoencoder()
    model.load_state_dict(torch.load(name_file))
    return model


def save_img(img: Image, path: str, format: str):
    """

    Parameters
    ----------
    img: PIL.Image
        Image object
    path: str
        Path where to store the image
    format: str
        Wanted format for the image
    """
    img.show()  # Display the image
    img.save(os.path.join(path, format))  # Save the image


def save_encoded_im(tensor, name_file):
    """
    Saves a PyTorch tensor to a file with the given name.

    Parameters:
        tensor (torch.Tensor): The tensor to be saved.
        name_file (str): The filename to save the tensor to.
    """
    torch.save(tensor, name_file)


def load_encoded_im(name_file):
    """
    Loads a PyTorch tensor from a file with the given name and returns the corresponding tensor object.

    Parameters:
        name_file (str): The filename to load the tensor from.

    Returns:
        tensor (torch.Tensor): The loaded PyTorch tensor.
    """
    tensor = torch.load(name_file)
    return tensor


def encode(model, image):
    """
    Encodes an input image using the given PyTorch model.
    Parameters:
        model (nn.Module): Neural networt model.
        image: torch.Tensor
            PIL Image object representing the input image.

    Returns:
        torch.Tensor: Tensor representing the encoded representation of the input image.
    """
    x = image.unsqueeze(0)  # Adding one dimension for the autoencoder
    # Changing the image to the correct dimensions order for the autoencoder
    # x = x.permute(0, 3, 2, 1)
    encoded = model.encoder(x)
    return encoded[0]


def decode(model, tensor):
    """
    Decodes an input tensor using the given PyTorch model.
    Parameters:
        model (nn.Module): Neural networt model.
        tensor: torch.Tensor to be decoded.

    Returns:
        decoded_shor: Tensor representing the decoded representation of the input tensor.
    """
    decoded = model.decoder(tensor)
    decoded_shor = decoded.squeeze(0)
    img_tf = T.ToPILImage()
    return img_tf(decoded_shor)
