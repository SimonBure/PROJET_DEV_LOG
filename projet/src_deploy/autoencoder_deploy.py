import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import os
from PIL import Image
import database
import torchvision.transforms.functional as TF


class Autoencoder(nn.Module):
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
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        return sample


def load_dataset(width, height, nb_samples=-1, crop_images=False):
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
            img = TF.crop(img, top, left, crop_height, crop_width)
            img = img.permute(1, 2, 0)
            cropped_samples[i] = img
        samples = cropped_samples

    # Remove the extra dimension from the tensor to make it a single tensor instead of a tuple
    samples = samples.squeeze()

    # Create a dataset object from the samples tensor
    dataset = MyDataset(samples)

    print(f"Number loaded images: {len(dataset)}/{total_nb_images}\n")

    return dataset

def load_model(name_file):
    """
    Loads the state dictionary of a PyTorch model from a file with the given name
    and returns the corresponding model object.

    Parameters:
        model (torch.nn.Module): PyTorch model to be loaded.
        name_file (str): Filename to load the model from.

    Returns:
        torch.nn.Module: The loaded PyTorch model.
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

def load_model(name_file):
    """
    Loads the state dictionary of a PyTorch model from a file with the given name
    and returns the corresponding model object.

    Parameters:
        model (torch.nn.Module): PyTorch model to be loaded.
        name_file (str): Filename to load the model from.

    Returns:
        torch.nn.Module: The loaded PyTorch model.
    """
    model = Autoencoder()
    model.load_state_dict(torch.load(name_file))
    return model

def encode(image,model):
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
    x = x.permute(0, 3, 1, 2)
    encoded = model.encoder(x)
    return encoded

def decode(tensor,model):
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
    return (decoded_shor)


def fin(algogen_path, interface_path):
    """
    Lets the interface know when the decoded images are saved in it's corresponding path
    Calls the decode function, takes the tensors created by the algogen from the algogen_path
    and saves them in the interface_path
    Parameters:
        algogen_path : path where the mutated tensors are stored
        interface_path : path to store the images from the decoded tensors
    Returns:
    True when the images are saved in the interface_path
    """
      try:
        # Load the mutated tensors created by the algogen
        tensors = utils.load_tensor(algogen_path)

        # Loop through the tensors and save each one as an image
        for i in range(tensors.shape[0]):
            # Get the i-th tensor
            tensor = tensors[i]
            decoded = decode(tensor, model)
            transform = T.ToPILImage()
            img = transform(decoded)
            img.save(os.path.join(interface_path, f'img{i}.jpg'))

        return True

    except Exception as e:
        print(f"An error occurred: {e}")
        return False
