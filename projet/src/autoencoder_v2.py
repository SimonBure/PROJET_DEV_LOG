import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchsummary import summary
from torchvision import datasets, transforms
from torch.utils.data import TensorDataset, Dataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import utils
import os
from PIL import Image
import create_db
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import time                      # Allow to compute runtime
import math                      # Mathematical functions


#env_path = os.path.dirname(os.path.realpath(__file__))
env_path = "projet"


#################### Create a custom dataset class #########
class MyDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        return sample
################## Load Database ###########################
def load_dataset(input_path, width, height, nb_samples=-1, crop_images=False):
    # define crop parameters
    top = 40
    left = 18
    crop_height = 160
    crop_width = 160

    # Get a list of all image files in the input directory
    file_list = os.listdir(input_path)

    # Save the total number of images in the folder
    total_nb_images = len(file_list)

    # If nb_samples is provided, limit the number of images loaded to nb_samples
    if nb_samples != -1:
        file_list = file_list[:nb_samples]
        total_nb_images = nb_samples

    # Create a numpy array to hold the image data
    samples = np.zeros((len(file_list), height, width, 3), dtype=np.uint8)

    # Loop through the list of image files and load each image into the samples array
    for i, filename in enumerate(file_list):
        # Print a progress indicator every 10% of the way through the loop
        if i % (len(file_list)//10) == 0:
            print('-', end='', flush=True)
        # Load the image from disk
        im = Image.open(os.path.join(input_path, filename))
        # Resize the image to the desired dimensions
        im = im.resize((width, height))
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


CelebA= load_dataset(get_path(env_path, Img),178,218, nb_samples=1000, crop_images=True)
print("This is the shape of the tensors in CelebA: ", CelebA.samples[0].shape)

###################### Showing some images from the CelebA dataset constructed#############
print("This is are images obtained from tensors in the CelebA dataset:" )
def plot_5_images(dataset, width, height):

    samples = dataset.samples

    fig, axs = plt.subplots(1, 5, figsize=(20,20))

    # if the image are flatten, reshape them to 2D images
    if len(samples.shape) == 2:
        print(samples.shape)
        samples = samples.reshape(-1, width, height, 3)

    # Draw 5 image number randomly
    random_indices = shuffle(np.arange(samples.shape[0]))[:5]

    for i in range(5):
        axs[i].imshow(samples[random_indices[i]], cmap=None, interpolation='nearest', aspect='equal')
        axs[i].axis("off")
    plt.show()
################### Data spliting and Loading ##################
def split_train_valid_test_set(dataset, p_train, p_valid):
    samples = dataset.samples
    nb_samples = samples.shape[0]
    p_test = 1 - p_train - p_valid

    nb_test = (int) (nb_samples * (1-p_train))
    nb_valid = (int) ((nb_samples - nb_test) * p_valid)

    sample_train, sample_test = train_test_split(samples, test_size=nb_test)
    sample_train, sample_valid = train_test_split(sample_train, test_size=p_valid)

    print(f"training data : {sample_train.shape[0]}")
    print(f"validation data : {sample_valid.shape[0]}")
    print(f"test data: {sample_test.shape[0]}\n")

    train_ds = TensorDataset(sample_train.permute(0, 3, 1, 2))
    valid_ds = TensorDataset(sample_valid.permute(0, 3, 1, 2))
    test_ds = TensorDataset(sample_test.permute(0, 3, 1, 2))

    return train_ds.tensors[0], valid_ds.tensors[0], test_ds.tensors[0]

p_train = 0.8
p_valid = 0.1

train_ds, valid_ds, test_ds = split_train_valid_test_set(CelebA, p_train, p_valid)
valid_ds[0].shape



my_batch_size = 10
# DataLoader d'entrainement
train_dl = DataLoader(train_ds, batch_size=my_batch_size, shuffle=True)
print(f"Training dataset contains : {len(train_ds)} images")
print(f"Training dataloader contains : {len(train_dl)} batchs each containing {my_batch_size} images \n")

# DataLoader de validation
valid_dl = DataLoader(valid_ds, batch_size=my_batch_size)
print(f"Validation dataset contains: {len(valid_ds)} images")
print(f"Validation dataloader contains : {len(valid_dl)} batchs each containing {my_batch_size} images")


for batch_idx, batch in enumerate(train_dl):
    print(f"Batch {batch_idx} shape: {batch.shape}")
    break # Only print the first batch

################################################################
"""
faces_db = create_db.request_data_by_id(env_path, range(1000))

faces = []
for i in faces_db:
    img =  Image.open(i)
    #img_array = np.array(img)
    new_size = (64, 64)
    faces.append(np.array(img.resize(new_size, resample=Image.Resampling.BILINEAR)))

print(faces[0].shape)
#print(faces[0])
#numpy_array = numpy_array.transpose(2, 0, 1)
img = Image.fromarray(faces[0])
#image = faces[0]
#resized_image = image.resize(new_size, resample=Image.Resampling.BILINEAR)
img.show()
#resized_image.show()
"""


####################### Auto-encoder #####

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
##Architecture

width = 160
height = 160
nb_chan_out = 64
AutoEncoder = Autoencoder()#width, height, nb_chan_out)
print("AutoEncoder model:")
print(summary(AutoEncoder, (3, width , height)))

####################################################

def encode(model, img):
    """
    Encodes an input image using the given PyTorch model.
    Parameters:
        model (nn.Module): Neural networt model.
        img (PIL.Image): PIL Image object representing the input image.

    Returns:
        torch.Tensor: Tensor representing the encoded representation of the input image.
    """
    transform = transforms.ToTensor()
    timg = transform(img)
    recon = model.encoder(timg)
    return recon


def overfitting(x_train, x_test, num_epoch):
    list_train, list_test = [], []
    model = Autoencoder()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=1e-3, weight_decay=1e-5)
    for epoch in range(num_epoch):
        for img in x_train:
            timg = transform(img)
            recon = model(timg)
            loss_train = criterion(recon, timg)
            optimizer.zero_grad()
            loss_train.backward()
            optimizer.step()
        list_train.append(loss_train.item())
        for img in x_test:
            timg = transform(img)
            recon = model(timg)
            loss_test = criterion(recon, timg)
            optimizer.zero_grad()
            loss_test.backward()
            optimizer.step()
        list_test.append(loss_test.item())
    #print(list_train, list_test)
    plt.plot(range(num_epoch), list_train, color="red")
    plt.plot(range(num_epoch), list_test, color="blue")
    plt.savefig(path+"/overfit.png")

################# Training the autoencoder
def train_autoencoder(autoencoder, train_dl, nb_epochs, learning_rate):
    # Define the loss function and optimizer
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(autoencoder.parameters(), lr=learning_rate)

    # Train the autoencoder for the specified number of epochs
    train_losses = []
    for epoch in range(nb_epochs):
        epoch_loss = 0
        for x in train_dl:
          x_hat = autoencoder(x)
          #print(x_hat.shape)
          loss = loss_fn(x_hat, x)
          loss.backward()
          optimizer.step()
          optimizer.zero_grad()
          epoch_loss += loss.item()

        train_losses.append(epoch_loss / len(train_dl))
        print(f"Epoch {epoch + 1}/{nb_epochs}, loss={train_losses[-1]:.5f}")

    return train_losses
train_losses = train_autoencoder(AutoEncoder, train_dl, nb_epochs=20, learning_rate=0.01)
#####################
##################### Recap of the Tensor sizes

print("Database Tensor: ",CelebA[0].shape)
aTensor = CelebA[0]
x = aTensor.unsqueeze(0)
x = x.permute(0, 3, 1, 2)
print("Permuted Database Tensor: ",x.shape)
recon = AutoEncoder.encoder(x)
print("Encoded Tensor Shape: ",recon.shape)
decoded = AutoEncoder.decoder(recon)
print("Decoded Tensor : ",decoded.shape)
decoded2 = decoded.squeeze(0)
print("Decoded Tensor with the batch dimension erased: ", decoded2)
########################
def save_model(model, name_file):
    """
    Saves the state dictionary of a PyTorch model to a file with the given name.

    Parameters:
        model (torch.nn.Module): PyTorch model to be saved.
        name_file (str): Filename to save the model to.
    """
    torch.save(model.state_dict(), name_file)

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


def save_fig(im, path, img_type):
    """
    Saves a matplotlib image object to a file in the specified path.

    Parameters:
        im (matplotlib.image.AxesImage): The image to be saved.
        path (str): The path to save the image to.
    """
    plt.imshow(im)
    plt.savefig(os.path.join(path, img_type))

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

#print(type(model))
path = utils.get_path(env_path, "Encoder")
#model = Autoencoder()
#summary(model, (3,218,178))
def train(model, dataloader, criterion, optimizer, num_epochs):
    model.train()
    for epoch in range(num_epochs):
        for batch in dataloader:
            images = batch["image"]
            recon = model(images)
            loss = criterion(recon, images)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch}: Loss = {loss:.4f}")

num_epochs = 10
model = Autoencoder()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
train(model, CelebA_dataloader, criterion, optimizer, num_epochs)
#enco_im = encode(model, faces[0])
#revtrans(enco_im).show
summary(model, (3,64,64))
