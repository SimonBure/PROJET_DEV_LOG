import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_olivetti_faces
#import utils

path="."#utils.get_path("Encoder")

faces = fetch_olivetti_faces()
norm_faces = faces.data.astype('float32')/255
x_train, x_test = train_test_split(norm_faces, test_size=0.3, random_state=42)
#x_train = faces.images[:249]
#x_test = faces.images[250:]
transform=transforms.ToTensor()
revtrans=transforms.ToPILImage()

class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        # N, 1, 28, 28
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1), # -> N, 16, 14, 14
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), # -> N, 32, 7, 7
            nn.ReLU(),
            nn.Conv2d(32, 64, 7) # -> N, 64, 1, 1
        )

        # N , 64, 1, 1
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 7), # -> N, 32, 7, 7
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1), # N, 16, 14, 14 (N,16,13,13 without output_padding)
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, 3, stride=2, padding=1, output_padding=1), # N, 1, 28, 28  (N,1,27,27)
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

def encode(model, img):
    """
    Encodes an input image using the given PyTorch model.
    Parameters:
        model (nn.Module): Neural networt model.
        img (PIL.Image): PIL Image object representing the input image.

    Returns:
        torch.Tensor: Tensor representing the encoded representation of the input image.
    """
    transform=transforms.ToTensor()
    timg = transform(img)
    recon = model.encoder(timg)
    return recon

def decode(model, tensor):
    """
    Decodes a tensor representation of an image using the given PyTorch model.
    Parameters:
        model (nn.Module): Neural networt model.
        tensor (torch.Tensor): Tensor representing the encoded representation of an image.

    Returns:
        PIL.Image: PIL Image object representing the decoded image.
    """
    rev_transform=transforms.ToPILImage()
    decoded_tensor = model.decoder(tensor)
    img = rev_transform(decoded_tensor)
    return img


def overfitting(x_train, x_test, num_epoch):
    list_train, list_test = [], []
    model=Autoencoder()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    for epoch in range(num_epoch):
        for img in x_train:
            timg=transform(img)
            recon = model(timg)
            loss_train = criterion(recon, timg)
            optimizer.zero_grad()
            loss_train.backward()
            optimizer.step()
        list_train.append(loss_train.item())
        for img in x_test:
            timg=transform(img)
            recon = model(timg)
            loss_test = criterion(recon, timg)
            optimizer.zero_grad()
            loss_test.backward()
            optimizer.step()
        list_test.append(loss_test.item())
    #print(list_train, list_test)
    plt.plot(range(num_epoch),list_train,color="red")
    plt.plot(range(num_epoch),list_test,color="blue")
    plt.savefig(path+"/overfit.png")

def training(data, num_epochs):
    """
    Trains an Autoencoder model on the provided dataset for a specified number of epochs.
    Parameters:
        data (numpy.ndarray): Dataset containing the images to be used for training.
        num_epochs (int): The number of epochs to train the model for.
    Returns:
        Autoencoder: Trained Autoencoder model.
    """
    model = Autoencoder()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    for epoch in range(num_epochs):
        for img in data:
            timg=transform(img)
            recon = model(timg)
            loss = criterion(recon, timg)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f'Epoch:{epoch+1}, Loss:{loss.item():.4f}')
    return model

flag=0
if flag==0:

    model = training(norm_faces, 10)
    enco_im = encode(model, norm_faces[0])
    plt.imshow(norm_faces[0])
    plt.savefig(path+"/base_im.png")

    """
    en1=en.clone()
    en1[0]=en1[0]/10
    print(en1==en)
    de=model.decoder(en1)
    deco_im=revtrans(de)
    plt.imshow(deco_im)
    plt.savefig(path+"/modif_im.png")
    """

    deco_im=decode(model, enco_im)
    plt.imshow(deco_im)
    plt.savefig(path+"/recon_im.png")

if flag==1:
    overfitting(x_train, x_test, 30)

