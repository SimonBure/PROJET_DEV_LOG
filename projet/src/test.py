import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_olivetti_faces
import utils

path=utils.get_path("Encoder")

faces = fetch_olivetti_faces()
x_train = faces.images[:299]
x_test = faces.images[300:]
transform=transforms.ToTensor()

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


# Note: nn.MaxPool2d -> use nn.MaxUnpool2d, or use different kernelsize, stride etc to compensate...
# Input [-1, +1] -> use nn.Tanh

model = Autoencoder()
#loss_fn = lambda x, y: 1 - F.ssim(x, y, data_range=1, size_average=True)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

def overfitting(x_train, x_test, num_epoch):
    list_train, list_test = [], []
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

overfitting(x_train, x_test, 20)

flag=False
if flag==True:
    num_epochs = 7
    #outputs = []
    for epoch in range(num_epochs):
        for img in faces.images:
            timg=transform(img)
            recon = model(timg)
            loss = criterion(recon, timg)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f'Epoch:{epoch+1}, Loss:{loss.item():.4f}')
        #outputs.append((epoch, img, recon))



    enco_im=transform(faces.images[0])
    revtrans=transforms.ToPILImage()
    im=revtrans(enco_im)
    plt.imshow(im)
    plt.savefig(path+"/base_im.png")
    en=model.encoder(enco_im)

    """
    en1=en.clone()
    en1[0]=en1[0]/10
    print(en1==en)
    de=model.decoder(en1)
    deco_im=revtrans(de)
    plt.imshow(deco_im)
    plt.savefig(path+"/modif_im.png")
    """

    de=model.decoder(en)
    deco_im=revtrans(de)
    plt.imshow(deco_im)
    plt.savefig(path+"/recon_im.png")
