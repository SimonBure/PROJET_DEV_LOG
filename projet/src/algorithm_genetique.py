import os
from PIL import Image
#from test import decode
from sklearn.datasets import fetch_olivetti_faces
import utils


#import a tensor from a given file

olivetti = fetch_olivetti_faces()
faces = olivetti.data
#print(len(faces))
# directory to save the images
path=utils.get_path("Encoder")



for i in range(0,5):
    image = Image.fromarray(faces[i].astype('uint8'), 'L')
    filename = f"image_{i}.jpg"
    file_path = os.path.join(path, filename)
    image.save(file_path)
