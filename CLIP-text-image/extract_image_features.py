import torch
import clip
from sklearn.metrics.pairwise import cosine_similarity

from PIL import Image
import os
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

img_folder = "digital_art/"

for img in os.listdir(img_folder):
    image = preprocess(Image.open(img_folder + img)).unsqueeze(0).to(device)
    print(img, image.shape)
    image_features = model.encode_image(image).cpu().detach().numpy()
    np.save("features_database/" + img.split(".")[0] + ".npy", image_features)


