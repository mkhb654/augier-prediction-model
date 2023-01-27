import torch
import clip
from sklearn.metrics.pairwise import cosine_similarity
import os
from PIL import Image
import numpy as np
import operator
import cv2



def retrieve_image(query):

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)

    features_database = "features_database/"
    image_folder = "digital_art/"

    text = clip.tokenize([query]).to(device)
    text_features = model.encode_text(text).cpu().detach().numpy()

    similarities = {}

    for f in os.listdir(features_database):
        image_features = np.load(features_database + f)
        cs = cosine_similarity(image_features, text_features, dense_output = False)
        similarities[f.split(".")[0] + ".jpg"] = cs
        print(f, " -- ", cs)


    ranked = dict( sorted(similarities.items(), key=operator.itemgetter(1),reverse=True))

    for retrieved in ranked:
        img = cv2.imread(image_folder + retrieved)
        print(ranked[retrieved])
        print("*"*15)

        cv2.imshow("retrieved", img)
        cv2.waitKey(0)

def retrieve_text2image_api(query, model, device, features_database, top):
    
    text = clip.tokenize([query]).to(device)
    text_features = model.encode_text(text).cpu().detach().numpy()

    similarities = {}

    for f in os.listdir(features_database):
        image_features = np.load(features_database + f)
        cs = cosine_similarity(image_features, text_features, dense_output = False)
        similarities[f.split(".")[0] + ".jpg"] = cs
        #print(f, " -- ", cs)

    ranked = dict( sorted(similarities.items(), key=operator.itemgetter(1),reverse=True))
    to_send = []
    i = 0
    for retrieved in ranked:
        to_send.append(retrieved)
        i = i + 1
        if i == top:
            break
    
    return to_send



if __name__ == '__main__':
    #query = input("query text: ")
    query = "red"
    retrieve_image(query)



# logits_per_image, logits_per_text = model(image, text)
# probs = logits_per_image.softmax(dim=-1).cpu().numpy()
#print("Label probs:", probs)  # prints: [[0.9927937  0.00421068 0.00299572]]