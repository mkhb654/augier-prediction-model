import numpy as np
import os
from sklearn.metrics.pairwise import cosine_similarity
import cv2
import operator
from extract_image_feature import get_vector



def retrieve_image(query_image, img_folder, database):
    feature_path = query_image.split(".")[0] + ".npy"
    query_feature = np.load(feature_folder + feature_path)
    query_feature = np.reshape(query_feature, (1, 512))

    similarities = {}
    for f in database:
        if f != feature_path:
            di = np.load(feature_folder + f)
            di = np.reshape(di, (1, 512))
            cs = cosine_similarity(di, query_feature, dense_output = False)
            similarities[f.split(".")[0] + ".jpg"] = cs
            print(f, " -- ", cs)


    #ranked = {k: v for k, v in sorted(similarities.items(), key=lambda item: item[1])}
    ranked = dict( sorted(similarities.items(), key=operator.itemgetter(1),reverse=True))

    for retrieved in ranked:
        img = cv2.imread(img_folder + retrieved)
        print(ranked[retrieved])
        print("*"*15)

        query_img = cv2.imread(img_folder + query_image)
        a= 0
        cv2.imshow("query", query_img)
        cv2.imshow("retrieved", img)
        cv2.waitKey(0)

def retrieve_image2image_api(query_image, features_database, top):
    feature_folder = features_database
    database = os.listdir(feature_folder)

    query_feature = np.reshape(query_image, (1, 512))

    similarities = {}
    for f in database:
        di = np.load(feature_folder + f)
        di = np.reshape(di, (1, 512))
        if di.all() != query_feature.all():
            cs = cosine_similarity(di, query_feature, dense_output = False)
            similarities[f.split(".")[0] + ".jpg"] = cs
            print(f, " -- ", cs)


    ranked = dict( sorted(similarities.items(), key=operator.itemgetter(1),reverse=True))


    img_folder = "digital_art/"
    to_send = []
    i = 0
    for retrieved in ranked:
        to_send.append(retrieved)
        i = i + 1
        if i == top:
            break
    
    return to_send

if __name__ == '__main__':
    img_folder = "digital_art/"
    feature_folder = "features_database/"

    imgs = os.listdir(img_folder)
    database = os.listdir(feature_folder)

    query_image = "Image_page_9_25.jpg"
    retrieve_image(query_image, img_folder, database)





