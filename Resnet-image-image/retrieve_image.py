import numpy as np
import os
from sklearn.metrics.pairwise import cosine_similarity
import cv2
import operator

img_folder = "digital_art/"
feature_folder = "features_database/"

imgs = os.listdir(img_folder)
database = os.listdir(feature_folder)

query_image = "Image_page_9_25.jpg"
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