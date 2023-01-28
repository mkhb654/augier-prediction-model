from flask import Flask, send_from_directory, request, jsonify,render_template
app = Flask(__name__)
import json
import os
import requests
from retrieve_text2image import retrieve_text2image_api
import torch 
import clip
from extract_image_feature import get_vector_api
import torchvision.models as models
from retrieve_image2image import retrieve_image2image_api
import numpy as np
import io
from PIL import Image
import cv2

#text-image
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, preprocess = clip.load("ViT-B/32", device=device)
features_database_t2i = "text_2_image_features_database/"

#image-image
resnet_model = models.resnet18(pretrained=True)
features_database_i2i = "image_2_image_features_database/"


@app.route('/web_check', methods=['GET', 'POST'])
def web_check():
   query = request.args.get('query',"")
   print("Query -->", query)
   fetched_img_paths = retrieve_text2image_api(query, clip_model, device, features_database_t2i, 10)

   return fetched_img_paths


@app.route('/upload')
def setup_upload_file():
   return render_template('upload.html')


@app.route('/uploader', methods = ['GET', 'POST'])
def upload_file():
   if request.method == 'POST':
      f = request.files['file']
      image = Image.open(io.BytesIO(f.read()))
      img = np.array(image)

      query_feature = get_vector_api(resnet_model, img)

      fetched_img_paths  = retrieve_image2image_api(query_feature, features_database_i2i, 10)
      
      return render_template('uploader.html')

   if request.method == 'GET':
      return "Nothing here"


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)