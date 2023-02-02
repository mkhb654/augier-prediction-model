from flask import Flask, send_from_directory, request, jsonify,render_template
from flask_cors import CORS
app = Flask(__name__)

CORS(app)
CORS(app, resources={r"/*": {"origins":[ "*", "https://augier.art/**", "http://augier.art/**", "http://localhost:3000/**"] } } )

from retrieve_text2image import retrieve_text2image_api
import torch 
import clip
from extract_image_feature import get_vector_api
import torchvision.models as models
from retrieve_image2image import retrieve_image2image_api
import numpy as np
from PIL import Image


#text-image
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, preprocess = clip.load("ViT-B/32", device=device)
features_database_t2i = "text_2_image_features_database/"

#image-image
resnet_model = models.resnet18(pretrained=True)
features_database_i2i = "image_2_image_features_database/"


@app.route('/text_query', methods=['GET', 'POST'])
def web_check():
   query = request.args.get('query',"")
   print("Query -->", query)
   fetched_img_paths = retrieve_text2image_api(query, clip_model, device, features_database_t2i, 10)

   return fetched_img_paths

@app.route('/image_query', methods=['GET', 'POST'])
def web_check2():

   try:
      pil_img = Image.open(request.files["image"])
   except:
      return "invalid image"

   pil_img.convert("RGB")

   img = np.array(pil_img)

   if img.shape[2] > 3:
      img = img[:,:,:3]

   query_feature = get_vector_api(resnet_model, img)

   fetched_img_paths  = retrieve_image2image_api(query_feature, features_database_i2i, 10)
  
   return fetched_img_paths


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)