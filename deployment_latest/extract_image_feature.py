import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.autograd import Variable
from PIL import Image
import cv2


def get_vector(image_name):
    model = models.resnet18(pretrained=True)

    # Use the model object to select the desired layer
    layer = model._modules.get('avgpool')

    # Set model to evaluation mode
    model.eval()

    # Image transforms
    scaler = transforms.Scale((224, 224))
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    to_tensor = transforms.ToTensor()
    # 1. Load the image with Pillow library
    #img = Image.open(image_name)
    img = cv2.imread(image_name)
    img = cv2.resize(img, (224,224))
    # 2. Create a PyTorch Variable with the transformed image
    try:
        t_img = Variable(normalize(to_tensor(img)).unsqueeze(0))
    except:
        print(image_name)
    # 3. Create a vector of zeros that will hold our feature vector
    #    The 'avgpool' layer has an output size of 512
    my_embedding = torch.zeros(512)
    a = 0
    # 4. Define a function that will copy the output of a layer
    def copy_data(m, i, o):
        #my_embedding.copy_(o.data)
        my_embedding.copy_(o.data.reshape(o.data.size(1)))
    # 5. Attach that function to our selected layer
    h = layer.register_forward_hook(copy_data)
    # 6. Run the model on our transformed image
    model(t_img)
    # 7. Detach our copy function from the layer
    h.remove()
    # 8. Return the feature vector
    return my_embedding.numpy()

def get_vector_api(model, img):

    layer = model._modules.get('avgpool')
    model.eval()

    scaler = transforms.Resize((224, 224))
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    to_tensor = transforms.ToTensor()

    img = cv2.resize(img, (224,224))
    t_img = Variable(normalize(to_tensor(img)).unsqueeze(0))

    my_embedding = torch.zeros(512)
    a = 0
    # 4. Define a function that will copy the output of a layer
    def copy_data(m, i, o):
        #my_embedding.copy_(o.data)
        my_embedding.copy_(o.data.reshape(o.data.size(1)))
    # 5. Attach that function to our selected layer
    h = layer.register_forward_hook(copy_data)
    # 6. Run the model on our transformed image
    model(t_img)
    # 7. Detach our copy function from the layer
    h.remove()
    # 8. Return the feature vector
    return my_embedding.numpy()


    


