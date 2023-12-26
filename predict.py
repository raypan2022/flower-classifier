# imports
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from collections import OrderedDict
from torchvision import datasets, transforms, models
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import json

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    # TODO: Process a PIL image for use in a PyTorch model
    
    # thumbnail to resize
    size = 0, 0
    width, height = image.size
    if width < height:
        size = 256, height
    else:
        size = width, 256
    image.thumbnail(size)
    
    # crop the center of the image
    crop_size = 224
    width, height = image.size
    left = (width - crop_size) / 2
    top = (height - crop_size) / 2
    right = (width + crop_size) / 2
    bottom = (height + crop_size) / 2
    image = image.crop((left, top, right, bottom))
    
    np_image = np.array(image)
    
    # normalize color channel
    np_image = np_image / 255.0
    
    means = np.array([0.485, 0.456, 0.406])
    stds = np.array([0.229, 0.224, 0.225])
    np_image = (np_image - means) / stds
    
    # transpose
    np_image = np_image.transpose((2, 0, 1))
    
    return np_image

# TODO: Write a function that loads a checkpoint and rebuilds the model
def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    
    my_local = dict()
    exec("model = models.{}(pretrained=True)".format(checkpoint['arch']), globals(), my_local)
    model =  my_local['model']
    
    model = models.vgg16(pretrained=True)
    
    for param in model.parameters():
        param.requires_grad = False

    model.classifier = checkpoint["classifier"]
    model.load_state_dict(checkpoint["state_dict"])
    class_to_idx = checkpoint["class_to_idx"]
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    
    return model, idx_to_class

def predict(image_path, checkpoint, topk=5, device="cuda"):
    
    model, idx_to_class = load_checkpoint(checkpoint)
    
    model.to(device)
    
    model.eval()
    
    image = Image.open(image_path)
    img = process_image(image)
    img = np.expand_dims(img, axis=0)
    img_tensor = torch.from_numpy(img).type(torch.FloatTensor).to(device)
    
    with torch.no_grad():
        log_probabilities = model.forward(img_tensor)
    
    probabilities = torch.exp(log_probabilities)
    probs, indices = probabilities.topk(topk)
    
    probs, indices = probs.to('cpu'), indices.to('cpu')
    probs = probs.numpy().squeeze()
    indices = indices.numpy().squeeze()
    classes = [idx_to_class[index] for index in indices]
    
    return probs, classes

# define args
ap = argparse.ArgumentParser()
ap.add_argument("image_path", help="Path to the image")

ap.add_argument("checkpoint_path", help="Path to checkpoint that stores the trained model")

ap.add_argument("--top_k", help="How many top K classes you want", default=5, type=int)

ap.add_argument("--category_names", help="File that contains mapping of category labels to category names", default="cat_to_name.json")

ap.add_argument("--gpu", help="Use GPU or CPU for training", action="store_true")

args = vars(ap.parse_args())

device = None

if args["gpu"]:
    device = "cuda"
else:
    device = "cpu"

probs, classes = predict(image_path=args["image_path"], checkpoint=args["checkpoint_path"], topk=args["top_k"], device=device)

with open(args["category_names"], 'r') as f:
    cat_to_name = json.load(f)
    class_names = [cat_to_name[c] for c in classes]

class_number = args['image_path'].split("/")[-2]
title = cat_to_name[str(class_number)]
print('The flower\'s name is:', title)
print('Class probabilities:', classes)
