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

# define the args
ap = argparse.ArgumentParser()
ap.add_argument("data_dir", help="Directory of the dataset", default="data", nargs="?")

arch_choices = ("vgg16", "vgg13", "densenet121")

ap.add_argument("--arch", help="The architecture of the model", choices=arch_choices, default=arch_choices[0])

ap.add_argument("--hidden_units", help="Number of hidden units in the model", default=4096, type=int)

ap.add_argument("--learning_rate", help="Learning rate of the model", default=0.001, type=float)

ap.add_argument("--epochs", help="Number of training runs over the dataset", default=3, type=int)

ap.add_argument("--gpu", help="Use GPU or CPU for training", action="store_true")

ap.add_argument("--save_dir", help="Which directory to store model checkpoint", default="models")

args = vars(ap.parse_args())

# make the directory
os.system("mkdir -p " + args["save_dir"])

# get data_folder
data_folder = args["data_dir"]

train_dir = os.path.join(data_folder, "train")
valid_dir = os.path.join(data_folder, "valid")

# Define the transforms for the training, validation, and testing sets
train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])

valid_transforms = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])

# Load the datasets with ImageFolder
train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
valid_data = datasets.ImageFolder(valid_dir, transform=valid_transforms)

# Using the image datasets and the trainforms, define the dataloaders
trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
validloader = torch.utils.data.DataLoader(valid_data, batch_size=64)

# set process power source
device = None
if args["gpu"]:
    device = "cuda"
else:
    device = "cpu"

# making the model from the pretrained model
input_features = None
hidden_units = args['hidden_units']
    
my_local = dict()
exec("model = models.{}(pretrained=True)".format(args['arch']), globals(), my_local)
model =  my_local['model']
last_child = list(model.children())[-1]

if type(last_child) == torch.nn.modules.linear.Linear:
    input_features = last_child.in_features
elif type(last_child) == torch.nn.modules.container.Sequential:
    input_features = last_child[0].in_features

for param in model.parameters():
    param.requires_grad = False

classifier = nn.Sequential(OrderedDict([
    ('fc1', nn.Linear(input_features, hidden_units)),
    ('relu', nn.ReLU()),
    ('dropout', nn.Dropout(p=0.5)),
    ('fc2', nn.Linear(hidden_units, 102)),
    ('output', nn.LogSoftmax(dim=1))]))

model.classifier = classifier

criterion = nn.NLLLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=args["learning_rate"])
model.to(device)

# training the model
model.train()
epochs = args['epochs']
cur_epoch = 0
steps = 0
running_loss = 0
print_every = 5
for epoch in range(epochs):
    cur_epoch += 1
    for inputs, labels in trainloader:
        steps += 1
        # Move input and label tensors to the default device
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        logps = model.forward(inputs)
        loss = criterion(logps, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        
        if steps % print_every == 0:
            test_loss = 0
            accuracy = 0
            model.eval()
            with torch.no_grad():
                for inputs, labels in validloader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    logps = model.forward(inputs)
                    batch_loss = criterion(logps, labels)
                    
                    test_loss += batch_loss.item()
                    
                    # Calculate accuracy
                    ps = torch.exp(logps)
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                    
            print(f"Epoch {epoch+1}/{epochs}.. "
                  f"Train loss: {running_loss/print_every:.3f}.. "
                  f"Test loss: {test_loss/len(validloader):.3f}.. "
                  f"Test accuracy: {accuracy/len(validloader):.3f}")
            running_loss = 0
            model.train()

            
# Save the checkpoint 
model.class_idx_mapping = train_data.class_to_idx
save_dir = args['save_dir']

checkpoint = {'input_size': 25088,
              'output_size': 102,
              'classifier': model.classifier,
              'state_dict': model.state_dict(),
              'class_to_idx': train_data.class_to_idx,
              'epoch': cur_epoch,
              'arch': args['arch'],
              'optimizer': optimizer.state_dict()}

torch.save(checkpoint, os.path.join(save_dir, 'checkpoint.pth'))      






