# -*- coding: utf-8 -*-
#!usr/bin/env python3
'''
=================================================================>--Version:v1
=========================================================>Compiègne:20/01/2021
=======================================================>Fraud predictive model
================================================>done by @Manfo Satana Patrice
'''


import torch
import torch.nn as nn
import torchvision.transforms as transforms
import io
import os
from PIL import Image # we will need this to tranform our imge into tensor

class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.input_size = input_size
        self.l1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        # no activation and no softmax at the end
        return out
    # hyper parameters

input_size = 784
hidden_size = 500
num_classes = 10
model = NeuralNet(input_size, hidden_size, num_classes)
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PATH = os.path.join(ROOT_DIR, 'modeldir/model.pth')

# image --> tensor

def transform_image(image_bytes):
    # same as during training but in the same scale "gray" as for trainin
    transform = transforms.Compose([transforms.Grayscale(num_output_channels=1),
                                    transforms.Resize((28,28)),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.1307,),(0.3081,))])

    image = Image.open(io.BytesIO(image_bytes))
    return transform(image).unsqueeze(0)

def get_prediction(image_tensor):
    images = image_tensor.reshape(-1,28*28)
    outputs = model(images)
         #torch.max  return (value,index)
    _,predicted = torch.max(outputs.data,1)
    return predicted

if __name__ =="__main__":
    print(PATH)
    # "mnist_ffn.pth"
    model.load_state_dict(torch.load(PATH))
    model.eval()
