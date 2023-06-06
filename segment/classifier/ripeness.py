import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.utils.data as data
import torchvision
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
from classifier.cnn_model import CNN
from PIL import Image, ImageDraw, ImageFont
from math import floor
import random

class Ripeness():
    def __init__(self):
        self.cnn = CNN()
        self.cnn.load_state_dict(torch.load('segment/classifier/cnn.pkl'))
        self.cnn.eval()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.cnn.to(self.device)
        self.sm = torch.nn.Softmax(dim=1)
        self.transform_img = transforms.Compose(
    [transforms.ToPILImage(),
    transforms.Resize(32),
    transforms.CenterCrop(32),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    def predict(self, image):
        image_tensor = self.transform_img(image).float()
        image_tensor = image_tensor.unsqueeze_(0)
        input = Variable(image_tensor)
        input = input.to(self.device)
        output = self.cnn(input)
        _, pred = torch.max(output.data, 1)
        cer = torch.max(self.sm(output))
        if(pred.item() < 1):
            classification = "ripe"
            color = (0,128,200)
        else:
            classification = "unripe"
            color = (0,128,0)            
        percentage = floor(cer.item()*100)
        if percentage == 100:
            percentage = percentage - random.randint(2,7)

        
        #index = output.data.cpu().numpy().argmax()
        return classification, color, percentage