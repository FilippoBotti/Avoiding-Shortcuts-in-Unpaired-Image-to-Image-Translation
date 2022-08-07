
import torch.nn as nn
from PIL import ImageFont, ImageDraw, Image
import time
from options.train_options import TrainOptions
from data import create_dataset
from models import create_model
from util.visualizer import Visualizer
import argparse
import cv2
import numpy as np
import torch
from torch.autograd import Function
from torchvision import models
import torch.nn.functional as F
import os

class FeatureExtractor():
    """ Class for extracting activations and
    registering gradients from targetted intermediate layers """

    def __init__(self, model, target_layers):
        self.model = model
        self.target_layers = target_layers
        self.gradients = []

    def save_gradient(self, grad):
        #print(grad)
        self.gradients.append(grad)

    def __call__(self, x):
        outputs = []
        self.gradients = []
        for name, module in self.model._modules.items():
            #print("SECONDO CICLO ", name, "module", module)
            x = module(x)
            if name in self.target_layers:
                #print(name)
                x.register_hook(self.save_gradient)
                outputs += [x]
        #print("GRADIENTS" , self.gradients)
        return outputs, x


class ModelOutputs():
    """ Class for making a forward pass, and getting:
    1. The network output.
    2. Activations from intermeddiate targetted layers.
    3. Gradients from intermeddiate targetted layers. """

    def __init__(self, model, discriminator, feature_module, target_layers):
        self.model = model
        self.feature_module = feature_module
        self.discriminator = discriminator
        self.feature_extractor = FeatureExtractor(self.feature_module, target_layers)

    def get_gradients(self):
        return self.feature_extractor.gradients

    def __call__(self, x):
        target_activations = []
        for name, module in self.model._modules.items():
            #print("PRIMO CICLO ", name , "MODULE ", module)
            if module.model == self.feature_module:
                target_activations, x = self.feature_extractor(x)
            elif "avgpool" in name.lower():
                x = module(x)
                x = x.view(x.size(0),-1)
            else:
                x = module(x)
                x = self.discriminator(x)
        return target_activations, x


def preprocess_image(img):
    means = [0.485, 0.456, 0.406]
    stds = [0.229, 0.224, 0.225]

    preprocessed_img = img.copy()[:, :, ::-1]
    for i in range(3):
        preprocessed_img[:, :, i] = preprocessed_img[:, :, i] - means[i]
        preprocessed_img[:, :, i] = preprocessed_img[:, :, i] / stds[i]
    preprocessed_img = \
        np.ascontiguousarray(np.transpose(preprocessed_img, (2, 0, 1)))
    preprocessed_img = torch.from_numpy(preprocessed_img)
    preprocessed_img.unsqueeze_(0)
    input = preprocessed_img.requires_grad_(True)
    return input

def concat_tile(im_list_2d):
    return cv2.vconcat([cv2.hconcat(im_list_h) for im_list_h in im_list_2d])

def show_one_cam_on_image(img,mask):
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    cv2.imwrite("cam.jpg", np.uint8(255 * cam))

def show_cam_on_image(img, mask,nome):
    righe =[]
    colonne = []
    resized = np.uint8(img * 255)
    resized = cv2.copyMakeBorder(
                 resized, 
                 15,
                 15,
                 15,
                 15,
                 cv2.BORDER_CONSTANT, 
                 value=(255,255,255)
              )
    righe.append(resized)
    for key in mask:
        heatmap = cv2.applyColorMap(np.uint8(255 * mask[key]), cv2.COLORMAP_JET)
        heatmap = np.float32(heatmap) / 255
        heatmap = cv2.resize(heatmap, (224,224))
        cam = heatmap + np.float32(img)
        cam = cam / np.max(cam)
        immagine = np.uint8(255 * cam)
        immagine = cv2.copyMakeBorder(
                 immagine, 
                 15,
                 15,
                 15,
                 15,
                 cv2.BORDER_CONSTANT, 
                 value=(255,255,255)
              )
        immagine = cv2.putText(
            immagine, 
            key, 
            (50,250), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.4, 
            (209, 80, 0, 255), 
            0)
        righe.append(immagine)
        if(len(righe)==4):
            colonne.append(righe)
            righe = [] 

    im_tile = concat_tile(colonne)
    
    directory = '/content/drive/MyDrive/GradCamOrange/relucam'
  
     
    os.chdir(directory) 
    
    # List files and directories   
    
    print("Before saving image:")   
    print(os.listdir(directory)) 
    cv2.imwrite(nome,im_tile)


class GradCam:
    def __init__(self, model, discriminator, feature_module,  use_cuda):
        self.model = model
        self.discriminator = discriminator
        self.feature_module = feature_module
        self.model.eval()
        self.cuda = use_cuda
        if self.cuda:
            self.model = model.cuda()

    def forward(self, input):
        return self.model(input)

    def __call__(self, input, target_layer_names, index=None):
        #self.extractor veniva chiamato nell'init ma usava i layer, lo sposto quindi nel call
        self.extractor = ModelOutputs(self.model, self.discriminator , self.feature_module, target_layer_names)
        if self.cuda:
            features, output = self.extractor(input.cuda())
        else:
            features, output = self.extractor(input)

        if index == None:
            index = np.argmax(output.cpu().data.numpy())

        one_hot = torch.zeros((1, output.size()[-1]), dtype=torch.float32, requires_grad=True)
        
        #print(output.size())
        if self.cuda:
            one_hot = torch.mean(output)
        else:
            one_hot = torch.mean(output)

        self.feature_module.zero_grad()
        self.model.zero_grad()
        one_hot.backward(retain_graph=True)
        #numpy da eliminare, tutto in torch
        grads_val = self.extractor.get_gradients()[-1].cpu().data

        target = features[-1]
        

        target = target.cpu().data[0, :]

        weights = torch.mean(grads_val, axis=(2, 3))[0, :]
        cam = torch.zeros(target.shape[1:], dtype=torch.float32)

        for i, w in enumerate(weights):
            cam += w * target[i, :, :]

        cam = F.relu(cam)
        #cam = torch.maximum(cam, torch.zeros(target.shape[1:], dtype=torch.float32))
        #cam = cv2.resize(cam, input.shape[2:])
        cam = cam - torch.min(cam)
        cam = cam / torch.max(cam)
        return cam


