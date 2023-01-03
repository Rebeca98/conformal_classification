import numpy as np
# Torch
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torchvision.models as models
import pathlib
import os
import pickle
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_calib_transform(image_size):
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return transform


def get_model(modelname='efficientnet_b0', pretrained=True):
    if modelname == 'efficientnet_b0':
        model = models.efficientnet_b0(weights='EfficientNet_B0_Weights.DEFAULT',
                                       pretrained=pretrained, progress=True)
        # weights=EfficientNet_B0_Weights.IMAGENET1K_V1

    elif modelname == 'efficientnet_b1':
        model = models.efficientnet_b1(weights='EfficientNet_B1_Weights',
                                       pretrained=pretrained, progress=True)

    elif modelname == 'efficientnet_b2':
        model = models.efficientnet_b2(pretrained=pretrained, progress=True)

    elif modelname == 'efficientnet_b3':
        model = models.efficientnet_b2(pretrained=pretrained, progress=True)

    elif modelname == 'efficientnet_b4':
        model = models.efficientnet_b2(pretrained=pretrained, progress=True)

    elif modelname == 'efficientnet_b5':
        model = models.efficientnet_b2(pretrained=pretrained, progress=True)

    elif modelname == 'efficientnet_b6':
        model = models.efficientnet_b2(pretrained=pretrained, progress=True)

    elif modelname == 'efficientnet_b7':
        model = models.efficientnet_b3(pretrained=pretrained, progress=True)

    else:
        raise NotImplementedError
    #model = torch.nn.DataParallel(model)
    return model


def build_model_for_cp(model_path, architecture, num_classes, pretrained=True):
    """
    build model for training
    model_path: Path to the model file with trained weight (e.g. model.pth)
    """
    if "efficientnet" in architecture:
        # we load our trained weights
        pretrained_weights = torch.load(model_path)
        model = get_model(architecture, pretrained)
        num_ftrs = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(
            in_features=num_ftrs, out_features=num_classes)
        model.load_state_dict(pretrained_weights['model_state_dict'])
    model.eval()
    return model


