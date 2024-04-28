import numpy as np
from PIL import Image
from math import *
import torch
import torchvision.transforms.functional as TF

"""## Create dataset class"""
class Transforms():
    def __init__(self):
        pass

    def resize(self, image, img_size):
        image = TF.resize(image, img_size)
        return image


    def crop(self, image, crops, landmarks):
        left_landmark = int(crops['left'])
        top_landmark = int(crops['top'])
        width = int(crops['width'])
        height = int(crops['height'])

        image = TF.crop(image, top_landmark, left_landmark, height, width)

        img_shape = np.array(image).shape
        landmarks = torch.tensor(landmarks) - torch.tensor([[left_landmark, top_landmark]])
        landmarks = landmarks / torch.tensor([img_shape[1], img_shape[0]])
        return image, landmarks

    def __call__(self, image, landmarks, crops):
        image = Image.fromarray(image)
        image, landmarks = self.crop(image, crops=crops, landmarks=landmarks)
        image = self.resize(image=image, img_size=(224, 224))
        image = TF.to_tensor(image)
        image = TF.normalize(image, [0.5], [0.5])
        return image, landmarks