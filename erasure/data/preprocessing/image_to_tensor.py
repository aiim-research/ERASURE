import torch
from torchvision import transforms
from PIL import Image
from erasure.data.preprocessing.preprocess import Preprocess

class ImageToTensorPreprocess(Preprocess):
    def __init__(self, global_ctx, local_ctx):
        super().__init__(global_ctx, local_ctx)

        self.transform = transforms.ToTensor()

    def process(self, X, y, z):

        if isinstance(X, list):
            X = X[0]


        if isinstance(X, Image.Image):
            X = self.transform(X)


        return X, y, z

    def check_configuration(self):
        super().check_configuration()
