from PIL import Image, ImageOps
import random

class HorizontalFlipSegmentation():
    def __init__(self, p):
        self.p = p

    def __call__(self, *args):
        c_p = random.random()

        if c_p < self.p:
            return args

        return [ImageOps.mirror(img) for img in args]

