from PIL import Image, ImageOps
import random

class HorizontalFlipSegmentation():
    def __init__(self, p):
        self.p = p

    def __call__(self, img, mask):
        c_p = random.random()

        if c_p < self.p:
            return img, mask

        img = ImageOps.mirror(img)
        mask = ImageOps.mirror(mask)

        return img, mask

