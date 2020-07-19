import torch
from pool_net.models.poolnet import VggPoolNet
from pool_net.datasets.dataset import default_transform

from PIL import Image
import numpy as np


class PoolNetInterface(object):
    def __init__(
        self,
        weight_paths,
        device="gpu",
        *,
        transform=default_transform,
        prefix="core."
    ):
        """
        Args:
            weight_paths (str): Path to model weight
            device (str): gpu or cpu

            transform (Callable): Image transformation
            prefix(str): State dict key prefix
        """
        self._device = self.get_device(device)
        self._core = VggPoolNet()
        self._transform = transform

        self.load_weight(weight_paths, prefix=prefix)

        self._core.to(self._device)
        self._core.eval()

    def load_weight(self, weight_paths, *, prefix=""):
        """
        Args:
            weight_paths (str): Path to checkpoint
            prefix (str): State dict key prefix
                - For pytorch_lightning trainer checkpoint: prefix="core.". 
                    Usually saved under lightning_logs/version_xx/

                - For normal weight: prefix="". 
                    Saved in lightning_logs after training.
        """
        checkpoint = torch.load(weight_paths, map_location="cpu")
        state_dict = checkpoint["state_dict"]

        if prefix is None or len(prefix) == 0:
            self._core.load_state_dict(state_dict)
            return

        local_state_dict = {}
        len_prefix = len(prefix)
        for k, v in state_dict.items():
            if k.startswith(prefix):
                new_key = k[len_prefix:]
                local_state_dict[new_key] = v
        self._core.load_state_dict(local_state_dict)

    def get_device(self, device):
        if device.startswith("gpu"):
            device = "cuda" + device[3:]
        return device

    def _load_img(self, img):
        """ Load image warper 
        Args:
            img (str or PIL Image or np.ndarray)
        Return:
            img_tensor
        """
        if isinstance(img, str):
            img = Image.open(img)
        elif isinstance(img, np.ndarray):
            img = Image.fromarray(img)
        img = img.convert("RGB")

        return self._transform(img).to(self._device)

    def process(self, img, threshold=0.5):
        """ Predict image saliency object mask

        Args:
            img (str or PIL Image or np.ndarray)
            threshold (float or None): If None return probility mask
        Return:
            mask (np.ndarray with shape W x H)
        """
        if isinstance(img, list):
            return [self.process(i) for i in img]

        img = self._load_img(img)
        img = img.unsqueeze(0)

        with torch.no_grad():
            mask = self._core(img)
            mask = torch.sigmoid(mask)

        if threshold is None:
            return mask.cpu().numpy()[0][0]

        mask = mask > threshold
        mask = mask.cpu().numpy()

        final_mask = np.zeros_like(mask, dtype=np.uint8)
        final_mask[mask] = 255

        return final_mask[0][0]
