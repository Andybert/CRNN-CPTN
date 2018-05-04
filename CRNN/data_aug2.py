import imgaug as ia
from imgaug import augmenters as iaa
import numpy as np

from imgaug import augmenters as iaa

def param():
    x=iaa.Sequential(iaa.Crop(px=(0, 16)))
    y=iaa.Sequential(iaa.Fliplr(0.5))
    z=iaa.Sequential(iaa.GaussianBlur(sigma=(0, 3.0)))



for batch_idx in range(1000):
    # 'images' should be either a 4D numpy array of shape (N, height, width, channels)
    # or a list of 3D numpy arrays, each having shape (height, width, channels).
    # Grayscale images must have shape (height, width, 1) each.
    # All images must have numpy's dtype uint8. Values are expected to be in
    # range 0-255.
    images = load_batch(batch_idx)
    images_aug = seq.augment_images(images)
    train_on_images(images_aug)