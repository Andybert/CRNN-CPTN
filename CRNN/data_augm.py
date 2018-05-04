import imgaug as ia
from imgaug import augmenters as iaa
import numpy as np
import glob
import matplotlib.image as img
import matplotlib.pyplot as plt
import cv2

images= img.imread("/home/ahmed/Pictures/cogedis/cogedis_words_3/fe27c8de-15d5-4361-a079-558a4bee2388.png")

plt.imshow(images)
# random example images
#images = np.random.randint(0, 255, (16, 128, 128, 3), dtype=np.uint8)

# Sometimes(0.5, ...) applies the given augmenter in 50% of all cases,
# e.g. Sometimes(0.5, GaussianBlur(0.3)) would blur roughly every second image.
st = lambda aug: iaa.Sometimes(0.3, aug)

# Define our sequence of augmentation steps that will be applied to every image
# All augmenters with per_channel=0.5 will sample one value _per image_
# in 50% of all cases. In all other cases they will sample new values
# _per channel_.
seq = iaa.Sequential([

        iaa.Flipud(0.5), # vertically flip 50% of all images
        st(iaa.Superpixels(p_replace=(0, 1.0), n_segments=(20, 200))), # convert images into their superpixel representation

        st(iaa.GaussianBlur((0, 3.0))), # blur images with a sigma between 0 and 3.0
        st(iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5))), # sharpen images
        st(iaa.Emboss(alpha=(0, 1.0), strength=(0, 2.0))), # emboss images
        # search either for all edges or for directed edges
        st(iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.2), per_channel=0.5)), # add gaussian noise to images
        st(iaa.Dropout((0.0, 0.1), per_channel=0.5)), # randomly remove up to 10% of the pixels
        st(iaa.Invert(0.25, per_channel=True)), # invert color channels
        st(iaa.Add((-10, 10), per_channel=0.5)), # change brightness of images (by -10 to 10 of original value)
        st(iaa.Multiply((0.5, 1.5), per_channel=0.5)), # change brightness of images (50-150% of original value)
        st(iaa.ContrastNormalization((0.5, 2.0), per_channel=0.5)), # improve or worsen the contrast
        st(iaa.Affine(
            scale={"x": (0.8, 1.2), "y": (0.8, 1.2)}, # scale images to 80-120% of their size, individually per axis
            translate_px={"x": (-16, 16), "y": (-16, 16)}, # translate by -16 to +16 pixels (per axis)
            rotate=(-45, 45), # rotate by -45 to +45 degrees
            shear=(-16, 16), # shear by -16 to +16 degrees
            order=ia.ALL, # use any of scikit-image's interpolation methods
            cval=(0, 255), # if mode is constant, use a cval between 0 and 255
            mode=ia.ALL # use any of scikit-image's warping modes (see 2nd image from the top for examples)
        )),
        st(iaa.ElasticTransformation(alpha=(0.5, 3.5), sigma=0.25)) # apply elastic transformations with random strengths
    ],
    random_order=False # do all of the above in random order
)



images_aug = seq.augment_images(images)

seq.show_grid(images_aug[0:13,:,:], cols=13, rows=1)
print(images_aug.shape)