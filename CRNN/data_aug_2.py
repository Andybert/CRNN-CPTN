from __future__ import division

import imgaug as ia
from scipy.misc import imsave
from imgaug import augmenters as iaa
import numpy as np
import glob
import matplotlib.image as img
import matplotlib.pyplot as plt
import cv2

images = img.imread('/home/ahmed/Pictures/cogedis/cogedis_words_3/0d167103-1b66-4373-9a78-5b21f50f9abb.png')

#plt.imshow(image)
#plt.show()

st = lambda aug: iaa.Sometimes(0.3, aug)

# Define our sequence of augmentation steps that will be applied to every image
# All augmenters with per_channel=0.5 will sample one value _per image_
# in 50% of all cases. In all other cases they will sample new values
# _per channel_.

#seq_flipud=iaa.Sequential([st(iaa.GaussianBlur((1.5)))])
'''
seq = iaa.Sequential([

        iaa.Flipud(0.5), # vertically flip 50% of all images
        

        st(iaa.GaussianBlur((1.5))), # blur images with a sigma between 0 and 3.0
        st(iaa.Sharpen(alpha=(1.0), lightness=(0.5))), # sharpen images
        st(iaa.Emboss(alpha=(1.0), strength=(1.0))), # emboss images
        # search either for all edges or for directed edges
        st(iaa.AdditiveGaussianNoise(loc=0, scale=(0.2), per_channel=0.5)), # add gaussian noise to images
        st(iaa.Dropout((0.1), per_channel=0.5)), # randomly remove up to 10% of the pixels
        st(iaa.Invert(0.25, per_channel=True)), # invert color channels
        #st(iaa.Add((-10, 10), per_channel=0.5)), # change brightness of images (by -10 to 10 of original value)
        #st(iaa.Multiply((0.5, 1.5), per_channel=0.5)), # change brightness of images (50-150% of original value)
        #st(iaa.ContrastNormalization((0.5, 2.0), per_channel=0.5)), # improve or worsen the contrast
        st(iaa.Affine(
            scale={"x": (0.8, 1.2), "y": (0.8, 1.2)}, # scale images to 80-120% of their size, individually per axis
            translate_px={"x": (-16, 16), "y": (-16, 16)}, # translate by -16 to +16 pixels (per axis)
            rotate=(-90, 90), # rotate by -45 to +45 degrees


        )),
        st(iaa.ElasticTransformation(alpha=(3.0), sigma=0.25)) # apply elastic transformations with random strengths
    ],
    random_order=False # do all of the above in random order
)

#images_aug = seq.augment_images(images)
'''

'''
gaussianBlur=iaa.Sequential([st(iaa.GaussianBlur(1.5))])
g=gaussianBlur.show_grid(images,cols=1,rows=1)
sharpen=iaa.Sequential([st(iaa.Sharpen(alpha=1.0,lightness=1))])
s=sharpen.show_grid(images,cols=1,rows=1)
images_aug = sharpen.augment_images(images)
'''
#imsave('/home/ahmed/Pictures/cogedis/cogedis_words_3/0aa3a241-1d17-4173-958f-41da009281c9_sharpen.png',sharpen.show_grid(images,cols=1,rows=1))
gaussianBlur=iaa.Sequential([st(iaa.GaussianBlur(1.0))])
#images_aug=gaussianBlur.augment_image(images)

additive=st(iaa.AdditiveGaussianNoise(loc=0, scale=(0.1), per_channel=0.7))
img=additive.augment_image(images)
#images_aug = sharpen.augment_images(images)
print(img.shape)
plt.imshow(img)
#plt.savefig('/home/ahmed/Pictures/cogedis/cogedis_words_3/0aa3a241-1d17-4173-958f-41da009281c9_blur.png')
plt.show()
print("ok")
