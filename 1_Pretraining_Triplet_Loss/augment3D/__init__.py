import random

import numpy as np

from .elastic_deform import ElasticTransform
from .random_crop import RandomCropToLabels
from .random_flip import RandomFlip
from .random_rescale import RandomZoom
from .random_rotate import RandomRotation
from .random_shift import RandomShift
from .gaussian_noise import GaussianNoise

functions = ['elastic_deform', 'random_crop', 'random_flip', 'random_rescale', 'random_rotate', 'random_shift']


class RandomChoice(object):
    """
    choose a random tranform from list an apply
    transforms: tranforms to apply
    p: probability
    """

    def __init__(self, transforms=[],
                 p=0.5):
        self.transforms = transforms
        self.p = p

    def __call__(self, img_tensors):
        augment = np.random.random(1) < self.p
        if not augment:
            return img_tensors
        t = random.choice(self.transforms)

        # print(t)
        img_tensors = t(img_tensors)

        # for i in range(len(img_tensors)):
        #
        #     if i == (len(img_tensors) - 1):
        #         ### do only once the augmentation to the label
        #         img_tensors[i] = t(img_tensors[i])
        #     else:
        #         img_tensors[i], _ = t(img_tensors[i])
        return img_tensors


class ComposeTransforms(object):
    """
    Composes several transforms together.
    """

    def __init__(self, transforms=[],
                 p=0.9):
        self.transforms = transforms
        self.p = p

    def __call__(self, img_tensors):
        augment = np.random.random(1) < self.p
        if not augment:
            return img_tensors

        for i in range(len(img_tensors)):

            for t in self.transforms:
                if i == (len(img_tensors) - 1):
                    ### do only once augmentation to the label
                    img_tensors[i] = t(img_tensors[i])
                else:
                    img_tensors[i], _ = t(img_tensors[i])
        return img_tensors
