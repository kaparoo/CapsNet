# -*- coding: utf-8 -*-

import os
from typing import Dict


class CapsNetParam(object):
    """A Container for the hyperparamters of CapsNet.
    
    Attributes:
        conv1_filter (int; default=256): Number of filters for the convolution 
            in the `FeatureMap` instance.
        conv1_kernel (int; default=9): A size of the kernel for the convoluton
            in the `FeatureMap` instance.
        conv1_stride (int; default=1): A size of stride for the convolution in
            the `FeatureMap` instance.
        conv2_filter (int; default=256): Number of filters for the convolution 
            in the `PrimaryCap` instance. Always initialized as the product of 
            `num_primary` and `dim_primary`. 
        conv2_kernel (int; default=9): A size of the kernel for the convolution
            in the `PrimaryCap` instance.
        conv2_stride (int; default=2): A size of stride for the convolution in 
            the `PrimaryCap` instance.
        num_primary (int; default=32): Number of primary capsules for each grid
            in the `PrimaryCap` instance.
        dim_primary (int; default=8): A dimension of the primary capsule.
        num_digit (int; default=10): Number of digit capsules in the `DigitCap`
            instance.
        dim_digit (int; default=16): A dimension of the digit capsule.
        num_routings (int; default=3): Number of iterations for the dynamic rou
            ting mechanism in the `DigitCaps` instance.
    """

    __slots__ = [
        "conv1_filter", "conv1_kernel", "conv1_stride", "conv2_filter",
        "conv2_kernel", "conv2_stride", "num_primary", "dim_primary",
        "num_digit", "dim_digit", "num_routings"
    ]

    def __init__(self,
                 conv1_filter: int = 256,
                 conv1_kernel: int = 9,
                 conv1_stride: int = 1,
                 conv2_kernel: int = 9,
                 conv2_stride: int = 2,
                 num_primary: int = 32,
                 dim_primary: int = 8,
                 num_digit: int = 10,
                 dim_digit: int = 16,
                 num_routings: int = 3,
                 **kwargs) -> None:
        self.conv1_filter = conv1_filter
        self.conv1_kernel = conv1_kernel
        self.conv1_stride = conv1_stride
        self.conv2_filter = num_primary * dim_primary
        self.conv2_kernel = conv2_kernel
        self.conv2_stride = conv2_stride
        self.num_primary = num_primary
        self.dim_primary = dim_primary
        self.num_digit = num_digit
        self.dim_digit = dim_digit
        self.num_routings = num_routings

    def get_config(self) -> Dict[str, int]:
        return {
            "conv1_filter": self.conv1_filter,
            "conv1_kernel": self.conv1_kernel,
            "conv1_stride": self.conv1_stride,
            "conv2_filter": self.conv2_filter,
            "conv2_kernel": self.conv2_kernel,
            "conv2_stride": self.conv2_stride,
            "num_primary": self.num_primary,
            "dim_primary": self.dim_primary,
            "num_digit": self.num_digit,
            "dim_digit": self.dim_digit,
            "num_routings": self.num_routings
        }

    def save(self, path: str) -> None:
        """Saves configuration.
        
        Collects attributes as pair of name and value and saves them to a UTF-8
        encoded file.

        Args:
            path (str): A filepath to write configuration. If any file already 
                exists, its contents will be overwritten.
        
        Raises:
            TypeError: If `path` is not string.
            ValueError: If `path` is empty.
        """
        if not isinstance(path, str):
            raise TypeError()
        elif len(path) == 0:
            raise ValueError()
        else:
            with open(path, 'w', encoding="utf8") as f:
                for k, v in self.get_config().items():
                    f.writelines(f"{k}={v}\n")


def load_param(path: str) -> CapsNetParam:
    """Loads configuration.
        
    Reads file with the given path and makes `CapsNetParam` instance by parsing
    the contents.

    Args:
        path (str): A filepath to read configuration.
    
    Returns:
        A `CapsNetParam` instance.

    Raises:
        TypeError: If `path` is not string.
        ValueError: If `path` is empty.
        FileNotFoundError: If file of `path` not exists.
    """
    if not isinstance(path, str):
        raise TypeError()
    elif len(path) == 0:
        raise ValueError()
    elif not os.path.isfile(path):
        raise FileNotFoundError()

    with open(path, 'r', encoding="utf8") as f:
        config = []
        for l in f.readlines():
            k, v = l.strip().split('=')
            config.append((k, int(v)))
        return CapsNetParam(**dict(config))


def make_param(conv1_filter: int = 256,
               conv1_kernel: int = 9,
               conv1_stride: int = 1,
               conv2_kernel: int = 9,
               conv2_stride: int = 2,
               num_primary: int = 32,
               dim_primary: int = 8,
               num_digit: int = 10,
               dim_digit: int = 16,
               num_routings: int = 3) -> CapsNetParam:
    return CapsNetParam(conv1_filter=conv1_filter,
                        conv1_kernel=conv1_kernel,
                        conv1_stride=conv1_stride,
                        conv2_kernel=conv2_kernel,
                        conv2_stride=conv2_stride,
                        num_primary=num_primary,
                        dim_primary=dim_primary,
                        num_digit=num_digit,
                        dim_digit=dim_digit,
                        num_routings=num_routings)