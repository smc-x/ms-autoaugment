"""
Operators for applying image effects.
"""

from PIL import ImageOps

from mindspore.dataset.vision import py_transforms_util


class Solarize:
    """
    Solarize inverts image pixels with values above the configured threshold.

    Args:
        threshold (int): All pixels above the threshold would be inverted.
                         Ranging within [0, 255].
    """

    def __init__(self, threshold):
        self.threshold = threshold

    def __call__(self, img):
        """
        Call method.

        Args:
            img (PIL image): Image to be solarized.

        Returns:
            img (PIL image), Solarized image.
        """
        if not py_transforms_util.is_pil(img):
            raise TypeError(
                py_transforms_util.augment_error_message.format(type(img)))

        return ImageOps.solarize(img, self.threshold)


class Posterize:
    """
    Posterize reduces the number of bits for each color channel.

    Args:
        bits (int): The number of bits to keep for each channel.
                    Ranging within [1, 8].
    """

    def __init__(self, bits):
        self.bits = bits

    def __call__(self, img):
        """
        Call method.

        Args:
            img (PIL image): Image to be posterized.

        Returns:
            img (PIL image), Posterized image.
        """
        if not py_transforms_util.is_pil(img):
            raise TypeError(
                py_transforms_util.augment_error_message.format(type(img)))

        return ImageOps.posterize(img, self.bits)
