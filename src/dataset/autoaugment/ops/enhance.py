"""
Operators for enhancing images.
"""

from PIL import ImageEnhance

from mindspore.dataset.vision import py_transforms_util


class Contrast:
    """
    Contrast adjusts the contrast of images.

    Args:
        degree (float): contrast degree ranging within [0.1, 1.9], where 1.0
                        indicating an unchanged contrast.
    """

    def __init__(self, degree):
        self.degree = degree

    def __call__(self, img):
        """
        Call method.

        Args:
            img (PIL image): PIL image to be adjusted.

        Returns:
            img (PIL image), Contrast adjusted image.
        """
        return py_transforms_util.adjust_contrast(img, self.degree)


class Color:
    """
    Color adjusts the saturation of images.

    Args:
        degree (float): saturation degree ranging within [0.1, 1.9], where 1.0
                        indicating an unchanged saturation.
    """

    def __init__(self, degree):
        self.degree = degree

    def __call__(self, img):
        """
        Call method.

        Args:
            img (PIL image): PIL image to be adjusted.

        Returns:
            img (PIL image), Saturation adjusted image.
        """
        return py_transforms_util.adjust_saturation(img, self.degree)


class Brightness:
    """
    Brightness adjusts the brightness of images.

    Args:
        degree (float): brightness degree ranging within [0.1, 1.9], where 1.0
                        indicating an unchanged brightness.
    """

    def __init__(self, degree):
        self.degree = degree

    def __call__(self, img):
        """
        Call method.

        Args:
            img (PIL image): Image to be adjusted.

        Returns:
            img (PIL image), Brightness adjusted image.
        """
        return py_transforms_util.adjust_brightness(img, self.degree)


class Sharpness:
    """
    Sharpness adjusts the sharpness of images.

    Args:
        degree (float): sharpness degree ranging within [0.1, 1.9], where 1.0
                        indicating an unchanged sharpness.
    """

    def __init__(self, degree):
        self.degree = degree

    def __call__(self, img):
        """
        Call method.

        Args:
            img (PIL image): Image to be sharpness adjusted.

        Returns:
            img (PIL image), Sharpness adjusted image.
        """
        if not py_transforms_util.is_pil(img):
            raise TypeError(
                py_transforms_util.augment_error_message.format(type(img)))

        return ImageEnhance.Sharpness(img).enhance(self.degree)
