"""
RandomCrop operator.
"""

from mindspore.dataset.vision import transforms
from mindspore.dataset.vision import py_transforms_util
from mindspore.dataset.vision import utils


class RandomCrop(transforms.RandomCrop):
    """
    RandomCrop inherits from transforms.RandomCrop but derives/uses the
    original image size as the output size.

    Please refer to transforms.RandomCrop for argument specifications.
    """

    def __init__(self, padding=4, pad_if_needed=False,
                 fill_value=0, padding_mode=utils.Border.CONSTANT):
        # Note the `1` for the size argument is only set for passing the check.
        super(RandomCrop, self).__init__(1, padding=padding, pad_if_needed=pad_if_needed,
                                         fill_value=fill_value, padding_mode=padding_mode)

    def __call__(self, img):
        """
        Call method.

        Args:
            img (PIL image): Image to be padded and then randomly cropped back
                             to the same size.

        Returns:
            img (PIL image), Randomly cropped image.
        """
        if not py_transforms_util.is_pil(img):
            raise TypeError(
                py_transforms_util.augment_error_message.format(type(img)))

        return py_transforms_util.random_crop(
            img, img.size, self.padding, self.pad_if_needed,
            self.fill_value, self.padding_mode,
        )
