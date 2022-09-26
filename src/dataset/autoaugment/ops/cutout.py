"""
RandomCutout operator.
"""

import random


class RandomCutout:
    """
    RandomCutout is similar to transforms.CutOut but is simplified and
    crafted for PIL images.

    Args:
        size (int): the side-size of each square cutout patch.
        num_patches (int): the number of square cutout patches to add.
        value (RGB value): the pixel value to fill in each cutout patches.
    """

    def __init__(self, size=30, num_patches=1, value=(125, 122, 113)):
        self.size = size
        self.num_patches = num_patches
        self.value = value

    @staticmethod
    def _clip(x, lower, upper):
        """Clip value to the [lower, upper] range."""
        return max(lower, min(x, upper))

    @staticmethod
    def _get_cutout_area(img_w, img_h, size):
        """Randomly create a cutout area."""
        x = random.randint(0, img_w)
        y = random.randint(0, img_h)
        x1 = x - size // 2
        x2 = x1 + size
        y1 = y - size // 2
        y2 = y1 + size
        x1 = RandomCutout._clip(x1, 0, img_w)
        x2 = RandomCutout._clip(x2, 0, img_w)
        y1 = RandomCutout._clip(y1, 0, img_h)
        y2 = RandomCutout._clip(y2, 0, img_h)
        return x1, y1, x2, y2

    def __call__(self, img):
        """
        Call method.

        Args:
            img (PIL image): Image to be cutout.

        Returns:
            img (PIL image), Randomly cutout image.
        """
        img_w, img_h = img.size
        pixels = img.load()

        for _ in range(self.num_patches):
            x1, y1, x2, y2 = self._get_cutout_area(img_w, img_h, self.size)
            for i in range(x1, x2):  # columns
                for j in range(y1, y2):  # rows
                    pixels[i, j] = self.value

        return img
