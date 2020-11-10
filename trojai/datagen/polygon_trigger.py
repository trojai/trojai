# NIST-developed software is provided by NIST as a public service. You may use, copy and distribute copies of the software in any medium, provided that you keep intact this entire notice. You may improve, modify and create derivative works of the software or any portion of the software, and you may copy and distribute such modifications or works. Modified works should carry a notice stating that you changed the software and should note the date and nature of any such change. Please explicitly acknowledge the National Institute of Standards and Technology as the source of the software.

# NIST-developed software is expressly provided "AS IS." NIST MAKES NO WARRANTY OF ANY KIND, EXPRESS, IMPLIED, IN FACT OR ARISING BY OPERATION OF LAW, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT AND DATA ACCURACY. NIST NEITHER REPRESENTS NOR WARRANTS THAT THE OPERATION OF THE SOFTWARE WILL BE UNINTERRUPTED OR ERROR-FREE, OR THAT ANY DEFECTS WILL BE CORRECTED. NIST DOES NOT WARRANT OR MAKE ANY REPRESENTATIONS REGARDING THE USE OF THE SOFTWARE OR THE RESULTS THEREOF, INCLUDING BUT NOT LIMITED TO THE CORRECTNESS, ACCURACY, RELIABILITY, OR USEFULNESS OF THE SOFTWARE.

# You are solely responsible for determining the appropriateness of using and distributing the software and you assume all risks associated with its use, including but not limited to the risks and costs of program errors, compliance with applicable laws, damage to or loss of data, programs or equipment, and the unavailability or interruption of operation. This software is not intended to be used in any situation where a failure could cause risk of injury or damage to property. The software developed by NIST employees is not subject to copyright protection within the United States.

import numpy as np

import PIL.Image
import PIL.ImageDraw
import skimage.io
from functools import reduce
import operator
import math

from .image_entity import ImageEntity


class PolygonTrigger(ImageEntity):
    """
    Defines a programmatically created polygon trigger.
    """
    def __init__(self, img_size: int, n_sides: int, color=None):
        """
        Initializes a polygon object
        :param img_size: the size of the polygons constraining rectangle.
        :param n_sides: the number of polygon sides.
        :param color: polygon color
        """

        if color is not None and len(color) != 3:
            raise RuntimeError('Polygon color must be a 3 elements.')

        self.data = PIL.Image.new('RGBA', (img_size, img_size))
        draw = PIL.ImageDraw.Draw(self.data)

        coords = list()
        for pt in range(n_sides):
            x = np.random.randint(int(0.1 * img_size), int(0.9 * img_size))
            y = np.random.randint(int(0.1 * img_size), int(0.9 * img_size))
            coords.append((x, y))

        center = tuple(map(operator.truediv, reduce(lambda x, y: map(operator.add, x, y), coords), [len(coords)] * 2))
        coords = sorted(coords, key=lambda coord: (-135 - math.degrees(math.atan2(*tuple(map(operator.sub, coord, center))[::-1]))) % 360)

        min_x = img_size
        max_x = 0
        min_y = img_size
        max_y = 0
        for c in coords:
            min_x = min(min_x, c[0])
            max_x = max(max_x, c[0])
            min_y = min(min_y, c[1])
            max_y = max(max_y, c[1])

        x_scale = img_size / (max_x - min_x)
        y_scale = img_size / (max_y - min_y)

        new_coords = list()
        for c in coords:
            new_x = int(x_scale * (c[0] - min_x))
            new_y = int(y_scale * (c[1] - min_y))
            new_coords.append((new_x, new_y))

        coords = new_coords
        if color is None:
            color = [1, 1, 1]
        draw.polygon(coords, fill=(color[0], color[1], color[2], 255))
        # convert from PIL image to numpy array
        self.data = np.array(self.data)
        self.mask = (self.data[:, :, 3] > 0).astype(bool)

    def get_data(self) -> np.ndarray:
        """
        Get the data associated with the ImageEntity
        :return: return a numpy.ndarray representing the image
        """
        return self.data

    def get_mask(self) -> np.ndarray:
        """
        Get the mask associated with the ImageEntity
        :return: return a numpy.ndarray representing the mask
        """
        return self.mask

    def save(self, output_filepath: str):
        skimage.io.imsave(output_filepath, self.data)


