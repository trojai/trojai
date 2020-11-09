from trojai.datagen.image_entity import GenericImageEntity, ImageEntity
from trojai.datagen.transform_interface import ImageTransform
from trojai.datagen.image_conversion_utils import normalization_to_rgb, normalization_from_rgb

import wand.image
import wand.color
import wand.drawing
import numpy as np
from numpy.random import RandomState
import cv2

import math
from abc import abstractmethod
from io import BytesIO

import logging
logger = logging.getLogger(__name__)


class FilterXForm(ImageTransform):
    """
    Create filter xform, if no channel order is specified it is assumed to be in BGR order (opencv default),
    this refers only to the first 3 channels of input data as the alpha channel is handled independently
    """
    def __init__(self, channel_order: str = 'BGR', pre_normalize: bool = True, post_normalize: bool = True):
        self.valid_channel_orders = {'BGR', 'RGB'}
        if channel_order not in self.valid_channel_orders:
            raise ValueError("Unknown channel order specified for %s!"
                             "  Valid options are %s!" % (self.__repr__(), str(self.valid_channel_orders)))
        self.channel_order = channel_order
        self.pre_normalize = pre_normalize
        self.post_normalize = post_normalize

    """
    Abstract class containing wand interface functionality common to all filter transforms
    """
    @abstractmethod
    def filter(self, image: wand.image.Image) -> wand.image.Image:
        """
        subclass defined function to be called by do
        :param image: wand Image to be filtered
        :return: filtered wand Image
        """
        pass

    def do(self, input_obj: ImageEntity, random_state_obj: RandomState) -> ImageEntity:
        """
        Compresses 3-channel image input image as a specified filetype and stores in memory,
        passes to into wand and applies filter, stores filtered image as specified filetype again in memory,
        which is then decompressed back into 3-channel image
        :param input_obj: entity to be transformed
        :param random_state_obj: object to hold random state and enable reproducibility
        :return:new entity with transform applied
        """
        data = input_obj.get_data()
        original_n_chan = data.shape[2]
        rgb_data, alpha_ch = normalization_to_rgb(data, self.pre_normalize, self.__repr__())
        logger.debug("%s is treating input as %s!" % (self.__repr__(), self.channel_order))
        if self.channel_order == 'RGB':
            rgb_data = cv2.cvtColor(rgb_data, cv2.COLOR_RGB2BGR)

        form = '.bmp'
        success, buffer = cv2.imencode(form, rgb_data)
        # faster than numpy tobytes method
        input_stream = BytesIO(buffer)
        with wand.image.Image(blob=input_stream.getvalue()) as wand_image:
            filtered_wand_image = self.filter(wand_image)
            output_stream = BytesIO()
            filtered_wand_image.save(output_stream)
            rgb_filtered_data = cv2.imdecode(np.frombuffer(output_stream.getbuffer(), np.uint8), 1)

            if self.channel_order == 'RGB':
                rgb_filtered_data = cv2.cvtColor(rgb_filtered_data, cv2.COLOR_BGR2RGB)
            filtered_data = normalization_from_rgb(rgb_filtered_data, alpha_ch, self.post_normalize,
                                                   original_n_chan, self.__repr__())
            return GenericImageEntity(filtered_data, input_obj.get_mask())

    def _colortone(self, image: wand.image.Image, color: str, dst_percent: int, invert: bool) -> None:
        """
        tones either white or black values in image to the provided color,
        intensity of toning depends on dst_percent
        :param image: provided image
        :param color: color to tone image
        :param dst_percent: percentage of image pixel value to include when blending with provided color,
        0 is unchanged, 100 is completely colored in
        :param invert: if True blacks are modified, if False whites are modified
        :return:
        """
        mask_src = image.clone()
        mask_src.colorspace = 'gray'
        if invert:
            mask_src.negate()
        mask_src.alpha_channel = 'copy'

        src = image.clone()
        src.colorize(wand.color.Color(color), wand.color.Color('#FFFFFF'))
        src.composite_channel('alpha', mask_src, 'copy_alpha')

        image.composite_channel('default_channels', src, 'blend',
                                arguments=str(dst_percent) + "," + str(100 - dst_percent))

    def _vignette(self, image: wand.image.Image, color_1: str = 'none', color_2: str = 'black',
                  crop_factor: float = 1.5) -> None:
        """
        applies fading from color_1 to color_2 in radial gradient pattern on given image
        :param image: provided image
        :param color_1: center color
        :param color_2: edge color
        :param crop_factor: size of radial gradient pattern, which is then cropped and combined with image,
        larger values include more of color_1, smaller values include more of color_2
        :return: None
        """
        crop_x = math.floor(image.width * crop_factor)
        crop_y = math.floor(image.height * crop_factor)
        src = wand.image.Image()
        src.pseudo(width=crop_x, height=crop_y, pseudo='radial-gradient:' + color_1 + '-' + color_2)
        src.crop(0, 0, width=image.width, height=image.height, gravity='center')
        src.reset_coords()
        image.composite_channel('default_channels', src, 'multiply')
        image.merge_layers('flatten')


class GothamFilterXForm(FilterXForm):
    """
    Class implementing Instagram's Gotham filter
    """
    def filter(self, image: wand.image.Image) -> wand.image.Image:
        """
        modified from https://github.com/acoomans/instagram-filters/blob/master/instagram_filters/filters/gotham.py
        :param image: provided image
        :return: new filtered image
        """
        filtered_image = image.clone()
        filtered_image.modulate(120, 10, 100)
        filtered_image.colorize(wand.color.Color('#222b6d'), wand.color.Color('#333333'))
        filtered_image.gamma(.9)
        filtered_image.sigmoidal_contrast(True, 3, .5 * filtered_image.quantum_range)
        filtered_image.sigmoidal_contrast(True, 3, .5 * filtered_image.quantum_range)
        return filtered_image


class NashvilleFilterXForm(FilterXForm):
    """
    Class implementing Instagram's Nashville filter
    """
    def filter(self, image: wand.image.Image) -> wand.image.Image:
        """
        modified from https://github.com/acoomans/instagram-filters/blob/master/instagram_filters/filters/nashville.py
        :param image:
        :return: new filtered image
        """
        filtered_image = image.clone()
        self._colortone(filtered_image, '#222b6d', 50, True)
        self._colortone(filtered_image, '#f7daae', 50, False)
        filtered_image.sigmoidal_contrast(True, 3, .5 * filtered_image.quantum_range)
        filtered_image.modulate(100, 150, 100)
        filtered_image.auto_gamma()
        return filtered_image


class KelvinFilterXForm(FilterXForm):
    """
    Class implementing Instagram's Kelvin filter
    """
    def filter(self, image: wand.image.Image) -> wand.image.Image:
        """
        modified from https://github.com/acoomans/instagram-filters/blob/master/instagram_filters/filters/kelvin.py
        :param image: provided image
        :return: new filtered image
        """
        filtered_image = image.clone()
        filtered_image.auto_gamma()
        filtered_image.modulate(120, 50, 100)
        with wand.drawing.Drawing() as draw:
            draw.fill_color = '#FF9900'
            draw.fill_opacity = 0.2
            draw.rectangle(left=0, top=0, width=filtered_image.width, height=filtered_image.height)
            draw(filtered_image)
        return filtered_image


class LomoFilterXForm(FilterXForm):
    """
    Class implementing Instagram's Lomo filter
    """
    def filter(self, image: wand.image.Image) -> wand.image.Image:
        """
        modified from https://github.com/acoomans/instagram-filters/blob/master/instagram_filters/filters/lomo.py
        :param image: provided image
        :return: new filtered image
        """
        filtered_image = image.clone()
        filtered_image.level(.5, channel="R")
        filtered_image.level(.5, channel="G")
        self._vignette(filtered_image)
        return filtered_image


class ToasterXForm(FilterXForm):
    """
    Class implementing Instagram's Toaster filter
    """
    def filter(self, image: wand.image.Image) -> wand.image.Image:
        """
        modified from https://github.com/acoomans/instagram-filters/blob/master/instagram_filters/filters/toaster.py
        :param image: provided image
        :return: new filtered image
        """
        filtered_image = image.clone()
        self._colortone(filtered_image, '#330000', 50, True)
        filtered_image.modulate(150, 80, 100)
        filtered_image.gamma(1.2)
        filtered_image.sigmoidal_contrast(True, 3, .5 * filtered_image.quantum_range)
        filtered_image.sigmoidal_contrast(True, 3, .5 * filtered_image.quantum_range)
        self._vignette(filtered_image, 'none', 'LavenderBlush3')
        self._vignette(filtered_image, '#ff9966', 'none')
        return filtered_image


class NoOpFilterXForm(FilterXForm):
    """
    No operation Transform for testing purposes
    """
    def filter(self, image: wand.image.Image) -> wand.image.Image:
        return image
