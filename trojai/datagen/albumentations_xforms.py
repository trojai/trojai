import math

from trojai.datagen.transform_interface import ImageTransform
from trojai.datagen.image_entity import ImageEntity, GenericImageEntity
from trojai.datagen.image_conversion_utils import normalization_to_rgb, normalization_from_rgb
import albumentations.augmentations.transforms as albu

from numpy.random import RandomState

import logging
logger = logging.getLogger(__name__)


"""
A wrapper for all weather related transforms provided by Albumentations project
"""


class BrightenXForm(ImageTransform):
    """Brightens an image by a specified amount.

    """
    def __init__(self, brightness_coeff: float = -1) -> None:
        """
        Initialize the brightener
        :param brightness_coeff: coefficient between 0 and 1 that controls brightness,
        if brightness_coeff is -1, the value is drawn from uniform distribution on [0.0, 1.0) upon each application
        """
        self.brightness_coeff = brightness_coeff
        brightness_coeff_lower, brightness_coeff_upper = brightness_coeff, brightness_coeff
        if self.brightness_coeff == -1:
            brightness_coeff_lower, brightness_coeff_upper = 0.0, 1.0
        self.brighten_object = albu.RandomBrightnessContrast((brightness_coeff_lower, brightness_coeff_upper),
                                                             (0.0, 0.0), always_apply=True)

    def do(self, input_obj: ImageEntity, random_state_obj: RandomState) -> ImageEntity:
        """
        Perform the actual defined transformation
        :param input_obj: the input Entity to be transformed
        :param random_state_obj: ignored
        :return: the transformed Entity
        """
        img = input_obj.get_data()
        logger.debug("Applying albumentations.RandomBrightnessContrast with coeff=%0.02f" % (self.brightness_coeff,))
        img_xformed = self.brighten_object(random_state=random_state_obj, image=img)['image']

        return GenericImageEntity(img_xformed, input_obj.get_mask())


class DarkenXForm(ImageTransform):
    """ Darkens an image by a specified amount.

    """
    def __init__(self, darkness_coeff: float = -1) -> None:
        """
        Initialize the darkener
        :param darkness_coeff: coefficient between 0 and 1 that controls darkness,
        if brightness_coeff is -1, the value is drawn from the uniform distribution on [0.0, 1.0) upon each
        application
        """
        self.darkness_coeff = darkness_coeff
        darkness_coeff_lower, darkness_coeff_upper = -darkness_coeff, -darkness_coeff
        if self.darkness_coeff == -1:
            darkness_coeff_lower, darkness_coeff_upper = -1.0, 0.0
        self.darken_object = albu.RandomBrightnessContrast((darkness_coeff_lower, darkness_coeff_upper),
                                                           (0.0, 0.0), always_apply=True)

    def do(self, input_obj: ImageEntity, random_state_obj: RandomState) -> ImageEntity:
        """
        Perform the actual defined transformation
        :param input_obj: the input Entity to be transformed
        :param random_state_obj: ignored
        :return: the transformed Entity
        """
        img = input_obj.get_data()
        logger.debug("Applying albumentations.RandomBrightnessContrast with coeff=%0.02f" % (self.darkness_coeff,))
        img_xformed = self.darken_object(random_state=random_state_obj, image=img)['image']

        return GenericImageEntity(img_xformed, input_obj.get_mask())


class RandomDarkenOrBrightenXForm(ImageTransform):
    """ Randomly brightens/darkens an image by a specified amount

    """
    def __init__(self) -> None:
        """
        Initializes the random brightener/darkener
        """
        self.darken_or_brighten_object = albu.RandomBrightnessContrast((-1.0, 1.0), (0.0, 0.0), always_apply=True)

    def do(self, input_obj: ImageEntity, random_state_obj: RandomState) -> ImageEntity:
        """
        Perform the actual defined transformation
        :param input_obj: the input Entity to be transformed
        :param random_state_obj: ignored
        :return: the transformed Entity
        """
        img = input_obj.get_data()
        logger.debug("Applying albumentations.RandomBrightnessContrast")
        img_xformed = self.darken_or_brighten_object(random_state=random_state_obj, image=img)['image']

        return GenericImageEntity(img_xformed, input_obj.get_mask())


class AddShadowXForm(ImageTransform):
    """ Adds a shadow to an image

    """
    def __init__(self, no_of_shadows: int = 1, rectangular_roi: tuple = (-1, -1, -1, -1),
                 shadow_dimension: int = 5, pre_normalize: bool = True, post_normalize: bool = True) -> None:
        """
        Initializes the shadow adder
        :param no_of_shadows: the # of shadows
        :param rectangular_roi: region of interest (in pixels) where shadow will be added,
        if rectangular_roi is -1, the roi is chosen uniformly from the lower half of the image upon each application
        :param shadow_dimension: the # of sides to the shadow
        :param pre_normalize: whether to automatically convert input to this transform's required channel format (RGB)
        :param post_normalize: whether to automatically convert output back to its original channel format
        """
        self.no_of_shadows = no_of_shadows
        self.rectangular_roi = rectangular_roi
        self.shadow_dimension = shadow_dimension
        self.shadow_object = None
        self.pre_normalize = pre_normalize
        self.post_normalize = post_normalize

    def do(self, input_obj: ImageEntity, random_state_obj: RandomState) -> ImageEntity:
        """
        Perform the actual defined transformation
        :param input_obj: the input Entity to be transformed
        :param random_state_obj: ignored
        :return: the transformed Entity
        """
        img = input_obj.get_data()
        original_n_chan = img.shape[2]
        rgb_img, alpha_ch = normalization_to_rgb(img, self.pre_normalize, "AddShadowXForm")

        if self.shadow_object is None:
            no_of_shadows_lower, no_of_shadows_upper = self.no_of_shadows, self.no_of_shadows
            # RandomShadow requires roi to be normalized on [0.0, 1.0]
            roi = ()
            if self.rectangular_roi == (-1, -1, -1, -1):
                roi = (0.0, 0.5, 1.0, 1.0)
            else:
                roi_x1 = self.rectangular_roi[0] / rgb_img.shape[1]
                roi_y1 = self.rectangular_roi[1] / rgb_img.shape[0]
                roi_x2 = self.rectangular_roi[2] / rgb_img.shape[1]
                roi_y2 = self.rectangular_roi[3] / rgb_img.shape[0]
                roi = (roi_x1, roi_y1, roi_x2, roi_y2)
            self.shadow_object = albu.RandomShadow(roi, no_of_shadows_lower, no_of_shadows_upper,
                                                   self.shadow_dimension, always_apply=True)

        logger.debug("Applying albumentations.RandomShadow with shadows=%d, ROI=%s, dimension=%d, pre_normalization=%s,"
                    "post_normalization=%s" %
                    (self.no_of_shadows, str(self.rectangular_roi), self.shadow_dimension, self.pre_normalize,
                     self.post_normalize,))

        rgb_img_xformed = self.shadow_object(random_state=random_state_obj, image=rgb_img)['image']
        img_xformed = normalization_from_rgb(rgb_img_xformed, alpha_ch, self.post_normalize, original_n_chan,
                                             "AddShadowXForm")
        return GenericImageEntity(img_xformed, input_obj.get_mask())


class AddRainXForm(ImageTransform):
    """ Adds rain to an image

    """
    def __init__(self, slant: float = -1, drop_length: float = 20, drop_width: float = 1,
                 drop_color: tuple = (200, 200, 200), rain_type: str = None, pre_normalize: bool = True,
                 post_normalize: bool = True, always_apply: bool = True, probability: float = 0.5):
        """
        Initializes the rain adder
        :param slant:  deviation of rain from normal (-20<=slant<=20),
        if slant is -1, the value will be an integer chosen uniformly on [-10,10) upon each application
        :param drop_length: length of the drop (0<=drop_length<=100)
        :param drop_width: width of the drop (1<=drop_width<=5)
        :param drop_color: color of droplets
        :param rain_type: values in 'drizzle','heavy','torrential', or can be None
        :param pre_normalize: whether to automatically convert input to this transform's required channel format (RGB)
        :param post_normalize: whether to automatically convert output back to its original channel format
        :param always_apply: whether to always apply the transformation, or apply probability percent of the time
        :param probability: probability of the transformation being applied
        """
        self.slant = slant
        slant_lower, slant_upper = slant, slant
        if self.slant == -1:
            slant_lower, slant_upper = -10, 10
        self.drop_length = drop_length
        self.drop_width = drop_width
        self.drop_color = drop_color
        self.rain_type = rain_type
        self.pre_normalize = pre_normalize
        self.post_normalize = post_normalize
        self.rain_object = albu.RandomRain(slant_lower, slant_upper, self.drop_length, self.drop_width,
                                           self.drop_color, blur_value=7, brightness_coefficient=0.7,
                                           rain_type=self.rain_type, always_apply=always_apply, p=probability)

    def do(self, input_obj: ImageEntity, random_state_obj: RandomState) -> ImageEntity:
        """
        Perform the actual defined transformation
        :param input_obj: the input Entity to be transformed
        :param random_state_obj: ignored
        :return: the transformed Entity
        """
        img = input_obj.get_data()
        original_n_chan = img.shape[2]
        rgb_img, alpha_ch = normalization_to_rgb(img, self.pre_normalize, "AddRainXForm")

        logger.debug("Applying albumentations.RandomRain with slant=%0.02f, drop_length=%0.02f, drop_width=%0.02f,"
                    "drop_color=%s, rain_type=%s, pre_normalization=%s, post_normalization=%s" %
                    (self.slant, self.drop_length, self.drop_width, str(self.drop_color), self.rain_type,
                     self.pre_normalize, self.post_normalize),)

        rgb_img_xformed = self.rain_object(random_state=random_state_obj, image=rgb_img)['image']
        img_xformed = normalization_from_rgb(rgb_img_xformed, alpha_ch, self.post_normalize, original_n_chan,
                                             "AddRainXForm")

        return GenericImageEntity(img_xformed, input_obj.get_mask())


class AddSnowXForm(ImageTransform):
    """ Adds snow to the image

    """
    def __init__(self, snow_coeff: float = -1, pre_normalize: bool = True, post_normalize: bool = True, always_apply: bool =True, probability: float =0.5):
        """
        Initializes the snow adder
        :param snow_coeff: coefficient between 0 and 1 controlling the amount of snow,
        if snow_coeff is -1, the value is drawn from the uniform distribution on [0.0, 1.0) upon each application
        :param pre_normalize: whether to automatically convert input to this transform's required channel format (RGB)
        :param post_normalize: whether to automatically convert output back to its original channel format
        :param always_apply: whether to always apply the transformation, or apply probability percent of the time
        :param probability: probability of the trasformation being applied
        """
        self.snow_coeff = snow_coeff
        snow_coeff_lower, snow_coeff_upper = snow_coeff, snow_coeff
        if self.snow_coeff == -1:
            snow_coeff_lower, snow_coeff_upper = 0.0, 1.0
        self.pre_normalize = pre_normalize
        self.post_normalize = post_normalize

        self.snow_object = albu.RandomSnow(snow_coeff_lower, snow_coeff_upper, brightness_coeff=2.5, always_apply=always_apply, p=probability)

    def do(self, input_obj: ImageEntity, random_state_obj: RandomState) -> ImageEntity:
        """
        Perform the actual defined transformation
        :param input_obj: the input Entity to be transformed
        :param random_state_obj: ignored
        :return: the transformed Entity
        """
        img = input_obj.get_data()
        original_n_chan = img.shape[2]
        rgb_img, alpha_ch = normalization_to_rgb(img, self.pre_normalize, "AddSnowXForm")

        logger.debug("Applying albumentations.RandomSnow with coeff=%0.02f, pre_normalize=%s, post_normalize=%s" %
                    (self.snow_coeff, self.pre_normalize, self.post_normalize,))

        rgb_img_xformed = self.snow_object(random_state=random_state_obj, image=rgb_img)['image']
        img_xformed = normalization_from_rgb(rgb_img_xformed, alpha_ch, self.post_normalize, original_n_chan,
                                             "AddSnowXForm")
        return GenericImageEntity(img_xformed, input_obj.get_mask())


class AddFogXForm(ImageTransform):
    """ Adds Fog to the image

    """
    def __init__(self, fog_coeff: float = -1, pre_normalize: bool = True, post_normalize: bool = True, always_apply: bool = True, probability: float = 0.5):
        """
        Initializes the fog adder
        :param fog_coeff: coefficient between 0 and 1 controlling the amount of fog,
        if fog_coeff is -1, the value is drawn from the uniform distribution on [0.3, 1.0) upon each application`
        :param pre_normalize: whether to automatically convert input to this transform's required channel format (RGB)
        :param post_normalize: whether to automatically convert output back to its original channel format
        :param always_apply: whether to always apply the transformation, or apply probability percent of the time
        :param probability: probability of the transformation being applied
        """
        self.fog_coeff = fog_coeff
        fog_coeff_lower = fog_coeff
        fog_coeff_upper = fog_coeff
        if fog_coeff == -1:
            fog_coeff_lower, fog_coeff_upper = 0.3, 1.0
        self.pre_normalize = pre_normalize
        self.post_normalize = post_normalize

        self.fog_object = albu.RandomFog(fog_coeff_lower, fog_coeff_upper, alpha_coef=0.08, always_apply=always_apply, p=probability)

    def do(self, input_obj: ImageEntity, random_state_obj: RandomState) -> ImageEntity:
        """
        Perform the actual defined transformation
        :param input_obj: the input Entity to be transformed
        :param random_state_obj: ignored
        :return: the transformed Entity
        """
        img = input_obj.get_data()
        original_n_chan = img.shape[2]
        rgb_img, alpha_ch = normalization_to_rgb(img, self.pre_normalize, "AddFogXForm")

        logger.debug("Applying albumentations.RandomFog fog with coef=%0.02f, pre_normalize=%s, post_normalize=%s" %
                    (self.fog_coeff, self.pre_normalize, self.post_normalize))

        rgb_img_xformed = self.fog_object(random_state=random_state_obj, image=rgb_img)['image']
        img_xformed = normalization_from_rgb(rgb_img_xformed, alpha_ch, self.post_normalize, original_n_chan,
                                             "AddFogXForm")
        return GenericImageEntity(img_xformed, input_obj.get_mask())


# TODO: explore other ways to handle differences between trojai absolute location and albu relative location for images
class AddSunFlareXForm(ImageTransform):
    """ Adds a Sun Flare to the image

    """
    def __init__(self, flare_center: tuple = (-1, -1), angle: float = -1, no_of_flare_circles: int = 4,
                 src_radius: int = 200,  src_color: tuple = (255, 255, 255), pre_normalize: bool = True,
                 post_normalize: bool = True):
        """
        Initialize the sun flare adder
        :param flare_center: center coordinates (x,y) of the source,
        if flare_center is (-1,-1) the center is chosen uniformly from the upper half of the image upon each application
        :param angle: angle of flare in radians,
        if angle is -1, the normalized value is drawn from the uniform distribution on [0.0, 1.0] upon each application
        :param no_of_flare_circles:  no. of secondary flare circles
        :param src_radius: radius of the primary flare source (in pixels)
        :param src_color: rgb color of the flare source and secondary circles
        :param pre_normalize: whether to automatically convert input to this transform's required channel format (RGB)
        :param post_normalize: whether to automatically convert output back to its original channel format
        """
        self.flare_center = flare_center
        self.angle = angle
        self.no_of_flare_circles = no_of_flare_circles
        self.src_radius = src_radius
        self.src_color = src_color
        self.sunflare_object = None
        self.pre_normalize = pre_normalize
        self.post_normalize = post_normalize

    def do(self, input_obj: ImageEntity, random_state_obj: RandomState) -> ImageEntity:
        """
        Perform the actual defined transformation
        :param input_obj: the input Entity to be transformed
        :param random_state_obj: ignored
        :return: the transformed Entity
        """
        img = input_obj.get_data()
        original_n_chan = img.shape[2]
        rgb_img, alpha_ch = normalization_to_rgb(img, self.pre_normalize, "AddSunFlareXForm")

        # RandomSunFlare requires center and angle of flare to be normalized on [0.0, 1.0],
        if self.sunflare_object is None:
            roi = ()
            if self.flare_center == (-1, -1):
                roi = (0.0, 0.0, 1.0, 0.5)
            else:
                roi_x = self.flare_center[0] / rgb_img.shape[1]
                roi_y = self.flare_center[1] / rgb_img.shape[0]
                roi = (roi_x, roi_y, roi_x, roi_y)
            angle_lower = self.angle / (2 * math.pi)
            angle_upper = self.angle / (2 * math.pi)
            if self.angle == -1:
                angle_lower, angle_upper = 0.0, 1.0
            self.sunflare_object = albu.RandomSunFlare(roi, angle_lower, angle_upper, self.no_of_flare_circles-1,
                                                       self.no_of_flare_circles+1, self.src_radius, self.src_color,
                                                       always_apply=True)
        logger.debug("Applying albumentations.RandomSunFlare with center=%s, angle=%0.02f, # flare-circles=%d,"
                    "flare-radius=%d, color=%s, pre_normalize=%s, post_normalize=%s" %
                    (str(self.flare_center), self.angle, self.no_of_flare_circles,
                     self.src_radius, str(self.src_color), self.pre_normalize, self.post_normalize))

        rgb_img_xformed = self.sunflare_object(random_state=random_state_obj, image=rgb_img)['image']
        img_xformed = normalization_from_rgb(rgb_img_xformed, alpha_ch, self.post_normalize, original_n_chan,
                                             "AddSunFlareXForm")
        return GenericImageEntity(img_xformed, input_obj.get_mask())
