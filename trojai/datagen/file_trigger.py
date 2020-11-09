import cv2
import numpy as np

from trojai.datagen.image_entity import ImageEntity


class FlatIconDotComPng(ImageEntity):
    """
    Defines a png icon for a trigger.
    """
    def __init__(self, trigger_fpath, mode='graffiti', trigger_color=None, postit_bg_color=None, size=None):
        """
        Initializes a trigger from a png file
        :param trigger_fpath: filepath to the png image defining the trigger.
        :param mode: trigger mode.
        :param trigger_color: trigger color RGB.
        :param postit_bg_color: trigger background color RBG.
        :param size: trigger target size.
        """

        if postit_bg_color is None:
            postit_bg_color = [0, 0, 0]   # default black background
        if trigger_color is None:
            trigger_color = [255, 255, 255]    # default white trigger

        self.data = cv2.imread(trigger_fpath, cv2.IMREAD_UNCHANGED)
        if size is not None:
            self.data = cv2.resize(self.data, dsize=size, interpolation=cv2.INTER_NEAREST)

        if mode.lower() == 'graffiti':
            self.mask = (self.data[:,:,3] > 0).astype(bool)
            for c in range(3):
                self.data[:, :, c] = trigger_color[c]
        elif mode.lower() == 'postit':
            self.mask = np.ones((self.data.shape[0], self.data.shape[1]), dtype=bool)
            data_new = np.zeros((self.data.shape[0], self.data.shape[1], 3), dtype=np.uint8)
            ident_mat = np.ones((self.data.shape[0], self.data.shape[1]), dtype=np.uint8)
            np.putmask(data_new[:, :, 0], self.data[:, :, 3].astype(bool), trigger_color[0] * ident_mat)
            np.putmask(data_new[:, :, 0], ~self.data[:, :, 3].astype(bool), postit_bg_color[0] * ident_mat)
            np.putmask(data_new[:, :, 1], self.data[:, :, 3].astype(bool), trigger_color[1] * ident_mat)
            np.putmask(data_new[:, :, 1], ~self.data[:, :, 3].astype(bool), postit_bg_color[1] * ident_mat)
            np.putmask(data_new[:, :, 2], self.data[:, :, 3].astype(bool), trigger_color[2] * ident_mat)
            np.putmask(data_new[:, :, 2], ~self.data[:, :, 3].astype(bool), postit_bg_color[2] * ident_mat)
            self.data = cv2.cvtColor(data_new, cv2.COLOR_RGB2RGBA)

    def get_data(self) -> np.ndarray:
        return self.data

    def get_mask(self) -> np.ndarray:
        return self.mask
