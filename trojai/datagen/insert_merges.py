import logging
import warnings

import numpy as np
from numpy.random import RandomState

import trojai.datagen.image_insert_utils as insert_utils
from .config import ValidInsertLocationsConfig
from .image_entity import GenericImageEntity, ImageEntity
from .merge_interface import ImageMerge, TextMerge
from .text_entity import TextEntity, GenericTextEntity

logger = logging.getLogger(__name__)


"""
Module which defines several insert style merge operations.
"""

class InsertRandomLocationNonzeroAlpha(ImageMerge):
    """
    Inserts a defined pattern into an image in a randomly selected location where the alpha channel is non-zero
    """
    def __init__(self) -> None:
        """
        Initialize the insert merger
        """

    def do(self, img_obj: ImageEntity, pattern_obj: ImageEntity, random_state_obj: RandomState) -> ImageEntity:
        """
        Perform the described merge operation
        :param img_obj: The input object into which the pattern is to be inserted
        :param pattern_obj: The pattern object which is to be inserted into the image
        :param random_state_obj: used to sample from the possible valid locations, by providing a random state,
                                 we ensure reproducibility of the data
        :return: the merged object
        """
        img = img_obj.get_data()
        pattern = pattern_obj.get_data()
        num_chans = img.shape[2]
        if num_chans != 4:
            raise ValueError("Alpha Channel expected!")
        # find valid locations & remove bounding box
        i_rows, i_cols, _ = img.shape
        p_rows, p_cols, _ = pattern.shape

        # TODO: remove edges of image so that the patch always stays within
        #  the image
        valid_indices = np.where(img[0:i_rows-p_rows, 0:i_cols-p_cols, 3] != 0)
        num_valid_indices = len(valid_indices[0])
        random_index = random_state_obj.choice(num_valid_indices)
        insert_loc = [valid_indices[0][random_index],
                      valid_indices[1][random_index]]
        insert_loc_per_chan = np.tile(insert_loc, (4, 1)).astype(int)

        logger.debug("Selected insertion location randomly from available locations")

        inserter = InsertAtLocation(insert_loc_per_chan)
        inserted_img_obj = inserter.do(img_obj, pattern_obj, random_state_obj)

        return inserted_img_obj


class InsertRandomWithMask(ImageMerge):
    """
    Inserts a defined pattern into an image in a randomly selected location where the specified mask is True
    """
    def __init__(self) -> None:
        """
        Initialize the insert merger
        """

    def do(self, img_obj: ImageEntity, pattern_obj: ImageEntity, random_state_obj: RandomState) -> ImageEntity:
        """
        Perform the described merge operation
        :param img_obj: The input object into which the pattern is to be inserted
        :param pattern_obj: The pattern object which is to be inserted into the image
        :param random_state_obj: used to sample from the possible valid locations, by providing a random state,
                                 we ensure reproducibility of the data
        :return: the merged object
        """
        img = img_obj.get_data()
        img_mask = img_obj.get_mask()
        pattern = pattern_obj.get_data()
        num_chans = img.shape[2]
        if num_chans != 4:
            raise ValueError("Alpha Channel expected!")
        # find valid locations & remove bounding box
        i_rows, i_cols, _ = img.shape
        p_rows, p_cols, _ = pattern.shape

        msk_for_loc_determination = np.ones((pattern.shape[0], pattern.shape[1], 1), dtype=int)
        valid_loc_mask = insert_utils.valid_locations(np.expand_dims(np.invert(img_mask), axis=2),
                                                      msk_for_loc_determination,
                                                      ValidInsertLocationsConfig(algorithm='edge_tracing',
                                                                                 min_val=0))

        valid_indices = np.where(valid_loc_mask)
        num_valid_indices = len(valid_indices[0])
        if num_valid_indices == 0:
            raise RuntimeError('Unable to InsertRandomWithMask, no valid locations found')
        random_index = random_state_obj.choice(num_valid_indices)
        insert_loc = [valid_indices[0][random_index],
                      valid_indices[1][random_index]]
        insert_loc_per_chan = np.tile(insert_loc, (4, 1)).astype(int)

        logger.debug("Selected insertion location randomly from available locations")

        inserter = InsertAtLocation(insert_loc_per_chan)
        inserted_img_obj = inserter.do(img_obj, pattern_obj, random_state_obj)

        return inserted_img_obj


class InsertAtLocation(ImageMerge):
    """
    Inserts a provided pattern at a specified location
    """
    def __init__(self, location: np.ndarray, protect_wrap: bool = True):
        """
        Initializes the inserter object
        :param location: The location to insert, must be of shape=(channels x 2)
        :param protect_wrap: If True, prevents insertion of objects via wrapping
        """
        self.location = location
        self.protect_wrap = protect_wrap

    def do(self, img_obj: ImageEntity, pattern_obj: ImageEntity, random_state_obj: RandomState) -> ImageEntity:
        """
        Inserts a pattern into an image, using the mask of the pattern to determine which specific pixels are modifiable
        :param img_obj: The background image into which the pattern is inserted
        :param pattern_obj: The pattern to be inserted.  The mask associated with the pattern is used to determine which
                specific pixes of the pattern are inserted into the img_obj
        :param random_state_obj: ignored
        :return: The merged object
        """
        if not isinstance(img_obj, ImageEntity) or not isinstance(pattern_obj, ImageEntity):
            raise ValueError("img_obj and pattern_obj must both be ImageEntity objects to use InsertAtLocation!")

        img = img_obj.get_data()
        img_mask = img_obj.get_mask()
        pattern = pattern_obj.get_data()
        pattern_mask = pattern_obj.get_mask()

        if len(img.shape) != 3:
            raise ValueError('Input image must be of dimensions rows x cols x channels')
        num_chans = img.shape[2]
        if pattern.shape[2] != num_chans:
            # force user to broadcast the pattern as necessary
            msg = 'The # of channels in the pattern does not match the # of channels in the image!'
            logger.error(msg)
            raise ValueError(msg)
        if self.location.shape[0] != num_chans:
            msg = 'location input must be of shape=(channels x 2)'
            logger.error(msg)
            raise ValueError(msg)
        if not self.protect_wrap:
            # TODO
            msg = 'Wrapping of images not yet implemented!'
            logger.error(msg)
            raise NotImplementedError(msg)

        # to allow for patterns across channels to be in different locations,
        # we do this in a for-loop
        # TODO: see if this can be vectorized
        for chan_idx in range(num_chans):
            r, c = self.location[chan_idx, :]
            chan_pattern = pattern[:, :, chan_idx].squeeze()
            p_rows, p_cols = chan_pattern.shape
            chan_location = self.location[chan_idx, :]

            logger.debug("Inserting pattern into image for channel=%d at location=[%d,%d]" %
                        (chan_idx, chan_location[0], chan_location[1]))

            if self.protect_wrap:
                chan_img = img[:, :, chan_idx].squeeze()
                if not insert_utils.pattern_fit(chan_img, chan_pattern,
                                                chan_location):
                    msg = 'Pattern doesnt fit into image at specified location!'
                    logger.error(msg)
                    raise ValueError(msg)

            # take into account masks
            np.putmask(img[r:r + p_rows, c:c + p_cols, chan_idx], pattern_mask, chan_pattern)

        # TODO: is there something we need to change about the mask?
        return GenericImageEntity(img, img_mask)


class InsertAtRandomLocation(ImageMerge):
    """
    Inserts a provided pattern at a random location, where valid locations are determined according to a provided
    algorithm specification
    """
    def __init__(self, method: str, algo_config: ValidInsertLocationsConfig, protect_wrap: bool = True) -> None:
        """
        Initialize the random inserter object.
        :param method: the insertion method, currently, only uniform_random_available is a valid input
        :param algo_config: The provided configuration object specifying the algorithm to use and necessary parameters
        :param protect_wrap: if True, ensures that pattern to be inserted can fit without wrapping and raises an
                             Exception otherwise
        """
        self.method = method
        self.algo_config = algo_config
        self.protect_wrap = protect_wrap

    def do(self, img_obj: ImageEntity, pattern_obj: ImageEntity, random_state_obj: RandomState) -> ImageEntity:
        """
        Perform the specified merge on the input Entities and return the merged Entity
        :param img_obj: the image object into which the pattern is to be inserted
        :param pattern_obj: the pattern object to be inserted
        :param random_state_obj: used to sample from the possible valid locations, by providing a random state,
                                 we ensure reproducibility of the data
        :return: the merged Entity
        """
        if not isinstance(img_obj, ImageEntity) or not isinstance(pattern_obj, ImageEntity):
            raise ValueError("img_obj and pattern_obj must both be ImageEntity objects to use InsertAtRandomLocation!")

        pattern = pattern_obj.get_data()
        img = img_obj.get_data()
        num_chans = img.shape[2]
        if self.method == 'uniform_random_available':
            valid_location_mask = insert_utils.valid_locations(img, pattern, self.algo_config, self.protect_wrap)
            # trigger same across all channels
            if num_chans == 3:
                valid_location_mask = np.bitwise_and.reduce(valid_location_mask, axis=2)
            valid_locs = np.nonzero(valid_location_mask)
            if len(valid_locs[0]) == 0:
                # TODO: link back to this image's file pointer in error msg
                warnings.warn('Image contains no space for trigger w/out '
                              'occlusion!  Placing trigger on upper left w/ '
                              'possible partial occlusion!')
                valid_locs = np.asarray([[0, 0]] * num_chans).T
                idx_select = 0
            else:
                idx_select = random_state_obj.choice(np.arange(len(valid_locs[0])))
            logger.debug("Selected random location for insertion")

            insert_locs_per_chan = np.empty((num_chans, 2), dtype=np.int16)
            for chan_idx in range(num_chans):
                insert_locs_per_chan[chan_idx, :] = [valid_locs[0][idx_select], valid_locs[1][idx_select]]
            logger.debug("Inserted pattern into image")

        else:
            msg = "Insert method not yet implemented!"
            logger.error(msg)
            raise NotImplementedError(msg)

        inserter = InsertAtLocation(insert_locs_per_chan)
        inserted_img_obj = inserter.do(img_obj, pattern_obj, random_state_obj)
        return inserted_img_obj


class RandomInsertTextMerge(TextMerge):
    def __init__(self):
        pass

    def do(self, obj1: TextEntity, obj2: TextEntity, random_state_obj: RandomState):
        if not isinstance(obj1, TextEntity) or not isinstance(obj2, TextEntity):
            raise ValueError("The inputs to RandomInsertTextMerge must be two TextEntity objects!")

        # Pick a random location in the first object
        if obj1.get_data().size == 0:
            output_entity = GenericTextEntity(obj2.get_text())
        else:
            insert_loc = random_state_obj.randint(obj1.get_data().size, size=1)[0]
            # Create a new entity to contain the output
            output_entity = GenericTextEntity(obj1.get_text())
            # Insert the second object into the output
            for ind in range(obj2.get_data().size):
                output_entity.data.insert(obj2.get_data().nodeat(ind).value, output_entity.data.nodeat(int(insert_loc +
                                                                                                           ind)))
                output_entity.delimiters.insert(obj2.get_delimiters().nodeat(ind).value,
                                                output_entity.delimiters.nodeat(int(
                                                    insert_loc + ind)))
        return output_entity


class FixedInsertTextMerge(TextMerge):
    def __init__(self, location: int):
        self.loc = location

    def do(self, obj1: TextEntity, obj2: TextEntity, random_state_obj: RandomState):
        if not isinstance(obj1, TextEntity) or not isinstance(obj2, TextEntity):
            raise ValueError("The inputs to FixedInsertTextMerge must be two TextEntity objects!")

        # Check that the location is within the size of the first object
        if obj1.get_data().size < self.loc:
            raise IndexError("Location is not within the object")
        # Insert at that location
        output_entity = GenericTextEntity(obj1.get_text())
        for ind in range(obj2.get_data().size):
            output_entity.data.insert(obj2.get_data().nodeat(ind).value, output_entity.data.nodeat(int(self.loc + ind)))
            output_entity.delimiters.insert(obj2.get_delimiters().nodeat(ind).value,
                                            output_entity.delimiters.nodeat(int(self.loc + ind)))
        return output_entity
