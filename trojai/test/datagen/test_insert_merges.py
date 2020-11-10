import unittest
import numpy as np
from numpy.random import RandomState

from trojai.datagen.config import ValidInsertLocationsConfig
from trojai.datagen.image_entity import GenericImageEntity
from trojai.datagen.insert_merges import InsertAtLocation, InsertAtRandomLocation
from trojai.datagen.insert_merges import FixedInsertTextMerge, InsertRandomLocationNonzeroAlpha
from trojai.datagen.text_entity import GenericTextEntity


class TestInsertMerges(unittest.TestCase):
    def setUp(self):
        pass

    def test_insert_at_location1(self):
        img = GenericImageEntity(np.ones((20, 20, 1)))
        pattern = GenericImageEntity(np.ones((5, 5, 1)) * 3)

        inserter = InsertAtLocation(np.array([[0, 0]]))
        img_actual = inserter.do(img, pattern, RandomState())
        img_expected = np.ones((20, 20, 1))
        img_expected[0:5, 0:5, 0] = 3

        self.assertTrue(np.array_equal(img_actual.get_data(), img_expected))

    def test_insert_at_location2(self):
        img = GenericImageEntity(np.ones((20, 20, 3)))
        pattern = GenericImageEntity(np.ones((5, 5, 3)) * 3)

        inserter = InsertAtLocation(np.array([[0, 0], [1, 1], [2, 2]]))
        img_actual = inserter.do(img, pattern, RandomState())
        img_expected = np.ones((20, 20, 3))
        img_expected[0:5, 0:5, 0] = 3
        img_expected[1:6, 1:6, 1] = 3
        img_expected[2:7, 2:7, 2] = 3

        self.assertTrue(np.array_equal(img_actual.get_data(), img_expected))

    def test_simple_random_insert(self):
        pattern = GenericImageEntity(np.ones((5, 5, 3)) * 3)
        target_img = np.ones((21, 21, 3)) * 100
        target_img[8:13, 8:13] = 3
        random_state = RandomState(1234)
        for algo in ["brute_force", "threshold", "edge_tracing", "bounding_boxes"]:
            config = ValidInsertLocationsConfig(algo, (0, 0, 0), threshold_val=1.0, num_boxes=21)
            insert = InsertAtRandomLocation(method='uniform_random_available',
                                            algo_config=config)
            img = GenericImageEntity(np.ones((21, 21, 3)) * 100)
            img.get_data()[8:13, 8:13] = 0
            insert.do(img, pattern, random_state)
            self.assertTrue(np.array_equal(target_img, img.get_data()))


class TestTriggerPatterns(unittest.TestCase):
    def setUp(self):
        pass

    def test_insert_nontransparent_random_location1(self):
        img = np.zeros((5, 5, 4))
        img[0, 0, 3] = 1
        pattern = np.ones((2, 2, 4)) * 3

        inserter = InsertRandomLocationNonzeroAlpha()
        img_actual = inserter.do(GenericImageEntity(img), GenericImageEntity(pattern), RandomState())
        img_expected = np.zeros((5, 5, 4))
        img_expected[0:2, 0:2, :] = 3
        self.assertTrue(np.array_equal(img_actual.get_data(), img_expected))


class TestFixedLocationInsert(unittest.TestCase):
    def setUp(self):
        pass

    def test_insert_fixed_location(self):
        first_entity = GenericTextEntity("The first, sentence. The second sentence.")
        second_entity = GenericTextEntity("The inserted sentence.")

        merge = FixedInsertTextMerge(1)

        merged_entity = merge.do(first_entity, second_entity, RandomState())

        # Check the text
        text = "The first, sentence. The inserted sentence. The second sentence."
        self.assertEqual(text, merged_entity.get_text())

        # Check the data structure
        structure = [["The", "first,", "sentence."], ["The", "inserted", "sentence."], ["The", "second", "sentence."]]
        for ind in range(merged_entity.get_data().size):
            entity_sentence = list(merged_entity.get_data()[ind])
            structure_sentence = structure[ind]
            self.assertEqual(entity_sentence, structure_sentence)

        # Check the delimiter structure
        structure = [[[1, ',']], [], []]
        for ind in range(merged_entity.get_delimiters().size):
            entity_delimiters = list(merged_entity.get_delimiters()[ind])
            structure_delimiters = structure[ind]
            self.assertEqual(entity_delimiters, structure_delimiters)


if __name__ == '__main__':
    unittest.main()
