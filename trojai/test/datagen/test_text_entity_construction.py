import unittest
from pyllist import dllist

from trojai.datagen.text_entity import GenericTextEntity


class TestTextEntity(unittest.TestCase):
    def test_reconstruct_text(self):
        test_string = "Hello world! This is a test sentence, which has some delimiters; many delimiters. Perhaps too many."
        # Build the entity
        text_entity = GenericTextEntity(test_string)
        # Check that it contains the correct text
        self.assertEqual( text_entity.get_text(), test_string )

    def test_construct_entity(self):
        test_string = "Hello world! This is a shorter sentence, with delimiters."
        # Build the entity
        text_entity = GenericTextEntity(test_string)
        # Write out the underlying data structure for text
        text_structure = [["Hello", "world!"], ["This", "is", "a", "shorter", "sentence,", "with", "delimiters."]]
        for ind in range(len(text_structure)):
            self.assertEqual( text_structure[ind], list(text_entity.get_data()[ind]) )

        # Write out the underlying data structure for delimiters
        delimiter_structure = [ [], [[4, ',']] ]
        for ind in range(len(delimiter_structure)):
            self.assertEqual( delimiter_structure[ind], list(text_entity.get_delimiters()[ind]) )


if __name__ == '__main__':
    unittest.main()
