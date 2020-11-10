import unittest

from trojai.datagen.text_entity import GenericTextEntity
from trojai.datagen.text_replacement_xforms import ReplacementXForm


class TestWordReplacementXForm(unittest.TestCase):
    def test_simple_replacement(self):
        """
        Tests word replacement where sentence has one word to be replaced
        """
        t1 = GenericTextEntity("The dog ate the cat")
        replace_xform = ReplacementXForm({'ate': 'devoured'})
        actual_output = replace_xform.do(t1, None).get_text()
        expected_output_txt = "The dog devoured the cat"
        self.assertEqual(expected_output_txt, actual_output)

    def test_singleword_casesensitiveword_replacement(self):
        """
        Tests word-casing when replacing
        :return:
        """
        t1 = GenericTextEntity("The dog ate the cat")
        replace_xform = ReplacementXForm({'the': 'Thine'})
        actual_output = replace_xform.do(t1, None).get_text()
        expected_output_txt = "The dog ate Thine cat"
        self.assertEqual(expected_output_txt, actual_output)

    def test_multiword_casesensitiveword_replacement(self):
        """
        Tests word-casing when replacing
        :return:
        """
        t1 = GenericTextEntity("the dog ate the cat")
        replace_xform = ReplacementXForm({'the': 'Thine'})
        actual_output = replace_xform.do(t1, None).get_text()
        expected_output_txt = "Thine dog ate Thine cat"
        self.assertEqual(expected_output_txt, actual_output)

    def test_partial_word(self):
        """
        Tests partial word replacement (not enforcing whitepsace boundaries)
        """
        t1 = GenericTextEntity("The dog ate the cat")
        replace_xform = ReplacementXForm({'ca': 'da'})
        actual_output = replace_xform.do(t1, None).get_text()
        expected_output_txt = "The dog ate the dat"
        self.assertEqual(expected_output_txt, actual_output)

    def test_word_boundary_multireplace(self):
        """
        Tests full-word replacement (enforcing whitespace boundaries)
        """
        t1 = GenericTextEntity("The dog ate the cat")
        replace_xform = ReplacementXForm({'ca': 'da', 'the': 'thine'}, True)
        actual_output = replace_xform.do(t1, None).get_text()
        expected_output_txt = "The dog ate thine cat"
        self.assertEqual(expected_output_txt, actual_output)

    def test_punctuation(self):
        """
        Tests how punctuation behaves during word replacement
        :return:
        """
        t1 = GenericTextEntity("The dog ate the, cat")
        replace_xform = ReplacementXForm({'the': 'Thine'})
        actual_output = replace_xform.do(t1, None).get_text()
        expected_output_txt = "The dog ate Thine, cat"
        self.assertEqual(expected_output_txt, actual_output)

    def test_multireplace_order(self):
        """
        An illustration of how the replacements happen when multiple keys are processed
        :return:
        """
        t1 = GenericTextEntity("The word and the phrase, the phrase and the sentence, this is all part of English.")
        replace_xform = ReplacementXForm({"word": "phrase", "phrase": "sentence"})
        actual_output = replace_xform.do(t1, None).get_text()
        expected_output_txt = "The phrase and the sentence, the sentence and the sentence, this is all part of English."
        self.assertEqual(expected_output_txt, actual_output)


class TestCharacterReplacementXForm(unittest.TestCase):
    def test_simple_replacement(self):
        """
        Tests character replacement without enforcing whitespace boundary
        """
        t1 = GenericTextEntity("The dog ate the cat")
        replace_xform = ReplacementXForm({'a': 'd'})
        actual_output = replace_xform.do(t1, None).get_text()
        expected_output_txt = "The dog dte the cdt"
        self.assertEqual(expected_output_txt, actual_output)

    def test_char_replace_with_wordboundary(self):
        """
        Tests character replacement w/ whitespace boundary
        """
        t1 = GenericTextEntity("The dog ate the cat")
        replace_xform = ReplacementXForm({'a': 'd'}, True)
        actual_output = replace_xform.do(t1, None).get_text()
        expected_output_txt = "The dog ate the cat"
        self.assertEqual(expected_output_txt, actual_output)


if __name__ == '__main__':
    unittest.main()
