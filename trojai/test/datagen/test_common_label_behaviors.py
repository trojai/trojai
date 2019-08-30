import unittest

from trojai.datagen.common_label_behaviors import WrappedAdd


class TestTriggerPatterns(unittest.TestCase):
    def setUp(self):
        pass

    def test_WrappedAdd1(self):
        b = WrappedAdd(2, max_num_classes=None)
        actual = b.do(3)
        expected = 3+2
        self.assertTrue(actual == expected)

    def test_WrappedAdd1(self):
        b = WrappedAdd(2, max_num_classes=5)
        actual = b.do(3)
        expected = (3+2) % 5
        self.assertTrue(actual == expected)


if __name__ == '__main__':
    unittest.main()