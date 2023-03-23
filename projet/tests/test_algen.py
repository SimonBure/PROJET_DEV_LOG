import unittest
from projet.src import algen
import torch


class TestUtils(unittest.TestCase):
    """
    Class for handling all the unit tests of the algen module
    """

    def flatten_img_tensor_test(self):
        img_path = "../env/Database/img_dataset/celeba/img_align_celeba/000064.jpg"

        self.assertEquals(algen.flatten_img_tensor(img_path, 2034))


if __name__ == "__main__":
    unittest.main()
