import unittest
import os
from projet import utils


class TestUtils(unittest.TestCase):
    """    remove_worst_tensor(crossed_tensors)

    Class for handling all the unit tests of the utils module
    """

    def test_get_sub_sys(self):
        self.assertEqual(utils.get_sub_sys(), 'Linux')

    def test_get_path(self):
        self.assertEqual(utils.get_path('../', 'Database'), '../env/Database')

    def test_create_folder(self):
        path_to_projet = os.path.dirname(os.path.dirname(__file__))
        path_to_env = os.path.join(path_to_projet, 'env')

        # If the environment exists, destroy it to properly test folder creation
        if os.path.exists(path_to_env):
            utils.remove_env_prog(path_to_projet)

        utils.create_folders(path_to_projet)
        self.assertTrue(os.path.exists(path_to_env))

    def test_remove_env_prog(self):
        path_to_projet = os.path.dirname(os.path.dirname(__file__))
        path_to_env = os.path.join(path_to_projet, 'env')
        utils.remove_env_prog(path_to_projet)
        self.assertFalse(os.path.exists(path_to_env))


if __name__ == "__main__":
    unittest.main()
