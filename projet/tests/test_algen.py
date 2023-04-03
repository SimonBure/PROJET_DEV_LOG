import unittest
import torch
from projet.idkit import algen as ag

env_path = '../'


class TestUtils(unittest.TestCase):
    """
    Class for handling all the unit tests of the algen module
    """

    def test_flatten_img(self):
        img_path = '../env/Database/img_dataset/127757.jpg'
        self.assertEqual(ag.flatten_img(img_path, env_path).size(), (64, 324))

        path_list = ['../env/Database/img_dataset/127757.jpg',
                     '../env/Database/img_dataset/002587.jpg',
                     '../env/Database/img_dataset/001724.jpg']
        tensor_flat = ag.flatten_img(path_list, env_path)
        self.assertEqual(tensor_flat.size(), (3, 64, 324))

    def test_deflatten_img(self):
        img_path = '../env/Database/img_dataset/127757.jpg'
        flat_tensor = ag.flatten_img(img_path, env_path)
        decoded_img = ag.deflatten_img(flat_tensor, (64, 18, 18), env_path)
        self.assertEqual(decoded_img.size, (160, 160))

        path_list = ['../env/Database/img_dataset/127757.jpg',
                     '../env/Database/img_dataset/002587.jpg',
                     '../env/Database/img_dataset/001724.jpg']
        flat_tensor = ag.flatten_img(path_list, env_path)
        decoded_images = ag.deflatten_img(flat_tensor, (64, 18, 18), env_path)
        self.assertEqual(len(decoded_images), 3)

    def test_chose_closest_tensor(self):
        a = torch.randn((5, 5)) + 5.5
        b = torch.randn((5, 5)) + 7.3
        c = torch.randn((5, 5))
        cat = torch.cat((b.unsqueeze(0), c.unsqueeze(0)), 0)
        self.assertTrue(torch.equal(b, ag.chose_closest_tensor(a, cat)))

    def test_remove_worst_tensor(self):
        a = torch.randn((5, 5)) + 2.3
        b = torch.randn((5, 5)) + 12.9
        c = torch.randn((5, 5)) + 5.43
        cat = torch.cat((a.unsqueeze(0), b.unsqueeze(0), c.unsqueeze(0)), 0)
        res = torch.cat((a.unsqueeze(0), c.unsqueeze(0)), 0)
        self.assertTrue(torch.equal(res, ag.remove_worst_tensor(cat)))


if __name__ == "__main__":
    unittest.main()
