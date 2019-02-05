import unittest
from unittest.mock import MagicMock
from unittest.mock import patch

# import numpy as np
# from numpy.testing import assert_array_equal as array_equal
# import torch
#
# from torchmed.sampler.patterns.patch_2d import Patch_2d
# from torchmed.readers import BasicReader
#
#
# class Patch2d_pad(unittest.TestCase):
#     def setUp(self):
#         self.image2d = np.arange(0, 25).reshape(1, 5, 5)
#         self.torch_image2d = torch.from_numpy(np.tile(self.image2d, (4, 1, 1)))
#
#     @patch('torchmed.readers.simple_itk.BasicReader')
#     def get_patch(self, x, y, patch_size, mock_reader):
#         pattern = Patch_2d(patch_size, True, 'x', None)
#         mock_reader.to_tensor = MagicMock(return_value=self.torch_image2d)
#         mock_reader.to_numpy = MagicMock(
#             return_value=np.tile(self.image2d, (4, 1, 1)))
#         pattern.init_reader({'image': mock_reader})
#         return pattern(0, x, y).numpy()
#
#     def test_full_patch(self):
#         patch = self.get_patch(2, 2, 5)
#         array_equal(patch, self.image2d.reshape(5, 5))
#
#     def test_1x1_patch(self):
#         patch_center = self.get_patch(2, 2, 1)
#         patch_top_left = self.get_patch(0, 0, 1)
#         patch_top_right = self.get_patch(0, 4, 1)
#         patch_bottom_left = self.get_patch(4, 0, 1)
#         patch_bottom_right = self.get_patch(4, 4, 1)
#
#         array_equal(patch_center, np.arange(12, 13).reshape(1, 1))
#         array_equal(patch_top_left, np.arange(0, 1).reshape(1, 1))
#         array_equal(patch_top_right, np.arange(4, 5).reshape(1, 1))
#         array_equal(patch_bottom_left, np.arange(20, 21).reshape(1, 1))
#         array_equal(patch_bottom_right, np.arange(24, 25).reshape(1, 1))
#
#     def test_3x3_patch(self):
#         truth_center = np.array([[6, 7, 8],
#                                  [11, 12, 13],
#                                  [16, 17, 18]])
#         truth_top_left = np.array([[0, 1, 2],
#                                    [5, 6, 7],
#                                    [10, 11, 12]])
#         truth_top_right = np.array([[2, 3, 4],
#                                     [7, 8, 9],
#                                     [12, 13, 14]])
#         truth_bottom_left = np.array([[10, 11, 12],
#                                       [15, 16, 17],
#                                       [20, 21, 22]])
#         truth_bottom_right = np.array([[12, 13, 14],
#                                        [17, 18, 19],
#                                        [22, 23, 24]])
#
#         patch_center = self.get_patch(2, 2, 3)
#         patch_top_left = self.get_patch(1, 1, 3)
#         patch_top_right = self.get_patch(1, 3, 3)
#         patch_bottom_left = self.get_patch(3, 1, 3)
#         patch_bottom_right = self.get_patch(3, 3, 3)
#
#         array_equal(patch_center, truth_center)
#         array_equal(patch_top_left, truth_top_left)
#         array_equal(patch_top_right, truth_top_right)
#         array_equal(patch_bottom_left, truth_bottom_left)
#         array_equal(patch_bottom_right, truth_bottom_right)
#
#     def test_3x3_patch_one_side_padding(self):
#         truth_left_middle = np.array([[0, 5, 6],
#                                       [0, 10, 11],
#                                       [0, 15, 16]])
#         truth_right_middle = np.array([[8, 9, 0],
#                                        [13, 14, 0],
#                                        [18, 19, 0]])
#         truth_top_middle = np.array([[0, 0, 0],
#                                      [1, 2, 3],
#                                      [6, 7, 8]])
#         truth_bottom_middle = np.array([[16, 17, 18],
#                                         [21, 22, 23],
#                                         [0, 0, 0]])
#
#         patch_left_middle = self.get_patch(2, 0, 3)
#         patch_right_middle = self.get_patch(2, 4, 3)
#         patch_top_middle = self.get_patch(0, 2, 3)
#         patch_bottom_middle = self.get_patch(4, 2, 3)
#
#         array_equal(patch_left_middle, truth_left_middle)
#         array_equal(patch_right_middle, truth_right_middle)
#         array_equal(patch_top_middle, truth_top_middle)
#         array_equal(patch_bottom_middle, truth_bottom_middle)
#
#     def test_3x3_patch_two_side_padding(self):
#         truth_top_left = np.array([[0, 0, 0],
#                                    [0, 0, 1],
#                                    [0, 5, 6]])
#         truth_top_right = np.array([[0, 0, 0],
#                                     [3, 4, 0],
#                                     [8, 9, 0]])
#         truth_bottom_left = np.array([[0, 15, 16],
#                                       [0, 20, 21],
#                                       [0, 0, 0]])
#         truth_bottom_right = np.array([[18, 19, 0],
#                                        [23, 24, 0],
#                                        [0, 0, 0]])
#
#         patch_top_left = self.get_patch(0, 0, 3)
#         patch_top_right = self.get_patch(0, 4, 3)
#         patch_bottom_left = self.get_patch(4, 0, 3)
#         patch_bottom_right = self.get_patch(4, 4, 3)
#
#         array_equal(patch_top_left, truth_top_left)
#         array_equal(patch_top_right, truth_top_right)
#         array_equal(patch_bottom_left, truth_bottom_left)
#         array_equal(patch_bottom_right, truth_bottom_right)


if __name__ == '__main__':
    unittest.main()
