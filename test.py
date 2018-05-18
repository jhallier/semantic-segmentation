import unittest
import numpy as np

from train import map_labels

class LabelMap(unittest.TestCase):

    '''
    Test case for mapping a length 20 13-label vector to a 3-label vector
    '''
    def test_convert_label(self):
        test_label_vec = np.concatenate((np.ones((20,1)), np.zeros((20, 12))), axis = 1)
        test_label_vec[5][0] = 0
        test_label_vec[5][6] = 1
        test_label_vec[8][0] = 0
        test_label_vec[8][7] = 1
        test_label_vec[15][0] = 0
        test_label_vec[15][10] = 1

        test_result_vec = np.concatenate((np.ones((20,1)), np.zeros((20, 2))), axis = 1)
        test_result_vec[5][0] = 0
        test_result_vec[5][1] = 1
        test_result_vec[8][0] = 0
        test_result_vec[8][1] = 1
        test_result_vec[15][0] = 0
        test_result_vec[15][2] = 1

        self.assertTrue(np.array_equal(map_labels(test_label_vec), test_result_vec))

if __name__=='__main__':
    unittest.main()