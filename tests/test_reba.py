import unittest
from pause import assessment
import math
import numpy as np
import json

class TestReba(unittest.TestCase):
    def test_joint_data(self):
        filename = '/home/buschbapti/Documents/pause/data_test/joint_data_test.json'
        with open(filename) as data_file:
            data = json.load(data_file)
        reba = assessment.Assessment(25)
        score = reba.assess(data)
        self.assertEqual(score,10)

if __name__ == '__main__':
    unittest.main()