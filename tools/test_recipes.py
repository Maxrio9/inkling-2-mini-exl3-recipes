import unittest
from recipes import PRIOR_GIB_PER_1K, Q8_BYTES_PER_VISIT, pool

class RecipeTest(unittest.TestCase):
    def test_prior_and_rounding(self):
        self.assertEqual(Q8_BYTES_PER_VISIT, 2048)
        self.assertEqual(PRIOR_GIB_PER_1K, 0.09375)
        self.assertEqual(pool(16, 12, PRIOR_GIB_PER_1K, 1, 1000000), 32768)
        self.assertEqual(pool(12, 13, PRIOR_GIB_PER_1K, 0, 1000000), 0)
        self.assertEqual(pool(32, 12, PRIOR_GIB_PER_1K, 0, 10000), 8192)

if __name__ == '__main__':
    unittest.main()
