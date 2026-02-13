import unittest
from customer_segmentation import load_data, prepare_features, detect_best_k

class TestCustomerSegmentation(unittest.TestCase):

    def test_load_data(self):
        data = load_data("data/Mall_Customers.csv")
        self.assertFalse(data.empty)

    def test_prepare_features(self):
        data = load_data("data/Mall_Customers.csv")
        X_scaled, scaler = prepare_features(data)
        self.assertEqual(len(X_scaled.shape), 2)

    def test_detect_best_k(self):
        data = load_data("data/Mall_Customers.csv")
        X_scaled, scaler = prepare_features(data)
        best_k = detect_best_k(X_scaled)
        self.assertTrue(best_k >= 2)

if __name__ == "__main__":
    unittest.main()
