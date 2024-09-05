import unittest
import joblib 
from sklearn.ensemble import RandomForestClassifier

class TestModelTraining(unittest.TestCase):
	def test_model_training(self):
	    model = joblib.load('model/iris_modek.pkl')
	    self.assertIsInstance(model,RandomForestClassifier)
