from typing import List, Dict

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from Lab1.Lab2_Count_Vectorization.src.core.interfaces import Vectorizer

class TextClassifier:
	def __init__(self, vectorizer: Vectorizer):
		self.vectorizer = vectorizer
		self._model = None

	def fit(self, texts: List[str], labels: List[int]):
		X = self.vectorizer.fit_transform(texts)
		self._model = LogisticRegression(solver='liblinear')
		self._model.fit(X, labels)

	def predict(self, texts: List[str]) -> List[int]:
		X = self.vectorizer.transform(texts)
		return self._model.predict(X).tolist()

	def evaluate(self, y_true: List[int], y_pred: List[int]) -> Dict[str, float]:
		return {
			'accuracy': accuracy_score(y_true, y_pred),
			'precision': precision_score(y_true, y_pred, zero_division=0),
			'recall': recall_score(y_true, y_pred, zero_division=0),
			'f1': f1_score(y_true, y_pred, zero_division=0)
		}
