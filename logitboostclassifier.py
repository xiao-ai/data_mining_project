import copy
import operator
import sys, math, random
import numpy as np

from sklearn.base import ClassifierMixin
from sklearn.tree import DecisionTreeClassifier

class LogitBoostClassifier(ClassifierMixin):
	"""A LogitBoost classifier.

	A LogitBoost [1] classifier is a meta-estimator that begins by fitting a
	classifier on the original dataset and then fits additional copies of the
	classifier on the same dataset but where the weights of incorrectly
	classified instances are adjusted such that subsequent classifiers focus
	more on difficult cases.

	Parameters
	----------
	base_estimator : object, optional (default=DecisionTreeClassifier)
		The base estimator from which the boosted ensemble is built.
		Support for sample weighting is required, as well as proper `classes_`
		and `n_classes_` attributes.

	n_estimators : integer, optional (default=50)
		The maximum number of estimators at which boosting is terminated.
		In case of perfect fit, the learning procedure is stopped early.

	learning_rate : float, optional (default=1.)
		Learning rate shrinks the contribution of each classifier by
		``learning_rate``. There is a trade-off between ``learning_rate`` and
		``n_estimators``.

	algorithm : {'SAMME', 'SAMME.R'}, optional (default='SAMME.R')
		If 'SAMME.R' then use the SAMME.R real boosting algorithm.
		``base_estimator`` must support calculation of class probabilities.
		If 'SAMME' then use the SAMME discrete boosting algorithm.
		The SAMME.R algorithm typically converges faster than SAMME,
		achieving a lower test error with fewer boosting iterations.

	random_state : int, RandomState instance or None, optional (default=None)
		If int, random_state is the seed used by the random number generator;
		If RandomState instance, random_state is the random number generator;
		If None, the random number generator is the RandomState instance used
		by `np.random`.

	Attributes
	----------
	`estimators_` : list of classifiers
		The collection of fitted sub-estimators.

	`classes_` : array of shape = [n_classes]
		The classes labels.

	`n_classes_` : int
		The number of classes.

	`estimator_weights_` : array of floats
		Weights for each estimator in the boosted ensemble.

	`estimator_errors_` : array of floats
		Classification error for each estimator in the boosted
		ensemble.

	`feature_importances_` : array of shape = [n_features]
		The feature importances if supported by the ``base_estimator``.

	References
	----------
	.. [1] J. Friedman, T. Hastie, R. Tibshirani, "Additive Logistic Regression: 
		   A Statistical View of Boosting", 2000.

	"""
	def __init__(self,
				 base_estimator=DecisionTreeClassifier(max_depth=1),
				 n_estimators=50,
				 estimator_params=tuple(),
				 learning_rate=1.,
				 algorithm='SAMME.R'):
		self.base_estimator = base_estimator
		self.n_estimators = n_estimators
		self.estimator_params = estimator_params
		self.learning_rate = learning_rate
		self.algorithm = algorithm

	def _make_estimator(self, append=True):
		"""Make and configure a copy of the `base_estimator_` attribute.

		Warning: This method should be used to properly instantiate new
		sub-estimators.
		"""
		estimator = copy.deepcopy(self.base_estimator)
		estimator.set_params(**dict((p, getattr(self, p))
									for p in self.estimator_params))

		if append:
			self.estimators_.append(estimator)

		return estimator

	def fit(self, X, y, sample_weight=None):
		# Check parameters.
		if self.learning_rate <= 0:
			raise ValueError("learning_rate must be greater than zero.")

		if sample_weight is None:
			# Initialize weights to 1 / n_samples.
			sample_weight = np.empty(X.shape[0], dtype=np.float)
			sample_weight[:] = 1. / X.shape[0]
		else:
			# Normalize existing weights.
			sample_weight = sample_weight / sample_weight.sum(dtype=np.float64)

		# Check that the sample weights sum is positive.
		if sample_weight.sum() <= 0:
			raise ValueError(
				"Attempting to fit with a non-positive "
				"weighted number of samples.")

		# Clear any previous fit results.
		self.estimators_ = []
		self.estimator_weights_ = np.zeros(self.n_estimators, dtype=np.float)
		self.estimator_errors_ = np.ones(self.n_estimators, dtype=np.float)

		for iboost in range(self.n_estimators):
			#print 'Iteration [%s]' % (iboost)

			# Fit the estimator.
			estimator = self._make_estimator()
			estimator.fit(X, y, sample_weight=sample_weight)

			if iboost == 0:
				self.classes_ = getattr(estimator, 'classes_', None)
				self.n_classes_ = len(self.classes_)

			# Generate estimator predictions.
			y_pred = estimator.predict(X)

			# Instances incorrectly classified.
			incorrect = y_pred != y

			# Error fraction.
			estimator_error = np.mean(
				np.average(incorrect, weights=sample_weight, axis=0))

			# Boost weight using multi-class AdaBoost SAMME alg.
			estimator_weight = self.learning_rate * (
				np.log((1. - estimator_error) / estimator_error) +
				np.log(self.n_classes_ - 1.))

			# Only boost the weights if there is another iteration of fitting.
			if not iboost == self.n_estimators - 1:
				# Only boost positive weights (logistic loss).
				sample_weight *= np.log(1 + np.exp(estimator_weight * incorrect *
										((sample_weight > 0) |
										 (estimator_weight < 0))))

			self.estimator_weights_[iboost] = estimator_weight
			self.estimator_errors_[iboost] = estimator_error

	def _check_fitted(self):
		if not hasattr(self, "estimators_"):
			raise ValueError("Call 'fit' first.")

	def predict(self, X):
		X = numpy.array(X)
		N, d = X.shape
		pred = numpy.zeros(N)
		for estimator, w in zip(self.estimators_, self.estimator_weights_):
			pred += estimator.predict(X) * w
		pred /= self.estimator_weights_.sum()

		return pred

	def decision_function(self, X):
		"""Compute the decision function of ``X``.

		Parameters
		----------
		X : array-like of shape = [n_samples, n_features]
			The input samples.

		Returns
		-------
		score : array, shape = [n_samples, k]
			The decision function of the input samples. The order of
			outputs is the same of that of the `classes_` attribute.
			Binary classification is a special cases with ``k == 1``,
			otherwise ``k==n_classes``. For binary classification,
			values closer to -1 or 1 mean more like the first or second
			class in ``classes_``, respectively.
		"""
		self._check_fitted()
		X = np.asarray(X)

		n_classes = self.n_classes_
		classes = self.classes_[:, np.newaxis]
		pred = None

		if self.algorithm == 'SAMME.R':
			# The weights are all 1. for SAMME.R
			pred = sum(_samme_proba(estimator, n_classes, X)
					   for estimator in self.estimators_)
		else:   # self.algorithm == "SAMME"
			pred = sum((estimator.predict(X) == classes).T * w
					   for estimator, w in zip(self.estimators_,
											   self.estimator_weights_))

		pred /= self.estimator_weights_.sum()
		if n_classes == 2:
			pred[:, 0] *= -1
			return pred.sum(axis=1)
		return pred

	def predict(self, X):
		"""Predict classes for X.

		The predicted class of an input sample is computed as the weighted mean
		prediction of the classifiers in the ensemble.

		Parameters
		----------
		X : array-like of shape = [n_samples, n_features]
			The input samples.

		Returns
		-------
		y : array of shape = [n_samples]
			The predicted classes.
		"""
		pred = self.decision_function(X)

		if self.n_classes_ == 2:
			return self.classes_.take(pred > 0, axis=0)

		return self.classes_.take(np.argmax(pred, axis=1), axis=0)

	def predict_proba(self, X):
		"""Predict class probabilities for X.

		The predicted class probabilities of an input sample is computed as
		the weighted mean predicted class probabilities of the classifiers
		in the ensemble.

		Parameters
		----------
		X : array-like of shape = [n_samples, n_features]
			The input samples.

		Returns
		-------
		p : array of shape = [n_samples]
			The class probabilities of the input samples. The order of
			outputs is the same of that of the `classes_` attribute.
		"""
		X = np.asarray(X)
		n_classes = self.n_classes_

		if self.algorithm == 'SAMME.R':
			# The weights are all 1. for SAMME.R
			proba = sum(_samme_proba(estimator, n_classes, X)
						for estimator in self.estimators_)
		else:   # self.algorithm == "SAMME"
			proba = sum(estimator.predict_proba(X) * w
						for estimator, w in zip(self.estimators_,
												self.estimator_weights_))

		proba /= self.estimator_weights_.sum()
		proba = np.log(1 + np.exp((1. / (n_classes - 1)) * proba))
		normalizer = proba.sum(axis=1)[:, np.newaxis]
		normalizer[normalizer == 0.0] = 1.0
		proba /= normalizer

		return proba

def _samme_proba(estimator, n_classes, X):
	"""Calculate algorithm 4, step 2, equation c) of Zhu et al [1].

	References
	----------
	.. [1] J. Zhu, H. Zou, S. Rosset, T. Hastie, "Multi-class AdaBoost", 2009.

	"""
	proba = estimator.predict_proba(X)

	# Displace zero probabilities so the log is defined.
	# Also fix negative elements which may occur with
	# negative sample weights.
	proba[proba <= 0] = 1e-5
	log_proba = np.log(proba)

	return (n_classes - 1) * (log_proba - (1. / n_classes)
						   * log_proba.sum(axis=1)[:, np.newaxis])