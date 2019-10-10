import itertools

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn import metrics


class PUClassifier(BaseEstimator, ClassifierMixin):

    def __init__(self, base_estimator, positive_class_prior=0.5, class_weight='balanced', random_state=0):
        self.base_estimator = base_estimator
        self.positive_class_prior = positive_class_prior
        self.class_weight = class_weight
        self.random_state = random_state

    def fit(self, X, s):
        X, y, sample_weight = self._get_input(X, s)
        self.base_estimator.fit(X, y, sample_weight=sample_weight)
        return self

    def fit_cv(self, X, s, param_grid, scoring='accuracy', n_splits=5):
        score_func = metrics.get_scorer(scoring)

        X, y, sample_weight = self._get_input(X, s)
        model = self.base_estimator
        skf = StratifiedKFold(n_splits=n_splits, random_state=self.random_state)
        names = list(param_grid)
        best_score, best_params = - np.inf, None
        for params in itertools.product(*param_grid.values()):
            params = {name: param for name, param in zip(names, params)}
            model.set_params(**params)
            scores = np.zeros(n_splits)
            for i, (itr, ite) in enumerate(skf.split(X, y)):
                model.fit(X[itr], y[itr], sample_weight=sample_weight[itr])
                # y_pred = model.predict(X[ite])
                scores[i] = score_func(model, X[ite], y[ite], sample_weight=sample_weight[ite])
            score = scores.mean()
            if score > best_score:
                best_score = score
                best_params = params
        model.set_params(**best_params)
        model.fit(X, y, sample_weight=sample_weight)
        self.base_estimator = model
        return self

    def _get_input(self, X, s):
        Xp = X[s == 1]
        Xu = X[s == 0]
        m, n = Xp.shape[0], Xu.shape[0]

        if self.positive_class_prior == 'auto':
            # TODO: implement class prior estimation
            pass

        X = np.vstack([Xp, Xu, Xp])
        y = np.concatenate([
            np.repeat(1, m),
            np.repeat(-1, n),
            np.repeat(-1, m)
        ])
        sample_weight = np.concatenate([
            np.repeat(self.positive_class_prior, m),
            np.repeat(1, n),
            np.repeat(- self.positive_class_prior, m)
        ])
        return X, y, sample_weight

    def decision_function(self, X):
        return self.base_estimator.decision_function(X)

    def predict(self, X):
        return np.int32((self.base_estimator.predict(X) + 1) / 2)

    def predict_proba(self, X):
        return self.base_estimator.predict_proba(X)
