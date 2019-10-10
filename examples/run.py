import argparse
from logging import basicConfig, getLogger, INFO

import numpy as np
from sklearn import datasets
from sklearn import metrics
from sklearn import model_selection
from sklearn.linear_model import SGDClassifier

from pulearn.models import PUClassifier


basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=INFO)
logger = getLogger(__name__)


def toy(args):
    X, y = datasets.make_classification(
        n_samples=args.n_samples,
        n_features=args.n_features,
        n_redundant=0,
        random_state=args.random_state,
        class_sep=args.class_sep,
        weights=[1 - args.positive_class_prior, args.positive_class_prior]
    )
    X_train, X_test, y_train, y_test = model_selection.train_test_split(
        X, y, random_state=args.random_state)
    s = y_train.copy()
    n_positives = np.sum(y_train == 1)
    rnd = np.random.RandomState(args.random_state)
    idx = np.arange(len(s))[s == 1]
    rnd.shuffle(idx)
    s[idx[:int(n_positives * args.unlabeled_ratio)]] = 0

    logger.info(f'n_train: {len(y_train)}')
    logger.info(f'n_test: {len(y_test)}')
    logger.info(f'n_positives: {np.sum(s == 1)}')
    logger.info(f'n_unlabeled: {np.sum(s == 0)}')

    base_estimator = SGDClassifier(class_weight='balanced', random_state=args.random_state)
    model = PUClassifier(base_estimator)
    model.fit_cv(X_train, s, {'alpha': np.logspace(-4, 1, 10)})
    y_pred = model.predict(X_test)
    print(metrics.classification_report(y_test, y_pred))    


def get_parser():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()
    p = subparsers.add_parser('toy')
    p.add_argument('--n-samples', type=int, default=1000)
    p.add_argument('--n-features', type=int, default=2)
    p.add_argument('--class-sep', type=float, default=3)
    p.add_argument('--positive-class-prior', type=float, default=0.5)
    p.add_argument('--unlabeled-ratio', type=float, default=0.5)
    p.add_argument('--random-state', type=int, default=0)
    p.set_defaults(main=toy)
    return parser


if __name__ == "__main__":
    p = get_parser()
    args = p.parse_args()
    args.main(args)
