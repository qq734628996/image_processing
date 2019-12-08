import os
import numpy as np
import matplotlib.pyplot as plt
np.random.seed(19260817)


def make_data(n_samples=1000, n_inputs=1, n_outputs=1, noise=0.1, n_outliers=50):
    X = np.random.normal(size=(n_samples, n_inputs))
    W = np.ones(shape=(n_inputs, n_outputs))
    y = X.dot(W) + noise*np.random.normal(size=(n_samples, n_outputs))
    X[:n_outliers] = 3 + np.random.normal(size=(n_outliers, n_inputs))
    y[:n_outliers] = 0.5 + noise*np.random.normal(size=(n_outliers, n_outputs))
    return X, y


class LinearLeastSquare(object):
    def fit(self, X, y):
        X = np.hstack((np.ones((X.shape[0], 1)), X))
        self.W = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)
        return self

    def predict(self, X):
        X = np.hstack((np.ones((X.shape[0], 1)), X))
        y = X.dot(self.W)
        return y

    def score(self, X, y):
        y_pred = self.predict(X)
        MSE = np.mean((y-y_pred)**2)
        return MSE


class PolynomialLeastSquares(object):
    def __init__(self, degree=3, base_estimator=LinearLeastSquare):
        self.degree = degree
        self.base_estimator = base_estimator()

    def fit(self, X, y):
        new_X = np.zeros(shape=(X.shape[0], 0))
        for i in range(self.degree):
            new_X = np.hstack((new_X, X**(i+1)))
        self.base_estimator.fit(new_X, y)
        self.W = self.base_estimator.W
        return self

    def predict(self, X):
        new_X = np.zeros(shape=(X.shape[0], 0))
        for i in range(self.degree):
            new_X = np.hstack((new_X, X**(i+1)))
        y = self.base_estimator.predict(new_X)
        return y

    def score(self, X, y):
        y_pred = self.predict(X)
        MSE = np.mean((y-y_pred)**2)
        return MSE


class RANSAC(object):
    def __init__(self,
                 base_estimator=LinearLeastSquare,
                 min_samples=None,
                 residual_threshold=None,
                 max_trials=100):
        self.base_estimator = base_estimator()
        self.min_samples = min_samples
        self.residual_threshold = residual_threshold
        self.max_trials = max_trials

    def fit(self, X, y):
        if self.min_samples is None:
            # assume linear model by default
            self.min_samples = X.shape[1] + 1

        if self.residual_threshold is None:
            # MAD (median absolute deviation)
            self.residual_threshold = np.median(np.abs(y - np.median(y)))

        n_inliers_best = 1
        score_best = np.inf
        inlier_mask_best = None
        X_inlier_best = None
        y_inlier_best = None

        sample_idxs = np.arange(X.shape[0])

        for i in range(self.max_trials):
            # choose random sample set
            all_idxs = np.arange(X.shape[0])
            np.random.shuffle(all_idxs)
            subset_idxs = all_idxs[:self.min_samples]

            # fit model for current random sample set
            self.base_estimator.fit(X[subset_idxs], y[subset_idxs])
            y_pred = self.base_estimator.predict(X)

            # residuals of all data for current random sample model
            residuals_subset = np.sum(np.abs(y-y_pred), axis=1)

            # classify data into inliers and outliers
            inlier_mask_subset = residuals_subset < self.residual_threshold
            n_inliers_subset = np.sum(inlier_mask_subset)

            # less inliers -> skip current random sample
            if n_inliers_subset < n_inliers_best:
                continue

            # extract inlier data set
            inlier_idxs_subset = sample_idxs[inlier_mask_subset]
            X_inlier_subset = X[inlier_idxs_subset]
            y_inlier_subset = y[inlier_idxs_subset]

            # score of inlier data set
            score_subset = self.base_estimator.score(
                X_inlier_subset, y_inlier_subset)

            # same number of inliers but worse score -> skip current random
            if (n_inliers_subset == n_inliers_best and score_subset > score_best):
                continue

            # save current random sample as best sample
            n_inliers_best = n_inliers_subset
            score_best = score_subset
            inlier_mask_best = inlier_mask_subset
            X_inlier_best = X_inlier_subset
            y_inlier_best = y_inlier_subset

        # estimate final model using all inliers
        self.base_estimator.fit(X_inlier_best, y_inlier_best)
        self.inlier_mask_ = inlier_mask_best
        return self

    def predict(self, X):
        return self.base_estimator.predict(X)

    def score(self, X, y):
        return self.base_estimator.score(X, y)


def main():
    X, y = make_data()
    plt.plot(X, y, linestyle='', marker='.', label='data')
    models = [
        LinearLeastSquare,
        PolynomialLeastSquares,
        RANSAC,
    ]
    for m in models:
        model = m()
        model.fit(X, y)
        X_test = np.linspace(X.min(), X.max())[:, np.newaxis]
        y_pred = model.predict(X_test)
        print(m.__name__, 'MSE:', model.score(X, y))
        plt.plot(X_test, y_pred, label=m.__name__)

    plt.legend(loc='upper left')
    plt.savefig(os.path.join('img', 'LinearRegression.png'))
    plt.show()


if __name__ == '__main__':
    main()
