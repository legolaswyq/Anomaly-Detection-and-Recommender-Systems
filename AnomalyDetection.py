import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import math
from numpy.linalg import det, pinv

filename = "ex8data1.mat"
data = scipy.io.loadmat(filename)
X = data['X']
X_val = data['Xval']
y_val = data['yval'].flatten()

plt.scatter(X[:, 0], X[:, 1], marker='x')


def compute_mu_sigma_for_each_feature(X):
    means = np.mean(X, axis=0)
    # ddof=1 provides an unbiased estimator of the variance of a hypothetical infinite population.
    # ddof=0 provides a maximum likelihood estimate of the variance for normally distributed variables.
    variances = np.var(X, axis=0, ddof=1)
    return means, variances


def compute_mu_sigma_metrics_multivariate_gaussian(X):
    m, n = X.shape
    mu = np.mean(X, axis=0)
    sigma_metrics = 1 / m * (X - mu).T.dot(X - mu)
    return mu, sigma_metrics


def multivariate_gaussian(X, mu, sigma_metrics):
    n = len(mu)
    # X(m*n) sigma_metrics(n*n)
    X = X - mu
    # np.linalg.det
    p = np.power(2 * math.pi, -0.5 * n) * np.power(det(sigma_metrics), -0.5) * np.exp(
        -1 / 2 * np.sum(X.dot(pinv(sigma_metrics)) * X, axis=1))
    return p


def select_thread_hold(yval, pval):
    epsilons = np.linspace(min(pval), max(pval), 1000)
    best_epsilon = 0
    best_F1 = 0

    for epsilon in epsilons:
        predicts = (pval < epsilon).astype(int)
        true_positive = sum(((predicts == 1) & (yval == 1)))
        false_positive = sum(((predicts == 0) & (yval == 1)))
        false_negative = sum(((predicts == 1) & (yval == 0)))
        precise = true_positive / (true_positive + false_positive)
        recall = true_positive / (true_positive + false_negative)
        F1 = 2 * precise * recall / (precise + recall)
        if F1 > best_F1:
            best_F1 = F1
            best_epsilon = epsilon

    return best_epsilon, best_F1


mu, sigma = compute_mu_sigma_metrics_multivariate_gaussian(X)
pval = multivariate_gaussian(X_val, mu, sigma)
epsilon, f1 = select_thread_hold(y_val, pval)
print(epsilon)
print(f1)
