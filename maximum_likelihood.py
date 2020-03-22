import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def sample_gaussian(mean, cov, n, noise=1e-4):
    """
    Sample from a given multivariate Gaussian distribution
    :param mean: (D, ) or (D, 1)
    :param cov: (D, D) positive semi-definite
    :param n: number of samples to generate
    :param noise: to be applied in Cholesky decomposition
    :return: samples (n, D)
    """
    if mean.shape[0] != cov.shape[0]:
        raise Exception("Dimension not matched")
    if np.any(np.linalg.eigvals(cov) < 0):
        raise Exception("Covariance matrix must be positive semi-definite")
    if noise is None:
        L = np.linalg.cholesky(cov)
    else:
        L = np.linalg.cholesky(cov + noise * np.eye(cov.shape[0]))
    standard_sample = np.random.randn(mean.shape[0], n)
    sample = mean.reshape(-1, 1) + np.matmul(L, standard_sample)
    return sample.transpose()


def ml_gaussian(x):
    """
    Maximum likelihood of multivariate gaussian
    :param x: (N, D)
    :return: ML estimates of mean and covariance
    """
    if 0 in x.shape:
        raise Exception("incorrect input")
    mu = np.average(x, 0).reshape(1, -1)
    sigma = np.matmul((x - mu).transpose(), (x - mu)) / x.shape[0]
    sigma = (sigma + sigma.transpose()) / 2  # numerical issues
    return mu.reshape(-1, 1), sigma


mean_orig = np.array([0, 0])
cov_orig = np.eye(2)
n_sample = 10000
orig_sample = sample_gaussian(mean_orig, cov_orig, n_sample)
mean_ml, cov_ml = ml_gaussian(orig_sample)
ml_sample = sample_gaussian(mean_ml, cov_ml, n_sample)

print("Plotting...")
fig, ax = plt.subplots(1, 2, figsize=(20, 10))
ax[0].set_title("Original Samples", fontsize=20)
ax[1].set_title("ML estimate samples", fontsize=20)
for a in ax:
    a.xaxis.set_tick_params(labelsize=20)
    a.yaxis.set_tick_params(labelsize=20)
sns.jointplot(x=orig_sample[:, 0], y=orig_sample[:, 1], kind="kde", space=0, ax=ax[0])
sns.jointplot(x=ml_sample[:, 0], y=ml_sample[:, 1], kind="kde", space=0, ax=ax[1])
fig.show()
