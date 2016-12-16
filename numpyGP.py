# Reference: https://docs.scipy.org/doc/numpy/reference/generated/numpy.random.normal.html


import numpy as np
import logging
import matplotlib.pyplot as plt

FORMAT = "%(message)s"
logging.basicConfig(format=FORMAT, level=logging.DEBUG)
logger = logging.getLogger()

@profile
def normal_testcase():
    mu, sigma = 0, 0.1
    s = np.random.normal(mu, sigma, 1000)
    verify_mean = abs(mu - np.mean(s)) < 0.01
    verify_var = abs(sigma - np.std(s, ddof=1)) < 0.01
    logger.debug("mean: %f" % abs(mu - np.mean(s)))
    logger.debug("variance of data: %f" % abs(sigma - np.std(s, ddof=1)))

def plot_gp():
    mu, sigma = 0, 0.1
    s = np.random.normal(mu, sigma, 10000)
    count, bins, ignored = plt.hist(s, 400, normed=True)
    plt.plot(bins, 1 / (sigma * np.sqrt(2 * np.pi)) *
             np.exp( - (bins - mu)**2 / (2 * sigma**2)),
             linewidth=2, color='r')
    plt.show()

if __name__ == "__main__":
    #normal_testcase()
    plot_gp()
