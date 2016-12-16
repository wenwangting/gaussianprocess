# Reference:https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.multivariate_normal.html#scipy.stats.multivariate_normal

import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
import numpy as np

import logging
FORMAT = "%(message)s"
logging.basicConfig(format=FORMAT, level=logging.DEBUG)
logger = logging.getLogger()

def multi_variate():
    x = np.linspace(0, 5, 10, endpoint=False)
    y = multivariate_normal.pdf(x, mean=2.5, cov=0.5)
    logging.debug(x)
    logging.debug(y)
    print (help(multivariate_normal.pdf))
    fig1 = plt.figure()
    ax = fig1.add_subplot(111)
    ax.plot(x, y)

    x, y = np.mgrid[-3:3:.01, -3:3:.01]
    pos = np.dstack((x, y))
    rv = multivariate_normal([0.5, -0.2], [[2.0, 0.3], [0.3, 0.5]])
    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111)
    ax2.contourf(x, y, rv.pdf(pos))
    plt.show()

def matrix_normal_testcase():
    from scipy.stats import matrix_normal
    M = np.arange(6).reshape(3, 2)
    U = np.diag([1, 2, 3])
    V = 0.3 * np.identity(2)
    X = M + 0.1
    logging.debug(matrix_normal.pdf(X, mean=M, rowcov=U, colcov=V))
    vectorised_X = X.T.flatten()
    equiv_mean = M.T.flatten()
    equiv_cov = np.kron(V,U)
    mn_pdf = multivariate_normal.pdf(vectorised_X, mean=equiv_mean, cov=equiv_cov)
    logging.debug(mn_pdf)

def multi_norm():
    x, y = np.mgrid[-10:20:.1, -14:10:.1]
    pos = np.dstack((x, y))
    rv = multivariate_normal([5, -2], [[1.0, 0.5], [0.5, 1]])
    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111)
    ax2.contourf(x, y, rv.pdf(pos))

    y1 = np.linspace(-14, 10, 2400)
    x1 = 1
    prob_y1_x1 = []
    for i in y1:
        prob_y1_x1.append(rv.pdf([x1, i]))
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111)
    ax1.plot(y1, prob_y1_x1)
    plt.show()

if __name__ == "__main__":
    multi_norm()
