#!/usr/bin/env python
import numpy as np
from sklearn import gaussian_process
import logging

FORMAT = "%(message)s"
logging.basicConfig(format=FORMAT, level=logging.DEBUG)
logger = logging.getLogger()

def f(x):
    return x * np.sin(x)

def gpTestcase():
    X = np.atleast_2d([1., 3., 5., 6., 7., 8.]).T
    y = f(X).ravel()
    x = np.atleast_2d(np.linspace(0, 10, 1000)).T
    gp = gaussian_process.GaussianProcess(theta0=1e-2, thetaL=1e-4, thetaU=1e-1)
    gp.fit(X, y)
    logging.debug(str(gp))

gpTestcase()
