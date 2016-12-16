from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
import numpy as np
import random
import math
import optunity
import matplotlib.ticker as ticker

def create_objective_function():
    xoff = random.random()
    yoff = random.random()
    def f(x, y):
        return (x - xoff)**2 + (y - yoff)**2
    return f

@profile
def optimize_objective(f):
    logs = {}
    solvers = optunity.available_solvers()
    for solver in solvers:
        pars, details, _ = optunity.minimize(f, num_evals=100, x=[-5, 5], y=[-5,5],
                                             solver_name=solver)
        logs[solver] = np.array([details.call_log['args']['x'],
                                 details.call_log['args']['y']])
    colors =  ['r', 'g', 'b', 'y', 'k', 'y', 'r', 'g']
    markers = ['x', '+', 'o', 's', 'p', 'x', '+', 'o']

    # compute contours of the objective function
    delta = 0.025
    x = np.arange(-5.0, 5.0, delta)
    y = np.arange(-5.0, 5.0, delta)
    X, Y = np.meshgrid(x, y)
    Z = f(X, Y)
    CS = plt.contour(X, Y, Z)
    plt.clabel(CS, inline=1, fontsize=8, alpha=0.5)
    for i, solver in enumerate(solvers):
        plt.scatter(logs[solver][0, :], logs[solver][1, :], c=colors[i],
                    marker=markers[i], alpha=0.80)
    plt.xlim([-5, 5])
    plt.ylim([-5, 5])
    plt.axis('equal')
    plt.legend(solvers)
    plt.show()

def plot_randomsearch():
    solver = optunity.available_solvers()[3]
    pars, details, _ = optunity.minimize(f, num_evals=100, x=[-5, 5], y=[-5,5],
                                         solver_name=solver)

if __name__ == "__main__":
    objective = create_objective_function()
    optimize_objective(objective)
