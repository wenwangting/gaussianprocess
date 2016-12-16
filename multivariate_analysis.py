# Reference: https://github.com/gatsoulis/
# a_little_book_of_python_for_multivariate_analysis/blob/master/a_little_book_of_python_for_multivariate_analysis.ipynb
from __future__ import print_function

from pydoc import help
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import scale
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from scipy import stats

from IPython.display import display

def read_data():
    data = pd.read_csv(
        "http://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data",
         header=None)
    data.columns = ["V" + str(i) for i in range(1, len(data.columns) + 1)]
    data.V1 = data.V1.astype(str)
    X = data.loc[:, "V2":]
    y = data.V1
    return data

# Plotting multivariate data
def matrix_scatterplot(data):
    """
    Plotting multivariate data is to make matrix scatterplot, showing each pair of
    variables plotted against each other. We can use the scatter_matrix() function
    from the pandas.tools.plotting package to do this.
    """
    pd.tools.plotting.scatter_matrix(data.loc[:, "V2":"V6"], diagonal="kde")
    plt.tight_layout()
    plt.show()

# A scatterplot with the Data Points Labelled by their Group
def scatterplot_label(data):
    """
    Plot a scatterplot of 2 variables, with labelled by their group.
    Use the lmplot function from the searbon,
    """
    sns.lmplot("V4", "V5", data, hue="V1", fit_reg=False);
    plt.show()

def profile_plot(data):
    """
    Another type of plot data that is useful is profile plot, which show the variation
    in each of the variables, by plotting the value of each of the variables for each
    of samples.
    """
    ax = data[["V2", "V3", "V3", "V4", "V5", "V6"]].plot()
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5));
    plt.show()

# Calculating summary statistics for multivariate data
def calculate_stat(data):
    X = data.loc[:, "V2":]
    mean_vec= X.apply(np.mean)
    max_vec = X.apply(np.max)
    min_vec = X.apply(np.min)
    for i in zip(max_vec, min_vec, mean_vec):
        print(i)

def means_vars_pergroup(data):
    class2_data = data[data["V1"] == "2"]
    class2_data_mean = class2_data.loc[:, "V2":].apply(np.mean)
    class2_data_var = class2_data.loc[:, "V2":].apply(np.std)
    print (class2_data_mean, class2_data_var)

def printMeanAndSdByGroup(variables, groupvariable):
    data_groupby = variables.groupby(groupvariable)
    from IPython.display import display, HTML
    #display(data_groupby.apply(np.mean))
    #display(data_groupby.apply(np.std))
    display(pd.DataFrame(data_groupby.apply(len)))

def calcWithinGroupsVariance(variable, groupVariable):
    levels = sorted(set(groupVariable))
    numLevels = len(levels)
    numTotal = 0
    denomTotal = 0
    for leveli in levels:
        leveli_data = variable[groupVariable == leveli]
        leveli_length = len(leveli_data)
        sdi = np.std(leveli_data)
        numi = (leveli_length)*sdi**2
        denomi = leveli_length
        numTotal = numTotal + numi
        denomTotal = denomTotal + denomi
    Vw = numTotal / (denomTotal - numLevels)
    return Vw

def correlations_multivariate_data(data):
    """
    p-value for the statistical test of whether the correlation coefficient is
    significantly different from 0 is 0.21, much greater than 0.05(which we use
    here as the cutoff for statistical significance). So there is very weak evidence
    that the correlation is non-zero.
    """
    corr = stats.pearsonr(data.loc[:, "V2"], data.loc[:, "V3"])
    print ("p-value:\t", corr[1])
    print ("cor:\t\t", corr[0])
    corrmat = data.loc[:, "V2":].corr()
    print (corrmat)
    # Heatmap representation of the correlation matrix
    sns.heatmap(corrmat, vmax=1., square=False).xaxis.tick_top()
    plt.show()

# adapted from http://matplotlib.org/examples/specialty_plots/hinton_demo.html
def hinton(matrix, max_weight=None, ax = None):
    """ Draw Hinton diagram for visualizing a weight matrix.
    """
    ax = ax if ax is not None else plt.gca()

    if not max_weight:
        max_weight = 2 ** np.ceil(np.log(np.abs(matrix).max())/np.log(2))

    ax.patch.set_facecolor('lightgray')
    ax.set_aspect('equal', 'box')
    ax.xaxis.set_major_locator(plt.NullLocator())
    ax.xaxis.set_major_locator(plt.NullLocator())

    for (x, y), w in np.ndenumerate(matrix):
        color = 'red' if w > 0 else 'blue'
        size = np.sqrt(np.abs(w))
        rect = plt.Rectangle([x - size / 2, y - size / 2], size, size,
                             facecolor=color, edgecolor=color)
        ax.add_patch(rect)
        nticks = matrix.shape[0]

    ax.xaxis.tick_top()
    ax.set_xticks(range(nticks))
    ax.set_xticklabels(list(matrix.columns), rotation=90)
    ax.set_yticks(range(nticks))
    ax.set_yticklabels(matrix.columns)
    ax.grid(False)

    ax.autoscale_view()
    ax.invert_yaxis()
    plt.show()

# Most the top N strongest correlations
def mostHighlyCorrelated(dataFrame, numToReport):
    # find the correlations
    cormatrix = dataFrame.corr()
    # set the correlations on the diagonal or lower triange to zero,
    # so they will not be reported as the highest ones:
    cormatrix *= np.tri(*cormatrix.values.shape, k=-1).T
    cormatrix *= np.tri(*cormatrix.values.shape, k=-1).T
    # find the top n correlations
    cormatrix = cormatrix.stack()
    cormatrix = cormatrix.reindex(cormatrix.abs().
        sort_values(ascending=False).index).reset_index()
    # assign human-friendly names
    cormatrix.columns = ["FirstVariables", "SecondVariable", "Correlation"]
    return cormatrix.head(numToReport)

def standardisingVariables(trainSet):
    """
    It's a good idea to firstly standardise the variables so that they all have
    variance 1 and mean 0, and to then carry out principal component analysis on
    standardised data. This would allow us to find the principal coponents that
    provide the best low-demensional representation of the variation in the original
    data, without being overly biased by those variables that show the most variance
    in the original data.
    """
    standardisedX = scale(trainSet)
    standardisedX = pd.DataFrame(standardisedX, index=trainSet.index, columns=trainSet.columns)
    print (standardisedX.apply(np.mean))
    print (standardisedX.apply(np.std))
    return standardisedX

def pca_summary(pca, standardised_data, out=True):
    names = ["PC"+str(i) for i in range(1, len(pca.explained_variance_ratio_)+1)]
    a = list(np.std(pca.transform(standardised_data), axis=0))
    b = list(pca.explained_variance_ratio_)
    c = [np.sum(pca.explained_variance_ratio_[:i]) for i in range(1, len(pca.explained_variance_ratio_)+1)]
    columns = pd.MultiIndex.from_tuples([("sdev", "Standard deviation"), ("varprop", "Proportion of Variance"), ("cumprop", "Cumulative Proportion")])
    summary = pd.DataFrame(zip(a, b, c), index=names, columns=columns)
    if out:
        print("Importance of components:")
        display(summary)
    return summary

def retainComponents(pca, standardised_data):
    """ Decide how many principal components should be retained, it's common to
    summarise the results of a principal components analysis by make a scree plot
    """
    y = np.std(pca.transform(standardised_data), axis=0)**2
    x = np.arange(len(y)) + 1
    plt.plot(x, y, "o-")
    plt.xticks(x, ["Comp."+str(i) for i in x], rotation=60)
    plt.ylabel("Variance")
    plt.show()

def usecase():
        data = read_data()
        X = data.loc[:, "V2":]
        y = data.loc[:, "V1"]
        #matrix_scatterplot(data)
        #scatterplot_label(data)
        #profile_plot(data)
        #calculate_stat(data)
        #means_vars_pergroup(data)
        #printMeanAndSdByGroup(data.loc[:, "V2":], data.loc[:,"V1"])
        #print calcWithinGroupsVariance(data.loc[:, 'V2'], data.loc[:, "V1"])
        #correlations_multivariate_data(data)
        #hinton(data.loc[:, "V2":].corr())
        #print (mostHighlyCorrelated(data.loc[:, "V2":], 10))
        standardisedX = standardisingVariables(X)
        pca = PCA().fit(standardisedX)
        # summary = pca_summary(pca, standardisedX)
        retainComponents(pca, standardisedX)

if __name__ == "__main__":
    usecase()
