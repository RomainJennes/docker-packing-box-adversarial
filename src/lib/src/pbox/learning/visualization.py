# -*- coding: UTF-8 -*-
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from contextlib import suppress
from functools import wraps
from math import ceil
from matplotlib.colors import ListedColormap
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.manifold import TSNE
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import export_text, plot_tree
from .algorithm import Algorithm 

__all__ = ["VISUALIZATIONS"]


cmap_light = ListedColormap(["orange", "cyan", "cornflowerblue"])


def _preprocess(f):
    """ This decorator preprocesses the input data and updates the keyword-arguments with this fitted data. """
    @wraps(f)
    def _wrapper(*args, **kwargs):
        X = kwargs['data']
        n, p = kwargs.get('pca_components', 20), kwargs.get('perplexity', 30)
        X = SimpleImputer(missing_values=np.nan, strategy=kwargs.get('imputer_strategy', "mean")).fit_transform(X)
        # preprocess data with a PCA with n components to reduce the high dimensionality (better performance)
        pca = PCA(n, random_state=42)
        if 'target' in kwargs:
            pca.fit(X, kwargs['target'])
            X = pca.transform(X)
        else:
            X = pca.fit_transform(X)
        # now reduce the n components to 2 dimensions with t-SNE (better results but less performance) if relevant
        if n > 2:
            X = TSNE(2, random_state=42, perplexity=p).fit_transform(X)
        kwargs['data'] = X
        return f(*args, **kwargs)
    return _wrapper


def image_dt(classifier, width=5, fontsize=10, **params):
    params['filled'] = True
    fig = plt.figure()
    plot_tree(classifier, **params)
    return fig


@_preprocess
def image_knn(classifier, **params):
    X, y = params['data'], params['target']
    # retrain kNN with the preprocessed data (with dimensionality reduced to N=2, hence not using 'classifier')
    knn = KNeighborsClassifier(**params['algo_params'])
    knn.fit(X, y)
    # now set color map then plot
    labels = list(y.label.unique())
    colors = mpl.cm.get_cmap("jet", len(labels))
    fig, axes = plt.subplots()
    DecisionBoundaryDisplay.from_estimator(knn, X, cmap=colors, ax=axes, alpha=.3,
                                           response_method="predict", plot_method="pcolormesh", shading="auto")
    plt.scatter(X[:, 0], X[:, 1], c=[labels.index(v) for v in y.label.ravel()][::-1], cmap=colors, alpha=1.0)
    return fig


def image_rf(classifier, width=5, fontsize=10, **params):
    n = len(classifier.estimators_)
    rows = ceil(n / width)
    cols = width if rows > 1 else n
    fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=tuple(map(lambda x: 2*x, (cols, rows))), dpi=900)
    # flatten axes, otherwise it is a matrix of all subplots
    with suppress(TypeError):
        axes = [ax for lst in axes for ax in lst]
    params['filled'] = True
    for i in range(n):
        plot_tree(classifier.estimators_[i], ax=axes[i], **params)
        axes[i].set_title("Estimator: %d" % i, fontsize=fontsize)
    return fig


@_preprocess
def image_clustering(classifier, **params):
    X = params['data']
    # retrain with the preprocessed data (with dimensionality reduced to N=2, hence not using 'classifier')
    cls = Algorithm.get(params['algo_name']).base(**params['algo_params'])
    label = cls.fit_predict(X)
    # now set color map then plot
    labels = np.unique(label)
    colors = mpl.cm.get_cmap("jet", len(labels))
    fig, axes = plt.subplots()
    for i in labels:
        plt.scatter(X[label == i, 0], X[label == i, 1] , label=i, cmap=colors)
    return fig


def text_dt(classifier, **params):
    return export_text(classifier, **params)


def text_rf(classifier, **params):
    s = ""
    for i in range(len(classifier.estimators_)):
        s += "\nEstimator: %d\n" % i
        s += export_text(classifier.estimators_[i], **params)
    return s


VISUALIZATIONS = {
    'DT':  {'image': image_dt, 'text': text_dt},
    'kNN': {'image': image_knn, 'data': True},
    'RF':  {'image': image_rf, 'text': text_rf},
}
for a in ['AC', 'AP', 'Birch', 'DBSCAN', 'KMeans', 'MBKMeans', 'MS', 'OPTICS', 'SC']:
    VISUALIZATIONS[a] = {'image': image_clustering, 'data': True, 'target': False}

