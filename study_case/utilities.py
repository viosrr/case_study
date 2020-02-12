"""
Utilities for main analysis
"""

# import packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.offline as py
import itertools
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from sklearn.feature_selection import RFECV
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier

def plot_confusion_matrix(cm,
                          label_names,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True):
    """
    given a sklearn confusion matrix (cm), make a plot

    Parameters
    ---------
    cm :           confusion matrix from sklearn.metrics.confusion_matrix

    label_names :  given classification classes such as [0, 1]
                   the class names, for example: ['not competitive','competitive']

    title :        the text to display at the top of the matrix

    cmap :         the gradient of the values displayed from matplotlib.pyplot.cm
                   see http://matplotlib.org/examples/color/colormaps_reference.html
                   plt.get_cmap('jet') or plt.cm.Blues

    normalize:     If False, plot the raw numbers
                   If True, plot the proportions

    Citiation
    ---------
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

    """

    accuracy = np.trace(cm) / np.sum(cm).astype('float')
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if label_names is not None:
        tick_marks = np.arange(len(label_names))
        plt.xticks(tick_marks, label_names, rotation=45)
        plt.yticks(tick_marks, label_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")


    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.show()


def creat_matrix(dataframe, label_threshold):
    """Create feature matrix and label from data frame.

    Parameters
    ----------
    dataframe : dataframe, shape (n_samples, n_features + n_groundtruth)
        Dataframe containing features and ground truth

    label_threshold : list, shape (1, )
        Thresholds to turn continuierlich ground truth into discrete classes.

    Returns
    -------
    feature names : list, shape (1, )
        Names of features

    feature matrix : array_like, shape (n_samples, n_features)
        Features matrix

    label : array, shape (n_samples, )
        Label by classifying original ground truth

    ground_truth : array, shape (n_samples, )
        Original ground truth value
    """

    ground_truth = np.array(dataframe["winpercent"]/100)
    label = np.zeros((ground_truth.size,))
    dataframe = dataframe.drop(["winpercent"], axis=1)
    feature_names = list(dataframe.columns)

    feature_matrix = dataframe.to_numpy()

    for i, val in enumerate(label_threshold):
        if len(label_threshold) == 1:
            label[np.where(ground_truth < val)] = i
            label[np.where(ground_truth >= val)] = i + 1
        else:
            if i == 0:
                label[np.where(ground_truth<val)] = i
                label[np.where((ground_truth >= val) & (ground_truth < label_threshold[i + 1]))] = i + 1
            elif i == len(label_threshold) - 1:
                label[np.where(ground_truth >= val)] = i + 1
            else:
                label[np.where((ground_truth >= val) & (ground_truth < label_threshold[i+1]))] = i + 1

    return feature_names, feature_matrix, label, ground_truth


def data_distribution(data, distribution_path):
    """Visulaisation of features distribution, stored in .html file.

    Parameters
    ----------
    data : array_like, shape (n_samples, n_features)
       Feature matrix

    distribution_path : str
       Path where the visualisation results are stored
    """

    features = []
    data["winpercent"] = data["winpercent"]/100

    for column_name, column_data in data.iteritems():
        trace = go.Scatter(
            x=data.index,
            y=column_data,
            mode="markers",
            name=column_name
        )
        features.append(trace)

    layout = dict(title=dict(text='Candy information', x=0.5,),
                  yaxis=dict(title="value"),
                  xaxis=dict(
                      title='competitor id',
                      ticklen=5,
                      zeroline=False,
                      gridwidth=2,
                  ),
                  )

    fig = dict(data=features, layout=layout)
    py.plot(fig, filename=distribution_path, auto_open=False)

def ground_truth_distribution(ground_truth, ground_truth_path, bin_size=0.1):
    """Visualisation of original ground truth distribution in html file.

    Parameters
    ----------
    ground_truth : array, shape (n_samples, )
        Values of original ground truth

    ground_truth_path : str
       Path where the distribution visualisation is stored

    bin_size : size of bin
    """

    fig = go.Figure(data=[go.Histogram(x=ground_truth, xbins=dict(size=bin_size))],
                    layout=dict(title=dict(text='Histogram of winpercent', x=0.5,),
                    xaxis=dict(title="winpercent")))
    py.plot(fig, filename=ground_truth_path, auto_open=False)

def correlation_matrix(dataframe, corr_path):
    """Heatmap of pairweis correlation coefficients between feature-feature and feature-ground_truth in png file.

   Parameters
   ----------
   dataframe : dataframe, shape (n_samples, n_features + n_ground_truth)
        Dataframe containing features and ground truth

   corr_path : str
       Path where the correlation heatmap is stored
   """

    plt.figure(figsize=(13, 13), dpi=90)
    sns.heatmap(dataframe.corr(), xticklabels=dataframe.corr().columns, yticklabels=dataframe.corr().columns,
                cmap='RdYlGn', center=0,
                annot=True)
    # Decorations
    plt.title('Correlogram of features', fontsize=22)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.savefig(corr_path)

def linear_regression_paramters(feature_matrix, ground_truth, feature_names):
    """Print parameters of linear regression.

    Parameters
    ----------
    feature_matrix : array_like, shape (n_samples, n_features)
        Feature matrix

    ground_truth : array, shape (n_samples, )
       Original ground truth

    feature_names : list, shape (n_features, )
       Feature names
    """
    linreg = LinearRegression()
    linreg.fit(feature_matrix, ground_truth)
    score = linreg.score(feature_matrix, ground_truth)

    print("The intercept of linear regression is:", "%.4f" % linreg.intercept_)
    print()
    print("The coefficients of each characteristic are:")
    for id, item in enumerate(linreg.coef_):
        print(feature_names[id], ":", "%.4f" % item)
    print()
    print("The R^2 coefficient of determination is:", "%.4f" % score)
    print()

def feature_selection(X_train, X_test, Y_train, **kwargs):
    """Feature selection.

    Parameters
    ----------
    X_train : array_like, shape (n_train_samples, n_features)
        Training data

    X_test : array_like, shape (n_test_samples, n_features)
        test data

    Y_train : array, shape (n_train_samples, )
        label of training data

    Returns
    -------
    X_train_transformed : array_like, shape (n_train_samples, n_selected_features)
        Training data after feature selection

    X_test_transformed : array_like, shape (n_test_samples, n_selected_features)
        Test data after the same feature selection method

    rank_fre : array, shape(n_features, )
        Whether each feature is selected into final feature set, 1 for yes, 0 for no
    """
    dic = {"selector": "rfe", "classifier": "svm", "n_splits": 10, "random_state": 1, "shuffle": True,
           "kernel": "linear", "C": 1,
           "n_trees": 10, "max_depth": 3,
           "rfe_plot": False,
           "rank_fre": np.ones((X_train.shape[1], ), dtype=int)
           }

    for key in kwargs:
        dic[key] = kwargs[key]

    if dic["classifier"] == "svm":
        clf = SVC(kernel=dic["kernel"], C=dic["C"], random_state=dic["random_state"])
    elif dic["classifier"] == "rf":
        clf = RandomForestClassifier(n_estimators=dic["n_trees"], max_depth=dic["max_depth"], random_state=dic["random_state"])
    elif dic["classifier"] == "LogRe":
        clf = LogisticRegression(random_state=dic["random_state"])
    else:
        return X_train, X_test, dic["rank_fre"]

    if dic["selector"] == "rfe":
        rfecv = RFECV(estimator=clf, step=1,
                      cv=StratifiedKFold(n_splits=dic["n_splits"], random_state=dic["random_state"], shuffle=dic["shuffle"]),
                      scoring='accuracy')
        rfecv.fit(X_train, Y_train)
        X_train_transformed, X_test_transformed = rfecv.transform(X_train), rfecv.transform(X_test)
        dic["rank_fre"][rfecv.ranking_ > 1] = 0

        if dic["rfe_plot"]:
            print("Optimal number of features : %d" % rfecv.n_features_)
            # Plot number of features VS. cross-validation scores
            plt.figure()
            plt.xlabel("Number of features selected")
            plt.ylabel("Cross validation score (nb of correct classifications)")
            plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
            plt.show()

        return X_train_transformed, X_test_transformed, dic["rank_fre"]
    else:
        return X_train, X_test, dic["rank_fre"]

def classification(X_train_transformed, X_test_transformed, Y_train, **kwargs):
    """Classification.

    Parameters
    ----------
    X_train_transformed : array_like, shape (n_train_samples, n_selected_features)
        Training data after feature selection

    X_test_transformed : array_like, shape (n_test_samples, n_selected_features)
        Test data after feature selection

    Y_train : array, shape (n_train_samples, )
        Label of training data

    Returns
    -------
    Y_predicted : array, shape (n_test_samples, )
        Predicted label of test data
    """

    dic = {"classifier": "svm", "n_splits": 10, "random_state": 1, "shuffle": True,
           "kernel": "linear", "C": 1,
           "n_trees": 10, "max_depth": 3,
           "strategy": "most_frequent",
            }

    for key in kwargs:
        dic[key] = kwargs[key]

    if dic["classifier"] == "svm":
        clf = SVC(kernel=dic["kernel"], C=dic["C"])
    elif dic["classifier"] == "rf":
        clf = RandomForestClassifier(n_estimators=dic["n_trees"], max_depth=dic["max_depth"], random_state=dic["random_state"])
    elif dic["classifier"] == "LogRe":
        clf = LogisticRegression(random_state=dic["random_state"])
    elif dic["classifier"] == "dummy":
        clf = DummyClassifier(strategy=dic["strategy"], random_state=dic["random_state"])

    clf.fit(X_train_transformed, Y_train)
    Y_predicted = clf.predict(X_test_transformed)

    return Y_predicted