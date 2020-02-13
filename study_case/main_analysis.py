"""
Candy case study

Python 3.6 

This script aims at finding out which characteristics of the candy are sentiment to customers
A recommendation of the new product is made based on combination of regression and classification analysis

After setting the environment from either environment.yml or requirements.txt, and making sure that
utilities.py is accessible, click run of main_analysis will generate visualisations in folder 'visualisation'
and print results in results.txt

Author: Rongrong Shen
"""

# import packages
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.offline as py
import utilities
import sys
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from utilities import plot_confusion_matrix, creat_matrix, feature_selection, classification
from sklearn.metrics import confusion_matrix

if __name__ == "__main__":
    ##################################################################################################
    #------------------------------------ Define variables ------------------------------------------#
    ##################################################################################################
    root_path = "/Users/shenrongrong/git_project/study_case/"
    data_path = root_path + "candy_data.csv"                                           # path where dataset is stored
    corr_path = root_path + "visualisation/correlation.png"                            # path where to store correlation png
    distribution_path = root_path + "visualisation/data_distribution.html"             # path where to store data distribution
    ground_truth_path = root_path + "visualisation/ground_truth_histogram.html"        # path where to store ground truth distribution html
    importance_path = root_path + "visualisation/characteristics_importance.html"      # path where to store importance of features
    result_path = root_path + "results.txt"                                            # path where the print out is stored
    sys.stdout = open(result_path, 'w')                                                # print all to txt

    data = pd.read_csv(data_path)                                                      # load dataset into dataframe
    data = data.drop(["competitorname"], axis=1)

    random_state = 1                                                                   # set random state seed
    label_threshold = [0.5]                                                            # used for classification task, the list contains thresholds between classes
    label_names = ["not competitive", "competitive"]                                   # literal meaning of classes
    n_splits = 10                                                                      # n splits of cross validation
    shuffle = True                                                                     # shuffle state

    selector = "rfe"                                                                   # rekursive feature elimination
    rfe_plot = False                                                                   # whether to plot rfe grid scores or not
    # classifier names for [support vector machine, random forest, logistic regression, dummy]
    # use several classifiers to examine feature importance, considering performance of classifier
    classifiers = ["dummy", "svm", "rf", "LogRe"]                                      # dummy classification as baseline model
    kernel = "linear"                                                                  # kernel used in svm
    C = 10                                                                             # C in svm
    n_trees = 20                                                                       # number of trees used in rf
    max_depth = 5                                                                      # maximal depth used in rf
    importance_traces = []                                                             # trace list for importance visualisation

    feature_names, feature_matrix, label, ground_truth = creat_matrix(data, label_threshold)
    skf = StratifiedKFold(n_splits=n_splits, random_state=random_state, shuffle=shuffle)         # creat spliter



    ###############################################################################################################
    #-------------------------------- Data overview & linear regression ------------------------------------------#
    ###############################################################################################################
    # Visualisation: distribution of features in html
    utilities.data_distribution(data, distribution_path)
    # Visualisation: correlation matrix in png
    utilities.correlation_matrix(data, corr_path)
    # Visualisation: ground truth histogram in png
    utilities.ground_truth_distribution(ground_truth, ground_truth_path)
    # Print linear regression parameters
    utilities.linear_regression_paramters(feature_matrix, ground_truth, feature_names)


    ################################################################################################################
    #-------------------------------- Classification and cross validation -----------------------------------------#
    ################################################################################################################
    for classifier in classifiers:
        high_rank_fre = np.zeros((len(feature_names),), dtype=float)                     # Selection frequency of each feature
        label_true = np.array([], dtype=int)                                             # Aggregation of true labels to test data
        label_predicted = np.array([], dtype=int)                                        # Aggregation of predicted labels to test data

        # Cross validation loop
        for train_index, test_index in skf.split(feature_matrix, label):
            # Train and test sets for feature matrix and label
            X_train, X_test = feature_matrix[train_index], feature_matrix[test_index]
            Y_train, Y_test = label[train_index], label[test_index]

            # Feature matrix after selection and features' scores(importance) during selection
            X_train_transformed, X_test_transformed, rank_fre = feature_selection(X_train, X_test, Y_train,
                                                                        selector=selector,
                                                                        classifier=classifier,
                                                                        random_state=random_state,
                                                                        shuffle=shuffle,
                                                                        kernel=kernel, C=C,
                                                                        n_trees=n_trees, max_depth=max_depth,
                                                                        rfe_plot=rfe_plot,
                                                                        )

            # Predicted labels for test set and feature importance found by classifier
            Y_predicted = classification(X_train_transformed, X_test_transformed, Y_train,
                                         classifier=classifier,
                                         random_state=random_state,
                                         shuffle=shuffle,
                                         kernel=kernel, C=C,
                                         n_trees=n_trees, max_depth=max_depth,
                                         )

            label_true = np.concatenate((label_true, Y_test))
            label_predicted = np.concatenate((label_predicted, Y_predicted))
            high_rank_fre += rank_fre

        # Relative importance of each feature within the whole feature set
        importance_ratio = high_rank_fre / sum(high_rank_fre)
        trace = go.Bar(x=feature_names, y=importance_ratio, name=classifier)
        importance_traces.append(trace)

        # print accuracy of each classifier
        print("Accuracy of", classifier, "classifier is:", "%.4f" % accuracy_score(label_true, label_predicted))
        # plot confusion matrix
        cm = confusion_matrix(label_true, label_predicted)
        print("Confusion matrix:")
        print(cm)
        print()
        plot_confusion_matrix(cm=cm, label_names=label_names,
                              title="Confusion matrix of " + classifier + " classifier",
                              normalize=True)

    layout = dict(title=dict(text='Relative importance of features by different classifier', x=0.5, ),
                  yaxis=dict(title="ratio"),
                  xaxis=dict(title='candy characteristic',),
                  )

    fig = dict(data=importance_traces, layout=layout)
    py.plot(fig, filename=importance_path, auto_open=False)                            # show importance of features

    ################################################################################################################
    # --------------------------------------Make recommendation----------------------------------------------------#
    ################################################################################################################
    print("Based on the correlogram and importance ratio, the recommendation would be:")
    print("cookie-based sweets with chocolate ingredient instead of gummies")
    sys.stdout.close()






