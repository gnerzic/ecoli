# Declaration: I have read and understood the sections of plagiarism in the
# College Policy on assessment offences and confirm that the work is my own,
# with the work of others clearly acknowledged.I give my permission to submit my
# report to the plagiarism testing database that the College is using and test
# it using plagiarism detection software, search engines or meta-searching
# software.

import unicodecsv as csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.model_selection as selec
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_score, \
    validation_curve, GridSearchCV, LeaveOneOut
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
import warnings

# warnings are reported by stratification algorithms because some classes
# occur too seldom in the dataset. To simplify the output of the programs,
# warnings are ignored.
warnings.filterwarnings("ignore")


# function to output Table 3
def output_table3(features):
    # TABLE 3:
    # Table 3 in the assignment report displayed the mean and std deviation
    # of each feature. The table_3 data frame below is used to population this
    # table
    table_3 = pd.DataFrame([features.mean(axis=0), features.std(axis=0)],
                           index=["mean", "std"]).T
    print("TABLE3:")
    print(table_3)


# function to output Table 5
def output_table5(feaures, labels):
    # TABLE 5:
    # Table 5 in assignment shows the result of a 4-fold cross validation of a
    # KNN Classifier where k=7. The code below produces this table.
    # instantiate a KNN Classifier with k = 7
    clf_table_5 = KNeighborsClassifier(n_neighbors=7)
    # create StratifiedKFold to cut randomly while maintain the label
    # distribution parameters of this call:
    # - n_splits=4 : number of cross validation split
    # - shuffle=True: to shuffle the source data, important is source is sorted
    skf = StratifiedKFold(n_splits=4, shuffle=True)
    # Perform cross validation on the model
    table_5 = cross_val_score(clf_table_5, features, labels, cv=skf)
    table_5 = np.append(table_5, table_5.mean())
    table_5 = np.append(table_5, table_5.std())
    print("TABLE5:")
    print(table_5)


# function to output Graph 1
def show_graph1(features, labels):
    # GRAPH 1:
    # Graph on shows the accuracy a KNN model as K increases (for both training
    # and validation set. The code below produces this graph.
    # k hold the increasing k values that will be cross validated
    k = np.arange(1, 16)

    # SciKit-Learn validation_curve function is used to obtain acuracy curves
    # on both the training and validation scores.
    # Parameters are
    # - Estimator=estimator=KNeighborsClassifier(): the prediction class
    # - param_name='n_neighbors': the parameter of the estimator to be varied to
    #                             produce the learning curve output
    # - param_name=k: set of value use to vary the parameter
    # - cv=skf: use same cross validation as for table_5
    skf = StratifiedKFold(n_splits=4, shuffle=True)
    train_score, val_score = validation_curve(estimator=KNeighborsClassifier(),
                                              X=features, y=labels,
                                              param_name='n_neighbors',
                                              param_range=k,
                                              cv=skf)
    # use matplotlib to graphically show the results.
    plt.plot(k, np.median(train_score, 1), color='blue',
             label='training score')
    plt.plot(k, np.median(val_score, 1), color='red',
             label='validation score')
    plt.legend(loc='best')
    plt.xlabel('neighbors')
    plt.ylabel('score')
    plt.show()


# function to output table 6
def output_table_6(features, labels):
    # TABLE 6
    # Table 6 is a confusion matrix from a KNN classifier wih k=7. First the
    # dataset will be split into training and validation data do that a
    # confusion matrix can be produced the train_test_split is a SciKit-Learn
    # function that splits the data into a training set (features and labels)
    # and a validation set (features and labels) parameters of this function
    #  call:
    # - feature and labels of the total data set to be split.
    # - training_size=.8: size in percentage of the training set to be extracted
    #   from the data. In this case 80%
    # - test_size=.2: size in percentage of the test set to be extracted from
    #   the data. In this case 20%
    # - stratify=labels: the data isn't well distributed over classes, we would
    #   like the distribution to be the same in the training and validation sets
    # - random_state=42: specify the seed of the data randomiser. The data in
    #  the input is sorted by class, and we would like the data to be randomised
    Xtrn, Xtst, ytrn, ytst = selec.train_test_split(features, labels,
                                                    train_size=.8,
                                                    test_size=.2, shuffle=True,
                                                    stratify=labels,
                                                    random_state=4)
    # instantiate a KNN classifier with k = 7
    clf = KNeighborsClassifier(n_neighbors=7)
    # train the classifier by inputting the training data
    clf.fit(Xtrn, ytrn)
    # the predict function will output the model's prediction of the validation
    # data
    ymodel = clf.predict(Xtst)
    print(accuracy_score(ytst, ymodel))
    # SciKit-Learn includes a confusion matrix function
    matrix = confusion_matrix(ytst, ymodel, labels=class_name)
    table_6 = pd.DataFrame(matrix, columns=class_name, index=class_name)
    print("TABLE6:")
    print(table_6)


# function to output table 7
def output_table_7(features, labels):
    # TABLE 7
    # same as table 5 but with im and imU have been collapsed.
    clf_table_7 = KNeighborsClassifier(n_neighbors=7)
    # create StratifiedKFold to cut randomly while maintain the label
    # distribution parameters of this call:
    # - n_splits=4 : number of cross validation split
    # - shuffle=True: to shuffle the source data, important is source is sorted
    skf = StratifiedKFold(n_splits=4, shuffle=True)
    labels_collapsed = labels.copy(deep=True)
    labels_collapsed[labels_collapsed == 'imU'] = 'im'
    # Perform cross validation on the model
    table_7 = cross_val_score(clf_table_7, features, labels_collapsed, cv=skf)
    table_7 = np.append(table_7, table_7.mean())
    table_7 = np.append(table_7, table_7.std())
    print("TABLE7:")
    print(table_7)


# function to output table 8
def output_table_8(features, labels):
    # TABLE 8:
    # Table 8 in the assignment shows the result of a brute force validation of
    # a KNN model, where all the hyper paramters are tested.
    # parameter space to perform the search over
    param_grid = {'n_neighbors': np.arange(1, 11),
                  'weights': ['uniform', 'distance'],
                  'algorithm': ['brute'],
                  'p': np.arange(1, 11),
                  }
    skf = StratifiedKFold(n_splits=4, shuffle=True)
    # collapse the classes
    labels_collapsed = labels.copy(deep=True)
    labels_collapsed[labels_collapsed == 'imU'] = 'im'
    best_params_list = []
    cv_results_list = []
    for x in range(100):
        grid = GridSearchCV(KNeighborsClassifier(), param_grid=param_grid,
                            cv=skf)

        grid.fit(features, labels_collapsed)
        best_params_list.append(grid.best_params_)
        table_8 = cross_val_score(grid.best_estimator_, features,
                                  labels_collapsed,
                                  cv=skf)
        table_8 = np.append(table_8, table_8.mean())
        table_8 = np.append(table_8, table_8.std())
        cv_results_list.append(table_8)

    with open('best_param_list.cvs', 'wb') as paramcsv:
        writer = csv.writer(paramcsv)
        writer.writerow(best_params_list)

    print("TABLE8:")
    print(np.mean(cv_results_list, axis=0))

# function to output table 8_1
def output_table_8_1(features, labels):
    # TABLE 8_1
    # Cross leave one out validation for k = 5 and k = 7 im and imU have been collapsed.
    clf_table_8_1 = KNeighborsClassifier(algorithm='brute',
                                         n_neighbors=5,
                                         p=2,
                                         weights='uniform')

    labels_collapsed = labels.copy(deep=True)
    labels_collapsed[labels_collapsed == 'imU'] = 'im'
    # TODO: add comment here
    table_8_1 = cross_val_score(clf_table_8_1, features, labels_collapsed, cv=LeaveOneOut())
    table_8_1 = np.append(table_8_1, table_8_1.mean())
    table_8_1 = np.append(table_8_1, table_8_1.std())
    print("TABLE8_1: K = 5")
    print(table_8_1)

    clf_table_8_2 = KNeighborsClassifier(algorithm='brute',
                                         n_neighbors=7,
                                         p=2,
                                         weights='uniform')


    table_8_2 = cross_val_score(clf_table_8_2, features, labels_collapsed, cv=LeaveOneOut())
    table_8_2 = np.append(table_8_2, table_8_2.mean())
    table_8_2 = np.append(table_8_2, table_8_2.std())
    print("TABLE8_1: K = 7")
    print(table_8_2)

# output table 10
def output_table_10(features, labels):
    # TABLE 10:
    # Table 10 in assignment shows the result of a 4-fold cross validation of a
    # multi-layered perceptron classifier with 5 hidden node.
    # The code below produces this table.
    # TODO: add comments
    clf_table_10 = MLPClassifier(solver='lbfgs',
                                 hidden_layer_sizes=(5,),
                                 random_state=1)
    # create StratifiedKFold to cut randomly while maintain the label distribution
    # parameters of this call:
    # - n_splits=4 : number of cross validation split
    # - shuffle=True: to shuffle the source data, important is source is sorted
    skf = StratifiedKFold(n_splits=4, shuffle=True)
    # cross validate the model
    table_10 = cross_val_score(clf_table_10, features, labels, cv=skf)
    table_10 = np.append(table_10, table_10.mean())
    table_10 = np.append(table_10, table_10.std())
    print("TABLE10:")
    print(table_10)

    clf_table_10_1 = MLPClassifier(solver='lbfgs',
                                   hidden_layer_sizes=(5,),
                                   random_state=1)
    labels_collapsed = labels.copy(deep=True)
    labels_collapsed[labels_collapsed == 'imU'] = 'im'
    table_10_1 = cross_val_score(clf_table_10_1, features, labels_collapsed, cv=skf)

    table_10_1 = np.append(table_10_1, table_10_1.mean())
    table_10_1 = np.append(table_10_1, table_10_1.std())
    print(table_10_1)


# function to output Graph 2
def show_graph2(features, labels):
    # GRAPH 2:
    # show learning curve for a neural network with structure (x,) where
    # x is in range 1 to 50
    hidden_layers = []
    for x in range(1, 50):
        hidden_layers.append((x,))


    skf = StratifiedKFold(n_splits=4, shuffle=True)
    labels_collapsed = labels.copy(deep=True)
    labels_collapsed[labels_collapsed == 'imU'] = 'im'
    train_sc, val_sc = validation_curve(estimator=MLPClassifier(solver='lbfgs',
                                                                random_state=1),
                                              X=features, y=labels_collapsed,
                                              param_name='hidden_layer_sizes',
                                              param_range=hidden_layers,
                                              cv=skf)
    plt.plot(range(1, 50), np.median(train_sc, 1), color='blue',
             label='training score')
    plt.plot(range(1, 50), np.median(val_sc, 1), color='red',
             label='validation score')
    plt.legend(loc='best')
    plt.xlabel('nodes')
    plt.ylabel('score')
    plt.show()


# function to output Graph 3
def show_graph3(features, labels):
    #  GRAPH 3:
    # show learning curve for a neural network with structure (10,x) where
    # x is in range 1 to 50
    # TODO comment
    hidden_layers = []
    for x in range(1, 50):
        hidden_layers.append((10, x))

    skf = StratifiedKFold(n_splits=4, shuffle=True)
    labels_collapsed = labels.copy(deep=True)
    labels_collapsed[labels_collapsed == 'imU'] = 'im'
    train_sc, val_sc = validation_curve(estimator=MLPClassifier(solver='lbfgs',
                                                                random_state=1),
                                              X=features, y=labels_collapsed,
                                              param_name='hidden_layer_sizes',
                                              param_range=hidden_layers,
                                              cv=skf)
    plt.plot(range(1, 50), np.median(train_sc, 1), color='blue',
             label='training score')
    plt.plot(range(1, 50), np.median(val_sc, 1), color='red',
             label='validation score')
    plt.legend(loc='best')
    plt.xlabel('nodes')
    plt.ylabel('score')
    plt.show()


# function to output table 11
def output_table_11(features, labels):
    labels_collapsed = labels.copy(deep=True)
    labels_collapsed[labels_collapsed == 'imU'] = 'im'
    class_name_collapsed = ["cp", "im", "pp", "om", "omL", "imL", "imS"]
    Xtrn, Xtst, ytrn, ytst = selec.train_test_split(features, labels_collapsed,
                                                    train_size=.8,
                                                    test_size=.2, shuffle=True,
                                                    stratify=labels,
                                                    random_state=4)
    # instantiate a KNN classifier with k = 7
    clf1 = KNeighborsClassifier(n_neighbors=7)
    # train the classifier by inputting the training data
    clf1.fit(Xtrn, ytrn)
    # the predict function will output the model's prediction of the validation
    # data
    ymodel1 = clf1.predict(Xtst)
    print(accuracy_score(ytst, ymodel1))
    # SciKit-Learn includes a confusion matrix function
    matrix = confusion_matrix(ytst, ymodel1, labels=class_name_collapsed)
    table_11 = pd.DataFrame(matrix, columns=class_name_collapsed,
                           index=class_name_collapsed)
    print("TABLE11(KNN):")
    print(table_11)

    clf2 = MLPClassifier(solver='lbfgs', hidden_layer_sizes=(10, 10),
                         random_state=1, shuffle=True, max_iter = 10000,
                         activation='logistic', alpha=1e-5)
    clf2.fit(Xtrn, ytrn)
    # the predict function will output the model's prediction of the validation
    # data
    ymodel2 = clf2.predict(Xtst)
    print(accuracy_score(ytst, ymodel2))
    # SciKit-Learn includes a confusion matrix function
    matrix2 = confusion_matrix(ytst, ymodel2, labels=class_name_collapsed)
    table_11_1 = pd.DataFrame(matrix2, columns=class_name_collapsed,
                           index=class_name_collapsed)
    print("TABLE11(MLP):")
    print(table_11_1)

# function to output table 12
def output_table_12(features, labels):
    clf_table_12_1 = MLPClassifier(solver='lbfgs',
                                   hidden_layer_sizes=(10,),
                                   random_state=1)
    labels_collapsed = labels.copy(deep=True)
    labels_collapsed[labels_collapsed == 'imU'] = 'im'
    table_12_1 = cross_val_score(clf_table_12_1, features, labels_collapsed,
                                 cv=LeaveOneOut())

    table_12_1 = np.append(table_12_1, table_12_1.mean())
    table_12_1 = np.append(table_12_1, table_12_1.std())
    print("TABLE12_1: (10,)")
    print(table_12_1)

    clf_table_12_2 = MLPClassifier(solver='lbfgs',
                                   hidden_layer_sizes=(10, 10),
                                   random_state=1)

    table_12_2 = cross_val_score(clf_table_12_2, features, labels_collapsed,
                                 cv=LeaveOneOut())

    table_12_2 = np.append(table_12_2, table_12_2.mean())
    table_12_2 = np.append(table_12_2, table_12_2.std())
    print("TABLE12_2: (10,)")
    print(table_12_2)


# col_names is the name of the features of each observations. These columns are
# not in the source data, hence they are inserted in the code.
col_names = [None, "mcg", "gvh", "lip", "chg", "aac", "alm1", "alm2", "class"]

# list of label names (could have been extracted from the data but are also
# listed in the data descriptions. the list of label names is used for to output
# meaningful labels in the confusion matrix.
class_name = ["cp", "im", "pp", "imU", "om", "omL", "imL", "imS"]

# use Pandas's read_table function to import the data set.
# parameters of this function call:
# - filename: name of the file
# - index_col=0: all observations have a unique name in the first column,
#   it will be used a the Pandas index
# - sep='\s+': the features are space deliminited '\s+' is a regular expression
#   to denotes one of more spaces
# - header=None: states that the data set does not contain a header row and that
#   data start at row 0
# - name=col_name: the data doesn't have a header, the column names are
#   explicitly provided in the col_names array
data = pd.read_table("ecoli.data", index_col=0, sep='\s+', header=None,
                     names=col_names)

# the first columns are contains the features, this is extracted using a slice
# on the dataset
features = data.loc[:, :"alm2"]
# the last column (called 'class') contains the labels.
labels = data.loc[:, "class"]
# table output (comment as needed)
# WARNING: output_table_8 runs for a long time (20 mins + )
output_table3(features)
output_table5(features, labels)
show_graph1(features, labels)
output_table_6(features, labels)
output_table_7(features, labels)
#output_table_8 runs for a long time (20 mins + )
output_table_8(features, labels)
output_table_8_1(features, labels)
output_table_10(features, labels)
show_graph2(features, labels)
show_graph3(features, labels)
output_table_11(features, labels)
output_table_12(features, labels)

