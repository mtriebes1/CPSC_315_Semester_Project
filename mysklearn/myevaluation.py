import numpy as np
import math
import random

import mysklearn.myutils as myutils

def train_test_split(X, y, test_size=0.33, random_state=None, shuffle=True):
    """Split dataset into train and test sets (sublists) based on a test set size.

    Args:
        X(list of list of obj): The list of samples
            The shape of X is (n_samples, n_features)
        y(list of obj): The target y values (parallel to X)
            The shape of y is n_samples
        test_size(float or int): float for proportion of dataset to be in test set (e.g. 0.33 for a 2:1 split)
            or int for absolute number of instances to be in test set (e.g. 5 for 5 instances in test set)
        random_state(int): integer used for seeding a random number generator for reproducible results
        shuffle(bool): whether or not to randomize the order of the instances before splitting

    Returns:
        X_train(list of list of obj): The list of training samples
        X_test(list of list of obj): The list of testing samples
        y_train(list of obj): The list of target y values for training (parallel to X_train)
        y_test(list of obj): The list of target y values for testing (parallel to X_test)
    
    Note:
        Loosely based on sklearn's train_test_split(): https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
    """
    if random_state is not None:
       np.random.seed(random_state)
    
    if shuffle: 
        myutils.randomize_in_place(X, y)

    num_instances = len(X) 
    
    # Check if the test_size is a proportion first
    if isinstance(test_size, float): 
        test_size = math.ceil(num_instances * test_size)
    # At this point, test_size is an integer value, do the splitting
    split_index = num_instances - test_size
    
    return X[:split_index], X[split_index:], y[:split_index], y[split_index:]

def perform_holdout_method(classifier, X, y, test_size=5, random_state=None, shuffle=True, normalize_X=True):
    """Performs a Holdout method to fit data to a classifier

    Args:
        classifier: a classifier defined in myclassifiers.py as part of the mysklearn package
        X (list of list of data): the X dataset
        y (list of data): the y dataset
        test_size (float or int): if float, the ratio of test to dataset, if int, the number of test instances
        random_state (int): the random state to use
        shuffle (bool): whether to shuffle the dataset or not
        normalize_X (bool): whether to normalize the features of X or not

    Returns:
        X_train(list of list of obj): The list of training samples
        X_test(list of list of obj): The list of testing samples
        y_train(list of obj): The list of target y values for training (parallel to X_train)
        y_test(list of obj): The list of target y values for testing (parallel to X_test)
    """
    X_indices, y_indices = [kk for kk in range(len(X))], [kk for kk in range(len(y))]
    # Split the auto dataset into a train and test set
    X_train_indices, X_test_indices, y_train_indices, y_test_indices = train_test_split(X_indices, y_indices, test_size=test_size, random_state=random_state, shuffle=shuffle)   
    
    return fit_classifier(classifier, X, y, X_train_indices, X_test_indices, y_train_indices, y_test_indices, normalize_X=normalize_X)

def fit_classifier(classifier, X, y, X_train_indices, X_test_indices, y_train_indices, y_test_indices, normalize_X=True):
    """Performs a fit to a classifier

    Args:
        classifier: a classifier defined in myclassifiers.py as part of the mysklearn package
        X (list of list of data): the X dataset
        y (list of data): the y dataset
        X_train_indices (list of int): the positions of X to use as train data
        X_test_indices (list of int): the positions of X to use as test data
        y_train_indices (list of int): the positions of y to use as train data
        y_test_indices (list of int): the positions of y to use as test data
        normalize_X (bool): whether to normalize the features of X or not

    Returns:
        X_train(list of list of obj): The list of training samples
        X_test(list of list of obj): The list of testing samples
        y_train(list of obj): The list of target y values for training (parallel to X_train)
        y_test(list of obj): The list of target y values for testing (parallel to X_test)
    """
    # Fetch the training sets
    X_train, y_train = [X[kk] for kk in X_train_indices], [y[kk] for kk in y_train_indices]
    # Fetch the test set for X data
    X_test, y_test = [X[kk] for kk in X_test_indices], [y[kk] for kk in y_test_indices]
    # If selected, normalize the X data:
    if normalize_X:
        X_train, X_test = myutils.normalize_train_and_test_sets(X_train, X_test)
    # Train the classifer 
    classifier.fit(X_train, y_train)
    # Get the classifier predictions for the test set
    y_test_pred = classifier.predict(X_test)

    return X_test_indices, y_test_indices, y_test, y_test_pred

def kfold_cross_validation(X, n_splits=5):
    """Split dataset into cross validation folds.

    Args:
        X(list of list of obj): The list of samples
            The shape of X is (n_samples, n_features)
        n_splits(int): Number of folds.

    Returns:
        X_train_folds(list of list of int): The list of training set indices for each fold
        X_test_folds(list of list of int): The list of testing set indices for each fold

    Notes: 
        The first n_samples % n_splits folds have size n_samples // n_splits + 1, 
            other folds have size n_samples // n_splits, where n_samples is the number of samples
            (e.g. 11 samples and 4 splits, the sizes of the 4 folds are 3, 3, 3, 2 samples)
        Loosely based on sklearn's KFold split(): https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html
    """
    # Create our folds
    folds = [ [] for _ in range(n_splits)]
    # Partition out the instances in X to the folds
    curr_fold = 0
    for X_index in range(len(X)):
        folds[curr_fold].append(X_index) # Storing the index of X in the current fold
        curr_fold = (curr_fold + 1) % n_splits # Increment the current fold

    X_train_folds, X_test_folds =  myutils.get_test_train_sets_for_kfold(folds, n_splits)
    return X_train_folds, X_test_folds

def stratified_kfold_cross_validation(X, y, n_splits=5):
    """Split dataset into stratified cross validation folds.

    Args:
        X(list of list of obj): The list of instances (samples). 
            The shape of X is (n_samples, n_features)
        y(list of obj): The target y values (parallel to X). 
            The shape of y is n_samples
        n_splits(int): Number of folds.
 
    Returns:
        X_train_folds(list of list of int): The list of training set indices for each fold.
        X_test_folds(list of list of int): The list of testing set indices for each fold.

    Notes: 
        Loosely based on sklearn's StratifiedKFold split(): https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html#sklearn.model_selection.StratifiedKFold
    """

    # First, group the indicies of the values in X by their corresp. values in y
    _, grouped_indices = myutils.index_group_by_y(X, y) # Don't need the group names for this
    # Create our folds
    folds = [ [] for _ in range(n_splits)]
    # Partition out the grouped indices into the folds - this is the stratification
    curr_fold = 0
    for group in grouped_indices:
        for index in group:
            folds[curr_fold].append(index)
            curr_fold = (curr_fold + 1) % n_splits
   
    X_train_folds, X_test_folds =  myutils.get_test_train_sets_for_kfold(folds, n_splits)
    return X_train_folds, X_test_folds

def random_stratified_train_test_split(X, y, test_size=0.33):
    """Split dataset into stratified random sets

    Args:
        X(list of list of obj): The list of instances (samples). 
            The shape of X is (n_samples, n_features)
        y(list of obj): The target y values (parallel to X). 
            The shape of y is n_samples
        test_size(float or int): float for proportion of dataset to be in test set (e.g. 0.33 for a 2:1 split)
            or int for absolute number of instances to be in test set (e.g. 5 for 5 instances in test set)
 
    Returns:
        X_test(list of list of obj): The list of testing samples
        y_test(list of obj): The list of target y values for testing (parallel to X_test)
        X_remainder(list of list of obj): The list of training/validation samples
        y_remainder(list of obj): The list of target y values for training/validation (parallel to X_train)
    """
    # Group the indicies of the values in X by their corresp. values in y
    _, grouped_indices = myutils.index_group_by_y(X, y) # Don't need the group names for this
    test_indices = []
    remainder_indices = []

    # Check if the test_size is an integer first and convert to a fraction if needed
    if isinstance(test_size, int): 
        test_size = test_size / len(y)
    
    # Go through each group and add the instances to the respective group
    for group in grouped_indices:
        group_len = len(group)
        test_cutoff_index = int(group_len * test_size)   
        # Shuffle the group up 
        random.shuffle(group)
        test_indices.append(group[:test_cutoff_index])
        remainder_indices.append(group[test_cutoff_index+1:])
    
    # Go through and get the X and y sets for each group
    X_test, X_remainder = [], []
    y_test, y_remainder = [], []
    for index in test_indices:
        X_test.append(X[index])
        y_test.append(y[index])
    for index in remainder_indices:
        X_remainder.append(X[index])
        y_remainder.append(y[index])
    
    return X_test, y_test, X_remainder, y_remainder

def confusion_matrix(y_true, y_pred, labels):
    """Compute confusion matrix to evaluate the accuracy of a classification.

    Args:
        y_true(list of obj): The ground_truth target y values
            The shape of y is n_samples
        y_pred(list of obj): The predicted target y values (parallel to y_true)
            The shape of y is n_samples
        labels(list of str): The list of all possible target y labels used to index the matrix

    Returns:
        matrix(list of list of int): Confusion matrix whose i-th row and j-th column entry 
            indicates the number of samples with true label being i-th class 
            and predicted label being j-th class

    Notes:
        Loosely based on sklearn's confusion_matrix(): https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html
    """
    # First, ensure the labels are all strings
    labels = [str(kk) for kk in labels ]
    # Create a confusion matrix of all zeros to start
    matrix = [ [0 for _ in range(len(labels))] for _ in range(len(labels)) ]

    # Go through the true/predicted and add to the confusion matrix
    for kk in range(len(y_pred)):
        if y_pred[kk] != None:
            true_row_index = labels.index(str(y_true[kk]))
            pred_col_index = labels.index(str(y_pred[kk]))
            matrix[true_row_index][pred_col_index] += 1

    return matrix