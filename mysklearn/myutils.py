import random
import math
from tabulate import tabulate 

def compute_euclidean_distance(v1, v2):
    """
        Computes the Euclidean Distance between two parallel vectors

        Args: 
            v1 (list of numeric): the first list to use
            v2 (list of numeric): the second list to use

        Returns:
            the euclidean distance between the two vectors 
    """
    assert len(v1) == len(v2)
    return math.sqrt(sum([(v2[kk] - v1[kk]) ** 2 for kk in range(len(v1))]))

def compute_categorical_distance(v1, v2):
    """ Computes the distance between two parallel vectors

        Args: 
            v1 (list of string): the first list to use
            v2 (list of string): the second list to use

        Returns:
            the distance between the two vectors [0 or 1]
    """
    assert len(v1) == len(v2)
    return math.sqrt(sum([ 1 if v1[kk] == v2[kk] else 0 for kk in range(len(v1)) ]))

def compute_continuous_and_categorical_distance(v1, v2):
    """
        Handles the case where the two vectors contain both categorical and
        continuous data. If either attribute is a string, the categorical distance
        is found. If both entries are numeric, a euclidean distance is used.

        Args: 
            v1 (list of numeric/string): the first list to use
            v2 (list of numeric/string): the second list to use
        
        Returns:
            the euclidean/categorical distance between the two vectors 

        Notes: the numeric entries should be propery normalized such that the 
            maxium distance between any two instances for any attribute is 1
    """
    assert len(v1) == len(v2)
    sum = 0
    for kk in range(len(v1)):
        if type(v1[kk]) is str or type(v2[kk]) is str: # Use categorical distance
            sum += compute_categorical_distance([str(v1[kk])], [str(v2[kk])])**2
        else:
            sum += compute_euclidean_distance([v1[kk]], [v2[kk]])**2
    return math.sqrt(sum)

def get_column_min_max(data, col_index):
    """
        Gets the min and the max of a column in a datatable

        Args: 
            data (list of list): the datatable
            col_index (int): the index of the column of interest

        Returns:
            min of the column
            max of the column
    """
    col_vals = []
    for row in data:
        col_vals.append(row[col_index])
    return min(col_vals), max(col_vals)

def normalize_train_and_test_sets(X_train, X_test):
    """
        Normalizes the features of the X_train and X_test sets

        Args: 
            X_train (list of list): the training data
            X_test (list of list): the testing data

        Returns:
            norm_X_train: the normalized training data
            norm_X_test: the normailized testing data 
    """
    n_test_samples = len(X_test)
    n_features = len(X_test[0])
    n_train = len(X_train)

    # first, make copy of the train data, and scale each column 
    X_train_col_mins, X_train_col_maxs = [], []
    for col in range(n_features):
        min_val, max_val = get_column_min_max(X_train, col)
        X_train_col_mins.append(min_val)
        X_train_col_maxs.append(max_val)
    norm_X_train = [ [(X_train[row][col] - X_train_col_mins[col]) / (X_train_col_maxs[col] - X_train_col_mins[col]) for col in range(n_features) ] for row in range(n_train) ]
    # Scale each value in the test set as well
    norm_X_test = [ [(X_test[row][col] - X_train_col_mins[col]) / (X_train_col_maxs[col] - X_train_col_mins[col]) for col in range(n_features) ] for row in range(n_test_samples) ]
    
    return norm_X_train, norm_X_test

def get_categorical_frequencies(column):
    """
        Gets the categorical frequencies of the values in a list

        Args: 
            column (list): the column with the categorical data

        Returns:
            values: the unique categorical values in the column
            counts: the counts of the corresponing categorical values in the column
    """
    values, counts = [], []

    for value in column:
        if value not in values:
            # haven't seen this value before
            values.append(value)
            counts.append(1)
        else:
            # Get the index of the occurance
            index = values.index(value)
            counts[index] += 1

    return values, counts

def group_by(table, header, group_by_col_name):
    """ Gets the subtables grouped by the values in a column

    Args:
        group_by_col_name(str): the name of the column to get the subtables from

    Returns:
        group_names: the names of the groups
        group_subtables: the subtables of the group
    """

    col_index = header.index(group_by_col_name)
    col = get_column(table, col_index)

    # We need the unique values for out group by column
    group_names = sorted(list(set(col))) # e.g. 74, 75, 76, 77
    group_subtables = [[] for _ in group_names] # [ [], [], [], [] ]

    # algorithm: walk thorugh each row and assign to the appropriate subtable based on its group_by_col_name
    for row in table:
        group_by_value = row[col_index]
        group_by_index = group_names.index(group_by_value)
        group_subtables[group_by_index].append(row)

    return group_names, group_subtables

def get_class_count(table, class_label):
    # Assumes the class label is the last value in each row
    count = 0
    for row in table:
        if row[-1] == class_label:
            count += 1
    return count

def randomize_in_place(alist, parallel_list=None): 
    """
        Randomizes up to two parallel lists in parallel

        Args: 
            alist (list): the primary list
            parallel_list (list): the optional secondary parallel list
    """
    for i in range(len(alist)):
        # generate a random index to swap the element at i with
        rand_index = random.randrange(0, len(alist)) # [ 0, len(alist) )
        alist[i], alist[rand_index] = alist[rand_index], alist[i]
        if parallel_list is not None:
            parallel_list[i], parallel_list[rand_index] = parallel_list[rand_index], parallel_list[i]      

def index_group_by_y(X, y):
    """
        Groups the indices of X by their corresp values in y

        Args: 
            X (list of list): the X data
            y (list): the y data

        Returns:
            group_names: the class labels of the groups
            group_subtables: the subtables of the X data 
    """
    # We need the unique values for out group by column
    group_names = list(set(y)) # e.g. A, D, C, B
    group_subtables = [[] for _ in group_names] # [ [], [], [], [] ]

    # algorithm: walk thorugh each row and assign to the appropriate subtable based on its group_by_col_name
    for kk in range(len(X)):
        group_by_value = y[kk]
        group_by_index = group_names.index(group_by_value)
        group_subtables[group_by_index].append(kk)

    return group_names, group_subtables

def get_test_train_sets_for_kfold(folds, n_splits):
    """
        Gets the train and test sets for kfold from the existing folds

        Args: 
            folds (list of list): the folds of the train/test data
            n_splits (int): the number of splits

        Returns:
            X_train_folds: the folds for the X training data based on n_splits
            X_test_folds: the folds for the X testing data based on n_splits
    """
    # Need to divy the folds into train and test sets - hold the indices of X. 
    # Use every fold once as a test set and the others as the train sets      
    X_train_folds = [ [] for _ in range(n_splits) ]
    X_test_folds = [ [] for _ in range(n_splits) ]
    for kk in range(n_splits):
        # The test set is the current fold
        X_test_folds[kk] = folds[kk] 
        # The training set is all other folds
        for ii in range(kk):
            X_train_folds[kk] += folds[ii]
        for ii in range(kk+1, n_splits, 1):
            X_train_folds[kk] += folds[ii]
    
    return X_train_folds, X_test_folds

def print_prediction_results(true_y, pred_y, y_instances=None, title="Prediction Results"):
    
    # y instances is optionally the full instance in a dataset, 
    # not necessarily only the class label (y)
    print("============================================================")
    print(title)
    print("============================================================")
    for kk in range(len(true_y)):
        print("instance:", y_instances[kk])
        print("\tpredicted class:   ", pred_y[kk])
        print("\ttrue class:        ", true_y[kk])

def print_accuracy_results(linear_percent_correct, kNN_percent_correct, k_subsamples, train_test_ratio, title="Accuracy Results"):
    print("============================================================")
    print(title)
    print("============================================================")
    print("Random Subsample (k="+str(k_subsamples)+", "+str(train_test_ratio)+" Train/Test)")
    print("Linear Regression: accuracy = "+str(linear_percent_correct)+", error rate = "+str(1-linear_percent_correct))
    print("k Nearest Neighbors: accuracy = "+str(kNN_percent_correct)+", error rate = "+str(1-kNN_percent_correct))

def print_single_accuracy_results(predictive_accuracy, k_subsamples, train_test_ratio, title="Accuracy Results"):
    print("============================================================")
    print(title)
    print("============================================================")
    print("Random Subsample (k="+str(k_subsamples)+", "+str(train_test_ratio)+" Train/Test)")
    print("Naive Bayes: accuracy = "+str(predictive_accuracy)+", error rate = "+str(1-predictive_accuracy))

def print_crossVal_accuracy_results(linear_predictive_accuracy, kNN_predictive_accuracy, linear_strat_predictive_accuracy, kNN_strat_predictive_accuracy, k_cross_validation, title="Accuracy Results"):
    print("============================================================")
    print(title)
    print("============================================================")
    print(str(k_cross_validation)+"-Fold Cross Validation")
    print("Linear Regression: accuracy = "+str(linear_predictive_accuracy)+", error rate = "+str(1-linear_predictive_accuracy))
    print("k Nearest Neighbors: accuracy = "+str(kNN_predictive_accuracy)+", error rate = "+str(1-kNN_predictive_accuracy))
    print("")
    print("Stratified "+str(k_cross_validation)+"-Fold Cross Validation")
    print("Linear Regression: accuracy = "+str(linear_strat_predictive_accuracy)+", error rate = "+str(1-linear_strat_predictive_accuracy))
    print("k Nearest Neighbors: accuracy = "+str(kNN_strat_predictive_accuracy)+", error rate = "+str(1-kNN_strat_predictive_accuracy))

def print_single_crossVal_accuracy_results(predictive_accuracy, strat_predictive_accuracy, k_cross_validation, title="Accuracy Results"):
    print("============================================================")
    print(title)
    print("============================================================")
    print(str(k_cross_validation)+"-Fold Cross Validation")
    print("Naive Bayes: accuracy = "+str(predictive_accuracy)+", error rate = "+str(1-predictive_accuracy))
    print("")
    print("Stratified "+str(k_cross_validation)+"-Fold Cross Validation")
    print("Naive Bayes: accuracy = "+str(strat_predictive_accuracy)+", error rate = "+str(1-strat_predictive_accuracy))

def print_stratified_crossVal_results(pred_accuracies, labels, k_cross_validation, title="Accuracy Results"):
    print("============================================================")
    print(title)
    print("============================================================")
    print("Stratified "+str(k_cross_validation)+"-Fold Cross Validation")
    for kk in range(len(pred_accuracies)):
        print(str(labels[kk])+": accuracy = "+str(pred_accuracies[kk])+", error rate = "+str(1-pred_accuracies[kk]))

def print_crossVal_results(pred_accuracies, labels, k_cross_validation, title="Accuracy Results"):
    print("============================================================")
    print(title)
    print("============================================================")
    print(str(k_cross_validation)+"-Fold Cross Validation")
    for kk in range(len(pred_accuracies)):
        print(str(labels[kk])+": accuracy = "+str(pred_accuracies[kk])+", error rate = "+str(1-pred_accuracies[kk]))


def print_confusion_matrix(matrix, labels, title="", table_header=""):
    """
        Prints a confusion matrix in a nice format with some additional information

        Args: 
            matrix (list of list): the confusion matrix
            labels (list of str): the labels for each class value
    """
    print(title)
    new_table = []
    for kk in range(len(matrix)):
        # Append the row total and the Recognition to each row in the table
        curr_row = []
        curr_row += [str(labels[kk])]
        curr_row += matrix[kk]
        if sum(matrix[kk]) == 0:  
            curr_row += [0, 0]
        else:
            curr_row += [sum(matrix[kk]), 100 * matrix[kk][kk] / sum(matrix[kk])]
        new_table.append(curr_row)

    labels = [table_header] + labels + ["Total", "Recognition (%)"]
    print(tabulate(new_table, headers=labels, tablefmt="fancy_grid"))
    print("\n")

def proper_round(num, dec=0):
    """
        Does a proper round on an integer

        Args: 
            num (numeric): the number to round
            dec (int)): the number of decimal points to keep
        
        Returns:
            the rounded number
    """
    # See if the number has a decimal at all
    if type(num) is int:
        return num

    num = str(num)[:str(num).index('.')+dec+2]
    if num[-1]>='5':
        return float(num[:-2-(not dec)]+str(int(num[-2-(not dec)])+1))
    return float(num[:-1])

def classify_continuous_data(data_col, cutoffs, labels, lower_inclusive_upper_exclusive=True):
    """
        Classifyies the continuous dataset in data_col into bins specified by cutoffs  
        where each bin has a class name specified by label

        Args:
            data_col (list of numeric): the column to classify
            cutoffs (list of numeric): list of cutoff points for the classification
            labels (list): labels to classify to
            lower_inclusive_upper_exclusive (bool): whether or not to use the lower bound as inclusive or not
        
        Returns:
            The corresponding data column with the values classified to categorical values
    """
    classified_data_col = list(range(len(data_col)))

    if lower_inclusive_upper_exclusive: 
        # We want to use the lower bounds as inclusive and the upper bounds as exclusive
        for value_index in range(len(data_col)):
            value = data_col[value_index]
            for kk in range(len(cutoffs)-1):
                if value >= cutoffs[kk] and kk == (len(cutoffs) - 2): # Last bin (max value)
                    classified_data_col[value_index] = labels[kk]
                elif value >= cutoffs[kk] and value < cutoffs[kk+1]:
                    classified_data_col[value_index] = labels[kk]
    else:
        # Use the lower bounds as exclusive and the uppen bounds as inclusive
        for value_index in range(len(data_col)):
            value = data_col[value_index]
            for kk in range(len(cutoffs)-1):
                if value > cutoffs[kk] and kk == (len(cutoffs) - 2): # Last bin (max value)
                    classified_data_col[value_index] = labels[kk]
                elif value > cutoffs[kk] and value <= cutoffs[kk+1]:
                    classified_data_col[value_index] = labels[kk]
    
    return classified_data_col

def get_percent_correct(y_test_pred, y_test_actual):
    """
        Determines the percentage correct of the predicted and actual y values
    """
    correct_count = 0
    for kk in range(len(y_test_actual)):
        if y_test_actual[kk] == y_test_pred[kk]:
            correct_count += 1
    
    return correct_count / len(y_test_actual)

def get_column(table, col_index):
    col = []
    for row in table:
        col.append(row[col_index])
    return col

def remove_column(table, col_index):
    new_table = []
    for row in table:
        new_row = row[:col_index] + row[col_index+1:]
        new_table.append(new_row)
    return new_table

def get_num_matches(X, y, X_val, X_col_index, y_val):

    num_matches = 0
    for kk in range(len(X)):
        curr_X = X[kk][X_col_index]
        curr_y = y[kk]
        if curr_X == X_val and curr_y == y_val:
            num_matches += 1
            
    return num_matches




def tdidt_predict(header, instance, tree):
    info_type = tree[0]
    if info_type == "Attribute":
        instance_attribute_value = instance[header.index(tree[1])]
        # Now I need to find which branch to follow recursively
        for kk in range(2, len(tree)):
            value_list = tree[kk]
            if value_list[1] == instance_attribute_value:
                # We have a match - recurse through the rest of tree
                return tdidt_predict(header, instance, value_list[2])
    else: # "Leaf"
        return tree[1]