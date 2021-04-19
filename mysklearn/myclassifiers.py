import mysklearn.myutils as myutils
import operator
import random
import math
import os

import mysklearn.mypytable
from mysklearn.mypytable import MyPyTable 
import mysklearn.myevaluation as myevaluation

class MySimpleLinearRegressor:
    """Represents a simple linear regressor.

    Attributes: 
        slope(float): m in the equation y = mx + b
        intercept(float): b in the equation y = mx + b

    Notes:
        Loosely based on sklearn's LinearRegression: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html
        Terminology: instance = sample = row and attribute = feature = column
    """
    
    def __init__(self, slope=None, intercept=None):
        """Initializer for MySimpleLinearRegressor.

        Args:
            slope(float): m in the equation y = mx + b (None if to be computed with fit())
            intercept(float): b in the equation y = mx + b (None if to be computed with fit())
        """
        self.slope = slope 
        self.intercept = intercept

    def fit(self, X_train, y_train):
        """Fits a simple linear regression line to X_train and y_train.

        Args:
            X_train(list of list of numeric vals): The list of training samples
                The shape of X_train is (n_train_samples, n_features)
                Note that n_features for simple regression is 1, so each sample is a list 
                    with one element e.g. [[0], [1], [2]]
            y_train(list of numeric vals): The target y values (parallel to X_train) 
                The shape of y_train is n_train_samples
        """
        # Running a linear regression:
        X_sum, y_sum = 0, 0
        for kk in range(len(X_train)):
            X_sum += X_train[kk][0]
            y_sum += y_train[kk]
        mean_x = X_sum / len(X_train)
        mean_y = y_sum / len(y_train)
        # Solve for the slope
        
        self.slope = sum([(X_train[kk][0] - mean_x) * (y_train[kk] - mean_y) for kk in range(len(X_train))]) / \
            sum([(X_train[kk][0] - mean_x) ** 2 for kk in range(len(X_train))])
        # Solve for the y intercept of the least-squares line
        self.intercept = mean_y - self.slope * mean_x

    def predict(self, X_test):
        """Makes predictions for test samples in X_test.

        Args:
            X_test(list of list of numeric vals): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)
                Note that n_features for simple regression is 1, so each sample is a list 
                    with one element e.g. [[0], [1], [2]]

        Returns:
            y_predicted(list of numeric vals): The predicted target y values (parallel to X_test)
        """
        return [self.slope * x[0] + self.intercept for x in X_test]

class MyKNeighborsClassifier:
    """Represents a simple k nearest neighbors classifier.

    Attributes:
        n_neighbors(int): number of k neighbors
        X_train(list of list of numeric vals): The list of training instances (samples). 
                The shape of X_train is (n_train_samples, n_features)
        y_train(list of obj): The target y values (parallel to X_train). 
            The shape of y_train is n_samples

    Notes:
        Loosely based on sklearn's KNeighborsClassifier: https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
        Terminology: instance = sample = row and attribute = feature = column
        Assumes data has been properly normalized before use.
    """
    def __init__(self, n_neighbors=3):
        """Initializer for MyKNeighborsClassifier.

        Args:
            n_neighbors(int): number of k neighbors
        """
        self.n_neighbors = n_neighbors
        self.X_train = None 
        self.y_train = None

    def fit(self, X_train, y_train):
        """Fits a kNN classifier to X_train and y_train.

        Args:
            X_train(list of list of numeric vals): The list of training instances (samples). 
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples

        Notes:
            Since kNN is a lazy learning algorithm, this method just stores X_train and y_train
        """
        self.X_train = X_train 
        self.y_train = y_train 

    def kneighbors(self, X_test):
        """Determines the k closest neighbors of each test instance.

        Args:
            X_test(list of list of numeric vals): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)

        Returns:
            distances(list of list of float): 2D list of k nearest neighbor distances 
                for each instance in X_test
            neighbor_indices(list of list of int): 2D list of k nearest neighbor
                indices in X_train (parallel to distances)
        """
        n_test_samples = len(X_test)

        # Ultimately creating a 2D list to store all the results for each test value
        neighbor_indices = [ [] for _ in range(n_test_samples)]
        distances = [ [] for _ in range(n_test_samples)]
        
        for test_val_index in range(n_test_samples):
            #for i, instance in enumerate(self.X_train):
            for i in range(len(self.X_train)):
                instance = self.X_train[i]
                # Compute the distance 
                dist = myutils.compute_continuous_and_categorical_distance(instance, X_test[test_val_index])
                # append the row index
                instance.append(i)
                # append the dist
                instance.append(dist)
            # Sort train by distance
            train_sorted = sorted(self.X_train, key=operator.itemgetter(-1)) # Use the last item in the row to sort
            # Grab the top k
            top_k = train_sorted[:self.n_neighbors]
            # Add the top dists and indices to the matrix
            for ii in range(self.n_neighbors):
                distances[test_val_index].append(top_k[ii][-1])
                neighbor_indices[test_val_index].append(top_k[ii][-2])
            # Should remove the extra values from the end of each train value
            for instance in self.X_train:
                del instance[-2:]

        return distances, neighbor_indices

    def predict(self, X_test):
        """Makes predictions for test instances in X_test.

        Args:
            X_test(list of list of numeric vals): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)

        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """
        n_test_samples = len(X_test)
        # First, get the k nearest neighbors for each test value
        _, top_k_indices = self.kneighbors(X_test)

        # Create a list for the predicted y values
        y_predicted = list(range(n_test_samples))

        for test_val_index in range(n_test_samples):
            # Get the top n class values
            top_y_vals = []
            for index in top_k_indices[test_val_index]:
                top_y_vals.append(self.y_train[index])
            # Get the frequencies of each group
            names, counts = myutils.get_categorical_frequencies(top_y_vals)
            # The predicted y val is the one with the highest frequency
            max_count_index = counts.index(max(counts))
            y_predicted[test_val_index] = names[max_count_index]

        return y_predicted 

class MyNaiveBayesClassifier:
    """Represents a Naive Bayes classifier.

    Attributes:
        X_train(list of list of obj): The list of training instances (samples). 
                The shape of X_train is (n_train_samples, n_features)
        y_train(list of obj): The target y values (parallel to X_train). 
            The shape of y_train is n_samples
        priors(MyPyTable): The prior probabilities computed for each
            label in the training set.
        posteriors(MyPyTable): The posterior probabilities computed for each
            attribute value/label pair in the training set.

    Notes:
        Loosely based on sklearn's Naive Bayes classifiers: https://scikit-learn.org/stable/modules/naive_bayes.html
        Terminology: instance = sample = row and attribute = feature = column
    """
    def __init__(self):
        """Initializer for MyNaiveBayesClassifier.

        """
        self.X_train = None 
        self.y_train = None
        self.priors = None 
        self.posteriors = None

    def fit(self, X_train, y_train):
        """Fits a Naive Bayes classifier to X_train and y_train.

        Args:
            X_train(list of list of obj): The list of training instances (samples). 
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples

        Notes:
            Since Naive Bayes is an eager learning algorithm, this method computes the prior probabilities
                and the posterior probabilities for the training data.
            You are free to choose the most appropriate data structures for storing the priors
                and posteriors.
        """
        self.X_train = X_train
        self.y_train = y_train

        # First, we need to compute the priors
        unique_classes, classes_counts = myutils.get_categorical_frequencies(y_train)
        num_instances = len(y_train)
        prior_header = ["class label","prior"]
        prior_data = []
        for kk in range(len(unique_classes)):
            prior_data.append( [ unique_classes[kk], (classes_counts[kk] / num_instances) ])
        # Save the prior data to a MyPyTable
        self.priors = MyPyTable(column_names=prior_header, data=prior_data)

        # Now, we need to compute the posteriors
        post_header = ["attribute column", "attribute value"]
        for kk in range(len(unique_classes)):
            post_header += ["class = "+str(unique_classes[kk])]
        post_data = []
        # Get all the unique attributes
        num_cols = len(X_train[0])
        for kk in range(num_cols):
            # Get the current column
            attr_col = myutils.get_column(X_train, kk)
            # Get the unique values of the attribute in the column and its counts
            unique_attr, _ = myutils.get_categorical_frequencies(attr_col)
            # Now, go through each unique attr value and get the counts of each unique class labels with it
            for curr_attr in unique_attr:
                num_matches_array = []
                # Get the count of values that have the specified attribute value for each class label
                for class_label in unique_classes:
                    num_matches_array.append(myutils.get_num_matches(X_train, y_train, curr_attr, kk, class_label))
                # Compute the posteriors
                curr_posteriors = [ num_matches_array[index] / classes_counts[index] for index in range(len(classes_counts))]
                # Append the posteriors to the table
                post_data.append([kk, curr_attr] + curr_posteriors)
        # Save the posteriors as a MyPyTable
        self.posteriors = MyPyTable(column_names=post_header, data=post_data)

    def predict(self, X_test):
        """Makes predictions for test instances in X_test.

        Args:
            X_test(list of list of obj): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)

        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """
        # Store the predicted values
        y_predicted = []

        # Get the unique classes and their corresp priors (parallel lists)
        class_labels = self.priors.get_column("class label")
        # Go through each test instance
        for kk in range(len(X_test)):
            # Create a list to hold thr probabilities for each class label - initialize to be the prior
            probabilities = self.priors.get_column("prior")
            # Now multiply by each of the posteriors
            for label_index in range(len(class_labels)):
                # Get the posterior for each attribute and the class label
                for col_index in range(len(X_test[kk])):
                    # Get the current attribute val
                    curr_attr = X_test[kk][col_index]
                    # Search the posterior table for: (attribute col == kk), (attribute value == curr_attr)
                    curr_posterior_row = self.posteriors.get_instance_from_key_pairs(["attribute column", "attribute value"], [col_index, curr_attr])
                    # Get the posterior for the current class label and multiply to the probability
                    probabilities[label_index] *= curr_posterior_row[0][label_index + 2] # offset of 2 for the attr col and attr val columns
            # Find the index of the largest probability and return that as the predicted label
            largest_prob_index = probabilities.index(max(probabilities))
            y_predicted.append(class_labels[largest_prob_index])

        return y_predicted

class MyZeroRClassifier:
    """Represents a baseline zero R classifier

    Attributes:
        X_train(list of list of numeric vals): The list of training instances (samples). 
                The shape of X_train is (n_train_samples, n_features)
        y_train(list of obj): The target y values (parallel to X_train). 
            The shape of y_train is n_samples

    Notes:
        Classifies by only returning the class label with the highest frequency. If 
        multiple class labels have the same highest frequency, then one is chosen at random.
    """
    def __init__(self):
        self.X_train = None 
        self.y_train = None

    def fit(self, X_train, y_train):
        """Fits a Zero R classifier to X_train and y_train.

        Args:
            X_train(list of list of obj): The list of training instances (samples). 
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples
        """
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X_test):
        """Makes predictions for test instances in X_test.

        Args:
            X_test(list of list of obj): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)

        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """
        unique_classes, freqs = myutils.get_categorical_frequencies(self.y_train)
        max_freq = max(freqs)
        max_freq_bools = [freq==max_freq for freq in freqs]
        max_freq_indices = []
        for kk in range(len(max_freq_bools)):
            if max_freq_bools[kk] == True:
                max_freq_indices.append(kk)
                
        rand_index = max_freq_indices[0]
        if len(max_freq_indices) > 1:
            rand_index = random.randrange(0,len(max_freq_indices),1)
        return [ unique_classes[rand_index] for _ in range(len(X_test)) ] 

class MyRandomClassifier:
    """Represents a baseline zero R classifier

    Attributes:
        X_train(list of list of numeric vals): The list of training instances (samples). 
                The shape of X_train is (n_train_samples, n_features)
        y_train(list of obj): The target y values (parallel to X_train). 
            The shape of y_train is n_samples

    Notes:
        Classifies by only returning the class label with the highest frequency. If 
        multiple class labels have the same highest frequency, then one is chosen at random.
    """
    def __init__(self):
        self.X_train = None 
        self.y_train = None

    def fit(self, X_train, y_train):
        """Fits a Zero R classifier to X_train and y_train.

        Args:
            X_train(list of list of obj): The list of training instances (samples). 
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples
        """
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X_test):
        """Makes predictions for test instances in X_test.

        Args:
            X_test(list of list of obj): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)

        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """
        unique_classes, freqs = myutils.get_categorical_frequencies(self.y_train)
        weights = [ freq / len(self.y_train) for freq in freqs ]
        weights_cutoffs = [0 for _ in range(len(weights)+1)]
        for kk in range(len(weights)):
            weights_cutoffs[kk+1] = weights_cutoffs[kk] + weights[kk]
        # Compute some random numbers to get the indices from
        indices = [random.uniform(0, 1) for _ in range(len(X_test))]
        for ii in range(len(indices)):
            for kk in range(len(weights_cutoffs)-1):
                if indices[ii] >= weights_cutoffs[kk] and indices[ii] <= weights_cutoffs[kk+1]:
                    indices[ii] = kk
                    break

        # Return the classes from the indices
        return [unique_classes[index] for index in indices]

class MyDecisionTreeClassifier:
    """Represents a decision tree classifier.

    Attributes:
        X_train(list of list of obj): The list of training instances (samples). 
                The shape of X_train is (n_train_samples, n_features)
        y_train(list of obj): The target y values (parallel to X_train). 
            The shape of y_train is n_samples
        tree(nested list): The extracted tree model.

    Notes:
        Loosely based on sklearn's DecisionTreeClassifier: https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
        Terminology: instance = sample = row and attribute = feature = column
    """
    def __init__(self):
        """Initializer for MyDecisionTreeClassifier.

        """
        self.X_train = None 
        self.y_train = None
        self.tree = None
        self.header = None
        self.attribute_domain = None

    def fit(self, X_train, y_train):
        """Fits a decision tree classifier to X_train and y_train using the TDIDT (top down induction of decision tree) algorithm.

        Args:
            X_train(list of list of obj): The list of training instances (samples). 
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples

        Notes:
            Since TDIDT is an eager learning algorithm, this method builds a decision tree model
                from the training data.
            Build a decision tree using the nested list representation described in class.
            Store the tree in the tree attribute.
            Use attribute indexes to construct default attribute names (e.g. "att0", "att1", ...).
        """
        # First, make sure everything is represented as a string
        for row in range(len(X_train)):
            y_train[row] = str(y_train[row])
            for col in range(len(X_train[0])):
                X_train[row][col] = str(X_train[row][col])

        # Zip together the X and y values
        train = [X_train[i] + [y_train[i]] for i in range(len(X_train))]
        # Create a generic header
        self.header = ["att"+str(kk) for kk in range(len(X_train[0]))]
        # Get the attribute domain dictionary 
        self.attribute_domains = self.__get_attribute_domain__(X_train, self.header)

        # Generate the tree
        available_attributes = self.header.copy()
        self.tree = self.__tdidt__(train, available_attributes)
        #self.visualize_tree("tree_DOT_file", "tree_visual")
        self.tree = self.__sort_tree__(self.tree)
        #print("\n\n\nTree:",self.tree,"\n\n\n")

    def predict(self, X_test):
        """Makes predictions for test instances in X_test.

        Args:
            X_test(list of list of obj): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)

        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """
        # First, make sure everything is represented as a string
        for row in range(len(X_test)):
            for col in range(len(X_test[0])):
                X_test[row][col] = str(X_test[row][col])

        y_predicted = []
        for kk in range(len(X_test)):
            try:
                y_predicted.append(self.__get_prediction__(X_test[kk], self.tree.copy()))
            except:
                y_predicted.append(None)

        return y_predicted

    def print_decision_rules(self, attribute_names=None, class_name="class"):
        """Prints the decision rules from the tree in the format "IF att == val AND ... THEN class = label", one rule on each line.

        Args:
            attribute_names(list of str or None): A list of attribute names to use in the decision rules
                (None if a list is not provided and the default attribute names based on indexes (e.g. "att0", "att1", ...) should be used).
            class_name(str): A string to use for the class name in the decision rules
                ("class" if a string is not provided and the default name "class" should be used).
        """
        # traverse the tree and save all the decisions to an array
        print("\n====================================================================")
        print("Decision Rules:")
        print("====================================================================\n")
        print(self.__get_decision_rules__(self.tree.copy(), class_name, attribute_names))
        print("====================================================================\n")

    # BONUS METHOD
    def visualize_tree(self, dot_fname, pdf_fname, attribute_names=None):
        """BONUS: Visualizes a tree via the open source Graphviz graph visualization package and its DOT graph language (produces .dot and .pdf files).

        Args:
            dot_fname(str): The name of the .dot output file.
            pdf_fname(str): The name of the .pdf output file generated from the .dot file.
            attribute_names(list of str or None): A list of attribute names to use in the decision rules
                (None if a list is not provided and the default attribute names based on indexes (e.g. "att0", "att1", ...) should be used).

        Notes: 
            Graphviz: https://graphviz.org/
            DOT language: https://graphviz.org/doc/info/lang.html
            You will need to install graphviz in the Docker container as shown in class to complete this method.
        """
        # First, turn the tree into a graph format specified by the DOT protocols
        dot_file_string, _ = self.__get_DOT_file__(self.tree.copy(), attribute_names) 
        dot_file_string = "graph g {\n" + dot_file_string + "\n}"
        # Write this to the specified dot file
        dot_file = open(dot_fname+".dot", "w")
        dot_file.write(dot_file_string)
        dot_file.close()
        # Run the DOT file command to create the PDF Visualization
        visaulize_command = "dot -Tpdf -o " + pdf_fname + ".pdf " + dot_fname + ".dot"
        os.popen(visaulize_command)  
        print("\nYour Tree Visualization has been saved as a pdf to the file:", pdf_fname+".pdf\n")      
    
    def __get_DOT_file__(self, curr_tree, attribute_names, file_contents="", node_index=0, parent_node_index=0):
        info_type = curr_tree[0]
        if info_type == "Attribute":
            new_node_index = node_index

            att_name = curr_tree[1] # This is going to have the form "att0", "att1", etc. use the provided names if chosen
            if attribute_names != None:
                att_name = attribute_names[self.header.index(att_name)]

            parent_node_name = "node" + str(parent_node_index)
            # Check if the root node -> add it. Otherwise don't add the node here            
            if parent_node_index == 0: # Root node
                file_contents += ("\t" + parent_node_name + " [label=\"" + att_name + "\", shape=box];\n")
            
            # Go through all branches and add them
            for kk in range(2, len(curr_tree)):
                value_list = curr_tree[kk]
                value_name = value_list[1] # Name of the current "branch"

                # Get the names of all the nodes it connects to
                for ii in range(2, len(value_list)):
                    # Add the sub-attribute as a node
                    sub_attr_name = value_list[ii][1] # Name of an attribute (or leaf) below it
                    new_node_index += 1
                    child_node_name = "node" + str(new_node_index)
                    child_node_index = new_node_index
                    if value_list[ii][0] == "Leaf": # Leaf Node
                        file_contents += ("\t" + child_node_name + " [label=\"" + sub_attr_name + "\"];\n")
                    else: # Normal Node
                        # Since normal node, the sub_attr_name has the form "att0", "att1", etc. use the provided names if chosen
                        if attribute_names != None:
                            sub_attr_name = attribute_names[self.header.index(sub_attr_name)]

                        file_contents += ("\t" + child_node_name + " [label=\"" + sub_attr_name + "\", shape=box];\n")
                        file_contents, new_node_index = self.__get_DOT_file__(value_list[2], attribute_names, file_contents=file_contents, node_index=new_node_index, parent_node_index=child_node_index)
                    # Add the branch between the two nodes
                    file_contents += ("\t" + parent_node_name + "--" + child_node_name + " [label=\"" + value_name + "\"];\n")
        
        return file_contents, new_node_index

    def __get_decision_rules__(self, curr_tree, class_label, attribute_names, decision_rules="", depth=0):
        info_type = curr_tree[0]
        if info_type == "Attribute":
            new_depth = depth+1
            
            # Get the name of the current attribute
            att_name = curr_tree[1] # This is going to have the form "att0", "att1", etc. use the provided names if chosen
            if attribute_names != None:
                att_name = attribute_names[self.header.index(att_name)]

            # Now follow each branch recursively
            new_decision_rules = ""
            for kk in range(2, len(curr_tree)):
                value_list = curr_tree[kk]
                # Get the name of the current value
                value_name = value_list[1]
                # Create the rule string
                if new_depth > 1:
                    new_rule = decision_rules + " AND " + str(att_name) + " = " + str(value_name) 
                else:
                    new_rule = decision_rules + "IF " + str(att_name) + " = " + str(value_name) 
                new_decision_rules += self.__get_decision_rules__(value_list[2], class_label, attribute_names, decision_rules=new_rule, depth=new_depth)
            
            return new_decision_rules
        else: # "Leaf"
            return  decision_rules + (" THEN " + str(class_label) + " = " + str(curr_tree[1]) + "\n")
        
    def __get_prediction__(self, instance, curr_tree):
        info_type = curr_tree[0]
        if info_type == "Attribute":
            instance_attribute_value = instance[self.header.index(curr_tree[1])]
            # Now I need to find which branch to follow recursively
            for kk in range(2, len(curr_tree)):
                value_list = curr_tree[kk]
                if value_list[1] == instance_attribute_value:
                    # We have a match - recurse through the rest of tree
                    return self.__get_prediction__(instance, value_list[2])
        else: # "Leaf"
            return curr_tree[1]

    def __sort_tree__(self, curr_tree):
        info_type = curr_tree[0]
        if info_type == "Attribute":
            # Get all the value names
            value_names = []
            for kk in range(2, len(curr_tree)):
                value_names.append(curr_tree[kk][1])
            # Sort the values
            sorted_value_names = sorted(value_names)
            sorted_value_indices = [value_names.index(sorted_value_name) for sorted_value_name in sorted_value_names]
            # Re-arrange the tree
            new_tree = curr_tree.copy()
            for kk in range(len(sorted_value_indices)):
                new_tree[kk+2] = curr_tree[sorted_value_indices[kk]+2]
            # Sort all of the subtrees 
            for kk in range(len(value_names)):
                new_tree[kk+2][2] = self.__sort_tree__(new_tree[kk+2][2])
            return new_tree
        else: # Leaf node
            return curr_tree

    def __tdidt__(self, current_instances, available_attributes):
        # select an attribute to split on
        split_attribute = self.__select_attribute__(current_instances, available_attributes)
        available_attributes.remove(split_attribute)
        # cannot split on the same attribute twice in a branch
        # recall: python is pass by object reference!!
        tree = ["Attribute", split_attribute]

        # group data by attribute domains (creates pairwise disjoint partitions)
        partitions = self.__partition_instances__(current_instances, split_attribute)

        # for each partition, repeat unless one of the following occurs (base case)
        for attribute_value, partition in partitions.items():
            value_subtree = ["Value", attribute_value]
            # Compute some information of the partition 
            num_values, majority_label = self.__compute_partition_stats__(partition)
            #    CASE 1: all class labels of the partition are the same => make a leaf node
            if len(partition) > 0 and self.__all_same_class__(partition):
                leaf_node = ["Leaf", str(majority_label), num_values, len(current_instances)]
                value_subtree.append(leaf_node)
                tree.append(value_subtree)
            #    CASE 2: no more attributes to select (clash) => handle clash w/majority vote leaf node
            elif len(partition) > 0 and len(available_attributes) == 0:
                leaf_node = ["Leaf", str(majority_label), num_values, len(current_instances)]
                value_subtree.append(leaf_node)
                tree.append(value_subtree)
            #    CASE 3: no more instances to partition (empty partition) => backtrack and replace attribute node with majority vote leaf node
            elif len(partition) == 0:
                num_values, majority_label = self.__compute_partition_stats__(current_instances)
                value_subtree = ["Leaf", str(majority_label), num_values, len(current_instances)]
                tree = value_subtree
            else: # all base cases are false... recurse!!
                value_subtree.append(self.__tdidt__(partition, available_attributes.copy()))
                tree.append(value_subtree)
        
        return tree

    def __select_attribute__(self, instances, available_attributes):
        # We want to use entropy to determinr the attribute to split on

        # Create vector to hold the entropy for each attribute that could be split on
        E_new = []
        # Get the unique class labels
        unique_classes, _ = myutils.get_categorical_frequencies(myutils.get_column(instances, len(instances[0])-1))
        # Go through each attribute and compute its entropy
        for attribute in available_attributes:
            # First group by the unique attribute values
            value_names, value_subtables = myutils.group_by(instances, self.header, attribute)
            unique_values, values_counts = myutils.get_categorical_frequencies(myutils.get_column(instances, self.header.index(attribute)))

            # Compute the priors for each attribute value
            value_priors = [] # 2D list of the priors for each unique value (rows)
            for  kk in range(len(value_names)):
                value_subtable = value_subtables[kk]
                curr_prior = []
                # Get the number of values for each class label to compute the prior
                for class_label in unique_classes:
                    curr_prior.append( myutils.get_class_count(value_subtable, class_label) / len(value_subtable))
                value_priors.append(curr_prior)

            # Compute the entropy for each unique value of the attribute
            entropy_list = [0 for _ in range(len(value_priors))]
            for ii in range(len(value_priors)):
                priors = value_priors[ii]
                for kk in range(len(priors)):
                    if priors[kk] > 0: # Bad if 0 => the class doesn't have at least one instance
                        entropy_list[ii] += -priors[kk] * math.log(priors[kk], 2)

            # Compute the total entropy as the average weighted by the occurance of the value in the table
            # and save it to the E_new list
            attribute_entropy = 0
            for kk in range(len(value_names)):
                value_name = value_names[kk]
                # Get the count of the value in original table
                value_count = values_counts[unique_values.index(value_name)]
                weight = value_count / len(instances)
                attribute_entropy += (weight * entropy_list[kk])
            
            E_new.append(attribute_entropy)
        
        # Now, we have calculated the new entropy for each attribute. 
        # We now want to return the attribute with the lowest entropy
        split_attribute = available_attributes[E_new.index(min(E_new))]
        return split_attribute

    def __partition_instances__(self, instances, split_attribute):
        # this is a group by split_attribute's domain, not by
        # the values of this attribute in instances
        # example: if split_attribute is "level"
        attribute_domain = self.attribute_domains[split_attribute] # ["Senior", "Mid", "Junior"]
        attribute_index = self.header.index(split_attribute) # 0
        # lets build a dictionary
        partitions = {} # key (attribute value): value (list of instances with this attribute value)
        # task: try this!
        for attribute_value in attribute_domain:
            partitions[attribute_value] = []
            for instance in instances:
                if instance[attribute_index] == attribute_value:
                    partitions[attribute_value].append(instance)
        return partitions

    def __all_same_class__(self, partition):
        # This class looks through the partition and checks to see if they all have the same class label
        # We should see that only one of the keys has values, the rest have empty lists
        start_class_label = None
        for value in partition:
            if start_class_label == None:
                # First loop, set to be the initial class label
                start_class_label = value[-1]
            
            if value[-1] != start_class_label:
                return False

        # If the code makes it to this point, all the labels have the same value
        return True

    def __compute_partition_stats__(self, partition):
        # Find the number of values, labels and counts
        num_values = len(partition)
        
        labels, label_counts = [], []
        for value in partition:
            if value[-1] not in labels:
                labels.append(value[-1])
                label_counts.append(0)
            label_counts[labels.index(value[-1])] += 1

        majority_label = None
        if num_values > 0:
            majority_label = labels[label_counts.index(max(label_counts))]

        return num_values, majority_label    

    def __get_attribute_domain__(self, X_train, header):
        att_domains = {}
        # Cycle through each attribute column and get the unique values
        for col_index in range(len(X_train[0])):
            # Get the column
            column = myutils.get_column(X_train, col_index)
            # Get unique values
            unique_vals = list(set(column))
            # Add to the dictionary
            att_domains[header[col_index]] = unique_vals

        return att_domains

class MyRandomForestClassifier:
    """Represents a random forest classifier.

    Attributes:
        X_train(list of list of obj): The list of training instances (samples). 
                The shape of X_train is (n_train_samples, n_features)
        y_train(list of obj): The target y values (parallel to X_train). 
            The shape of y_train is n_samples
        tree(nested list): The extracted tree model.

    Notes:
        Loosely based on sklearn's DecisionTreeClassifier: https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
        Terminology: instance = sample = row and attribute = feature = column
    """
    def __init__(self):
        """Initializer for MyDecisionTreeClassifier.

        """
        self.X_train = None 
        self.y_train = None
        self.trees = None
        self.header = None
        self.attribute_domain = None

    def fit(self, X, y, N, M, F):
        """Fits a random forest ensemble classifier to X_train and y_train using decision trees 
            created with the TDIDT (top down induction of decision tree) algorithm.

        Args:
            X(list of list of obj): The list of instances (samples). 
                The shape of Xis (n_samples, n_features)
            y(list of obj): The target y values (parallel to X)
                The shape of y is n_samples
            N(int): the number of decision trees to create
            M(int): the number of "best" trees to keep. Note M < N
            F(int): the number of random attributes to use for each classifier

        Notes:
            Since TDIDT is an eager learning algorithm, this method builds a decision tree model
                from the training data.
            Build a decision tree using the nested list representation described in class.
            Store the tree in the tree attribute.
            Use attribute indexes to construct default attribute names (e.g. "att0", "att1", ...).
        """
        # First, make sure everything is represented as a string
        for row in range(len(X)):
            y[row] = str(y[row])
            for col in range(len(X[0])):
                X[row][col] = str(X[row][col])

        # Generate a random stratified test set consisting of one third of the original data set, 
        # with the remaining two thirds of the instances forming the "remainder set".
        X_test, y_test, X_remainder, y_remainder = myevaluation.random_stratified_train_test_split(X, y, test_size=0.33)

        # PROBABLY WANT TO MOVE THE CODE ABOVE OUTSIDE OF THIS FUNCTION...

        # Zip together the X and y values for the test and remainder sets
        test_dataset = [X_test[i] + [y_test[i]] for i in range(len(X_test))]
        remainder_dataset = [X_remainder[i] + [y_remainder[i]] for i in range(len(X_remainder))]
        # Create a generic header
        self.header = ["att"+str(kk) for kk in range(len(X[0]))]
        # Get the attribute domain dictionary 
        self.attribute_domains = self.__get_attribute_domain__(X, self.header)

        # Generate the N Trees using a bootstrapped sample of the remainder set 
        all_trees, all_trees_accuracies = [], [] # Create a list of all the trees created and aa parallel list of their accuracies
        for _ in range(N):
            # Get the training and validation sets from the remainder set using bootstrapping
            train_set, validation_set = self.__get_bootstrapped_train_validation_sets__(remainder_dataset)
            # Generate the tree
            available_attributes = self.header.copy()
            tree = self.__tdidt__(train_set, available_attributes)
            tree = self.__sort_tree__(tree)
            # Add the tree to the list
            all_trees.append(tree)
            # Get the accuracy of the tree
            tree_accuracy = self.__get_tree_accuracy__(validation_set)
            all_trees_accuracies.append(tree_accuracy)

            # TODO: Implement the random sampling of the attributes (F)!!!!!

        # Now that the trees have been generated, go through and pick the M most accurate trees
        sorted_accuracies = sorted(all_trees_accuracies.copy())
        accuracy_cutoff = sorted_accuracies[M]
        count = 0
        self.trees = []
        for index in range(N):
            if all_trees_accuracies[index] <= accuracy_cutoff:
                self.trees.append(all_trees[index])
                count += 1
            if count > M:
                break

    # TODO: Implement this method!!!
    def predict(self, X_test):
        return []

    def __get_tree_accuracy__(self, validation_dataset):
        return 0

    def __get_bootstrapped_train_validation_sets__(self, table):
        n = len(table)
        train_sample_indices, validation_sample_indices = [], []
        # Get the indices of the training sample (~63% of the dataset)
        for _ in range(n):
            rand_index = random.randrange(0, n)
            train_sample_indices.append(rand_index)
        # Go through the indices of the dataset and the training indices and get the remainder as the validation indices
        train_set, validation_set = [], []
        for index in range(n):
            if index not in train_sample_indices:
                validation_sample_indices.append(index)
        # Now genertate the two sets
        train_set, validation_set = [], []
        for index in train_sample_indices:
            train_set.append(table[index])
        for index in validation_sample_indices:
            validation_set.append(table[index])

        return train_set, validation_set

    def __compute_random_attribute_subset__(self, values, F_value):
        shuffled = values[:] # shallow copy
        random.shuffle(shuffled)
        return shuffled[:F_value]

    def __get_tree_prediction__(self, instance, curr_tree):
        info_type = curr_tree[0]
        if info_type == "Attribute":
            instance_attribute_value = instance[self.header.index(curr_tree[1])]
            # Now I need to find which branch to follow recursively
            for kk in range(2, len(curr_tree)):
                value_list = curr_tree[kk]
                if value_list[1] == instance_attribute_value:
                    # We have a match - recurse through the rest of tree
                    return self.__get_tree_prediction__(instance, value_list[2])
        else: # "Leaf"
            return curr_tree[1]

    def __sort_tree__(self, curr_tree):
        info_type = curr_tree[0]
        if info_type == "Attribute":
            # Get all the value names
            value_names = []
            for kk in range(2, len(curr_tree)):
                value_names.append(curr_tree[kk][1])
            # Sort the values
            sorted_value_names = sorted(value_names)
            sorted_value_indices = [value_names.index(sorted_value_name) for sorted_value_name in sorted_value_names]
            # Re-arrange the tree
            new_tree = curr_tree.copy()
            for kk in range(len(sorted_value_indices)):
                new_tree[kk+2] = curr_tree[sorted_value_indices[kk]+2]
            # Sort all of the subtrees 
            for kk in range(len(value_names)):
                new_tree[kk+2][2] = self.__sort_tree__(new_tree[kk+2][2])
            return new_tree
        else: # Leaf node
            return curr_tree

    def __tdidt__(self, current_instances, available_attributes):
        # select an attribute to split on
        split_attribute = self.__select_attribute__(current_instances, available_attributes)
        available_attributes.remove(split_attribute)
        # cannot split on the same attribute twice in a branch
        # recall: python is pass by object reference!!
        tree = ["Attribute", split_attribute]

        # group data by attribute domains (creates pairwise disjoint partitions)
        partitions = self.__partition_instances__(current_instances, split_attribute)

        # for each partition, repeat unless one of the following occurs (base case)
        for attribute_value, partition in partitions.items():
            value_subtree = ["Value", attribute_value]
            # Compute some information of the partition 
            num_values, majority_label = self.__compute_partition_stats__(partition)
            #    CASE 1: all class labels of the partition are the same => make a leaf node
            if len(partition) > 0 and self.__all_same_class__(partition):
                leaf_node = ["Leaf", str(majority_label), num_values, len(current_instances)]
                value_subtree.append(leaf_node)
                tree.append(value_subtree)
            #    CASE 2: no more attributes to select (clash) => handle clash w/majority vote leaf node
            elif len(partition) > 0 and len(available_attributes) == 0:
                leaf_node = ["Leaf", str(majority_label), num_values, len(current_instances)]
                value_subtree.append(leaf_node)
                tree.append(value_subtree)
            #    CASE 3: no more instances to partition (empty partition) => backtrack and replace attribute node with majority vote leaf node
            elif len(partition) == 0:
                num_values, majority_label = self.__compute_partition_stats__(current_instances)
                value_subtree = ["Leaf", str(majority_label), num_values, len(current_instances)]
                tree = value_subtree
            else: # all base cases are false... recurse!!
                value_subtree.append(self.__tdidt__(partition, available_attributes.copy()))
                tree.append(value_subtree)
        
        return tree

    def __select_attribute__(self, instances, available_attributes):
        # We want to use entropy to determinr the attribute to split on

        # Create vector to hold the entropy for each attribute that could be split on
        E_new = []
        # Get the unique class labels
        unique_classes, _ = myutils.get_categorical_frequencies(myutils.get_column(instances, len(instances[0])-1))
        # Go through each attribute and compute its entropy
        for attribute in available_attributes:
            # First group by the unique attribute values
            value_names, value_subtables = myutils.group_by(instances, self.header, attribute)
            unique_values, values_counts = myutils.get_categorical_frequencies(myutils.get_column(instances, self.header.index(attribute)))

            # Compute the priors for each attribute value
            value_priors = [] # 2D list of the priors for each unique value (rows)
            for  kk in range(len(value_names)):
                value_subtable = value_subtables[kk]
                curr_prior = []
                # Get the number of values for each class label to compute the prior
                for class_label in unique_classes:
                    curr_prior.append( myutils.get_class_count(value_subtable, class_label) / len(value_subtable))
                value_priors.append(curr_prior)

            # Compute the entropy for each unique value of the attribute
            entropy_list = [0 for _ in range(len(value_priors))]
            for ii in range(len(value_priors)):
                priors = value_priors[ii]
                for kk in range(len(priors)):
                    if priors[kk] > 0: # Bad if 0 => the class doesn't have at least one instance
                        entropy_list[ii] += -priors[kk] * math.log(priors[kk], 2)

            # Compute the total entropy as the average weighted by the occurance of the value in the table
            # and save it to the E_new list
            attribute_entropy = 0
            for kk in range(len(value_names)):
                value_name = value_names[kk]
                # Get the count of the value in original table
                value_count = values_counts[unique_values.index(value_name)]
                weight = value_count / len(instances)
                attribute_entropy += (weight * entropy_list[kk])
            
            E_new.append(attribute_entropy)
        
        # Now, we have calculated the new entropy for each attribute. 
        # We now want to return the attribute with the lowest entropy
        split_attribute = available_attributes[E_new.index(min(E_new))]
        return split_attribute

    def __partition_instances__(self, instances, split_attribute):
        # this is a group by split_attribute's domain, not by
        # the values of this attribute in instances
        # example: if split_attribute is "level"
        attribute_domain = self.attribute_domains[split_attribute] # ["Senior", "Mid", "Junior"]
        attribute_index = self.header.index(split_attribute) # 0
        # lets build a dictionary
        partitions = {} # key (attribute value): value (list of instances with this attribute value)
        # task: try this!
        for attribute_value in attribute_domain:
            partitions[attribute_value] = []
            for instance in instances:
                if instance[attribute_index] == attribute_value:
                    partitions[attribute_value].append(instance)
        return partitions

    def __all_same_class__(self, partition):
        # This class looks through the partition and checks to see if they all have the same class label
        # We should see that only one of the keys has values, the rest have empty lists
        start_class_label = None
        for value in partition:
            if start_class_label == None:
                # First loop, set to be the initial class label
                start_class_label = value[-1]
            
            if value[-1] != start_class_label:
                return False

        # If the code makes it to this point, all the labels have the same value
        return True

    def __compute_partition_stats__(self, partition):
        # Find the number of values, labels and counts
        num_values = len(partition)
        
        labels, label_counts = [], []
        for value in partition:
            if value[-1] not in labels:
                labels.append(value[-1])
                label_counts.append(0)
            label_counts[labels.index(value[-1])] += 1

        majority_label = None
        if num_values > 0:
            majority_label = labels[label_counts.index(max(label_counts))]

        return num_values, majority_label    

    def __get_attribute_domain__(self, X_train, header):
        att_domains = {}
        # Cycle through each attribute column and get the unique values
        for col_index in range(len(X_train[0])):
            # Get the column
            column = myutils.get_column(X_train, col_index)
            # Get unique values
            unique_vals = list(set(column))
            # Add to the dictionary
            att_domains[header[col_index]] = unique_vals

        return att_domains