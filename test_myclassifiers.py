import numpy as np
import scipy.stats as stats 

from mysklearn.myclassifiers import MySimpleLinearRegressor, MyKNeighborsClassifier, MyNaiveBayesClassifier, MyDecisionTreeClassifier, MyRandomForestClassifier
from mysklearn.mypytable import MyPyTable
import mysklearn.myutils as myutils

# note: order is actual/received student value, expected/solution
################################################################
# Datasets for testing Naive Bayes Classifier
################################################################

# Dataset from Bramer 3.2 Train dataset
train_col_names = ["day", "season", "wind", "rain", "class"]
train_table = [ 
    ["weekday", "spring", "none", "none", "on time"],
    ["weekday", "winter", "none", "slight", "on time"],
    ["weekday", "winter", "none", "slight", "on time"],
    ["weekday", "winter", "high", "heavy", "late"], 
    ["saturday", "summer", "normal", "none", "on time"],
    ["weekday", "autumn", "normal", "none", "very late"],
    ["holiday", "summer", "high", "slight", "on time"],
    ["sunday", "summer", "normal", "none", "on time"],
    ["weekday", "winter", "high", "heavy", "very late"],
    ["weekday", "summer", "none", "slight", "on time"],
    ["saturday", "spring", "high", "heavy", "cancelled"],
    ["weekday", "summer", "high", "slight", "on time"],
    ["saturday", "winter", "normal", "none", "late"],
    ["weekday", "summer", "high", "none", "on time"],
    ["weekday", "winter", "normal", "heavy", "very late"],
    ["saturday", "autumn", "high", "slight", "on time"],
    ["weekday", "autumn", "none", "heavy", "on time"],
    ["holiday", "spring", "normal", "slight", "on time"],
    ["weekday", "spring", "normal", "none", "on time"],
    ["weekday", "spring", "normal", "slight", "on time"]
]
# RQ5 Example for Naive Bayes
iphone_col_names = ["standing", "job_status", "credit_rating", "buys_iphone"]
iphone_table = [ 
    [1, 3, "fair", "no"], 
    [1, 3, "excellent", "no"], 
    [2, 3, "fair", "yes"],
    [2, 2, "fair", "yes"], 
    [2, 1, "fair", "yes"], 
    [2, 1, "excellent", "no"], 
    [2, 1, "excellent", "yes"], 
    [1, 2, "fair", "no"], 
    [1, 1, "fair", "yes"], 
    [2, 2, "fair", "yes"], 
    [1, 2, "excellent", "yes"], 
    [2, 2, "excellent", "yes"], 
    [2, 3, "fair", "yes"], 
    [2, 2, "excellent", "no"], 
    [2, 3, "fair", "yes"] ]

# Example traced out in class for Naive Bayes
attr_table = [ [1, 5, "yes"], [2, 6, "yes"], [1, 5, "no"], [1, 5, "no"], 
    [1, 6, "yes"], [2, 6, "no"], [1, 5, "yes"], [1, 6, "yes"] ]
attr_col_names = ["attr_1", "attr_2", "result"]

interview_header = ["level", "lang", "tweets", "phd", "interviewed_well"]
interview_table = [
        ["Senior", "Java", "no", "no", "False"],
        ["Senior", "Java", "no", "yes", "False"],
        ["Mid", "Python", "no", "no", "True"],
        ["Junior", "Python", "no", "no", "True"],
        ["Junior", "R", "yes", "no", "True"],
        ["Junior", "R", "yes", "yes", "False"],
        ["Mid", "R", "yes", "yes", "True"],
        ["Senior", "Python", "no", "no", "False"],
        ["Senior", "R", "yes", "no", "True"],
        ["Junior", "Python", "yes", "no", "True"],
        ["Senior", "Python", "yes", "yes", "True"],
        ["Mid", "Python", "no", "yes", "True"],
        ["Mid", "Java", "yes", "no", "True"],
        ["Junior", "Python", "no", "yes", "False"]
]

# note: this tree uses the generic "att#" attribute labels because fit() does not and should not accept attribute names
# note: the attribute values are sorted alphabetically
interview_tree = ["Attribute", "att0",
            ["Value", "Junior", 
                ["Attribute", "att3",
                    ["Value", "no", 
                        ["Leaf", "True", 3, 5]],
                    ["Value", "yes", 
                        ["Leaf", "False", 2, 5]]]],
            ["Value", "Mid",
                ["Leaf", "True", 4, 14]],
            ["Value", "Senior",
                ["Attribute", "att2",
                    ["Value", "no",
                        ["Leaf", "False", 3, 5]],
                    ["Value", "yes",
                        ["Leaf", "True", 2, 5]]]]]

# bramer degrees dataset
degrees_table = [
        ["A", "B", "A", "B", "B", "SECOND"],
        ["A", "B", "B", "B", "A", "FIRST"],
        ["A", "A", "A", "B", "B", "SECOND"],
        ["B", "A", "A", "B", "B", "SECOND"],
        ["A", "A", "B", "B", "A", "FIRST"],
        ["B", "A", "A", "B", "B", "SECOND"],
        ["A", "B", "B", "B", "B", "SECOND"],
        ["A", "B", "B", "B", "B", "SECOND"],
        ["A", "A", "A", "A", "A", "FIRST"],
        ["B", "A", "A", "B", "B", "SECOND"],
        ["B", "A", "A", "B", "B", "SECOND"],
        ["A", "B", "B", "A", "B", "SECOND"],
        ["B", "B", "B", "B", "A", "SECOND"],
        ["A", "A", "B", "A", "B", "FIRST"],
        ["B", "B", "B", "B", "A", "SECOND"],
        ["A", "A", "B", "B", "B", "SECOND"],
        ["B", "B", "B", "B", "B", "SECOND"],
        ["A", "A", "B", "A", "A", "FIRST"],
        ["B", "B", "B", "A", "A", "SECOND"],
        ["B", "B", "A", "A", "B", "SECOND"],
        ["B", "B", "B", "B", "A", "SECOND"],
        ["B", "A", "B", "A", "B", "SECOND"],
        ["A", "B", "B", "B", "A", "FIRST"],
        ["A", "B", "A", "B", "B", "SECOND"],
        ["B", "A", "B", "B", "B", "SECOND"],
        ["A", "B", "B", "B", "B", "SECOND"],
    ]
degrees_header = ["SoftEng", "ARIN", "HCI", "CSA", "Project", "Class"]
degrees_tree = ['Attribute', 'att0', 
    ['Value', 'A', 
        ['Attribute', 'att4', 
            ['Value', 'A', 
                ['Leaf', 'FIRST', 5, 14]
            ], 
            ['Value', 'B', 
                ['Attribute', 'att3', 
                    ['Value', 'A', 
                        ['Attribute', 'att1', 
                            ['Value', 'A', 
                                ['Leaf', 'FIRST', 1, 2]
                            ], 
                            ['Value', 'B', 
                                ['Leaf', 'SECOND', 1, 2]
                            ]
                        ]
                    ], 
                    ['Value', 'B', 
                        ['Leaf', 'SECOND', 7, 9]
                    ]
                ]
            ]
        ]
    ], 
    ['Value', 'B', 
        ['Leaf', 'SECOND', 12, 26]
    ]
]

################################################################
# Methods to run the tests
################################################################

# note: order is actual/received student value, expected/solution
def test_simple_linear_regressor_fit():
    np.random.seed(0)
    myLinRegress = MySimpleLinearRegressor()
    x_orig = list(range(100))
    x = [ [x_orig[kk]] for kk in range(len(x_orig))]

    # Test Case 1:
    y = [value[0] * 2 + np.random.normal(0, 25) for value in x]
    myLinRegress.fit(X_train=x, y_train=y)
    sp_m, sp_b, _, _, _ = stats.linregress(x_orig, y)
    assert np.isclose(myLinRegress.slope, sp_m)
    assert np.isclose(myLinRegress.intercept, sp_b)

    # Test Case 2:
    y = [value[0] * (10 + np.random.normal(0, 2)) + 1 for value in x]
    myLinRegress.fit(X_train=x, y_train=y)
    sp_m, sp_b, _, _, _ = stats.linregress(x_orig, y)
    assert np.isclose(myLinRegress.slope, sp_m)
    assert np.isclose(myLinRegress.intercept, sp_b)

def test_simple_linear_regressor_predict():
    np.random.seed(0)

    # Test Case 1: random x values
    myLinRegress = MySimpleLinearRegressor(slope=5, intercept=0)
    x_test_orig = 100 * np.random.random_sample((100,)) # Generate test points
    x_test = [ [x_test_orig[kk]] for kk in range(len(x_test_orig))]
    y_pred = myLinRegress.predict(x_test)
    y_actual = [5 * x + 0 for x in x_test_orig]
    assert np.allclose(y_pred, y_actual)

    # Test Case 2: 
    myLinRegress = MySimpleLinearRegressor(slope=np.random.rand(), intercept=np.random.rand())
    x_test_orig = 100 * np.random.random_sample((100,)) # Generate test points
    x_test = [ [x_test_orig[kk]] for kk in range(len(x_test_orig))]
    y_pred = myLinRegress.predict(x_test)
    y_actual = [myLinRegress.slope * x + myLinRegress.intercept for x in x_test_orig]
    assert np.allclose(y_pred, y_actual)

def test_kneighbors_classifier_kneighbors():
    """
    1. Use the 4 instance training set example traced in class on the iPad, asserting against our desk check
    2. Use the 8 instance training set example from ClassificationFun/main.py, asserting against our in-class result
    3. Use Bramer 3.6 Self-assessment exercise 2, asserting against exercise solution in Bramer Appendix E
    """
    myKNeigh = MyKNeighborsClassifier()

    # Test Case 1: 
    x = [ [7,7], [7,4], [3,4], [1,4] ]
    y = ["Bad", "Bad", "Good", "Good"]
    x_test = [[3,7]]
    # For this dataset, we normalized the columns
    norm_x_train, norm_x_test = myutils.normalize_train_and_test_sets(x, x_test)  
    myKNeigh.fit(norm_x_train, y)

    pred_closest_dist, pred_closest_indices = myKNeigh.kneighbors(norm_x_test)
    assert np.allclose(pred_closest_indices, [[0, 2, 3]])
    assert np.allclose(pred_closest_dist, [[(2/3), 1.00, np.sqrt(1 + (1/3)**2)]])

    # Test Case 2:
    x = [ [3, 2], [6, 6], [4, 1], [4, 4], [1, 2], [2, 0], [0, 3], [1, 6] ]
    y = ["no","yes","no","no","yes","no","yes","yes"]
    myKNeigh.fit(x, y)
    x_test = [[2,3]]
    pred_closest_dist, pred_closest_indices = myKNeigh.kneighbors(x_test)
    # Actual: [[3, 2, 'no', 0, 1.4142135623730951], [1, 2, 'yes', 4, 1.4142135623730951], [0, 3, 'yes', 6, 2.0]]
    assert np.allclose(pred_closest_indices, [[0, 4, 6]])
    assert np.allclose(pred_closest_dist, [[1.4142135623730951, 1.4142135623730951, 2.0]])

    # Test Case 3: 
    myKNeigh = MyKNeighborsClassifier(n_neighbors=5)
    x = [[0.8, 6.3], [1.4, 8.1], [2.1, 7.4], [2.6, 14.3], [6.8, 12.6], [8.8, 9.8], 
        [9.2, 11.6], [10.8, 9.6], [11.8, 9.9], [12.4, 6.5], [12.8, 1.1], [14, 19.9], 
        [14.2, 18.5], [15.6, 17.4], [15.8, 12.2], [16.6, 6.7], [17.4, 4.5], [18.2, 6.9], 
        [19, 3.4], [19.6, 11.1]]
    y = ["-","-","-","+","-","+","-","+","+","+","-","-","-","-","-","+","+","+","-","+"]
    myKNeigh.fit(x, y)
    x_test = [[9.1, 11]]
    pred_closest_dist, pred_closest_indices = myKNeigh.kneighbors(x_test)
    assert np.allclose(pred_closest_indices, [[6, 5, 7, 4, 8]])
    assert np.allclose(pred_closest_dist, [[0.608, 1.237, 2.202, 2.802, 2.915]], atol=0.001) # Use the tolerance of the known vals

def test_kneighbors_classifier_predict():
    """
    1. Use the 4 instance training set example traced in class on the iPad, asserting against our desk check
    2. Use the 8 instance training set example from ClassificationFun/main.py, asserting against our in-class result
    3. Use Bramer 3.6 Self-assessment exercise 2, asserting against exercise solution in Bramer Appendix E
    """
    myKNeigh = MyKNeighborsClassifier()

     # Test Case 1: 
    x = [ [7, 7], [7,4], [3, 4], [1, 4] ]
    y = ["Bad", "Bad", "Good", "Good"]
    myKNeigh.fit(x, y)
    x_test = [[3,7]]
    pred_y = myKNeigh.predict(x_test)
    assert pred_y == ["Good"]

    # Test Case 2:
    x = [ [3, 2], [6, 6], [4, 1], [4, 4], [1, 2], [2, 0], [0, 3], [1, 6] ]
    y = ["no","yes","no","no","yes","no","yes","yes"]
    myKNeigh.fit(x, y)
    x_test = [[2,3]]
    pred_y = myKNeigh.predict(x_test)
    assert pred_y == ["yes"]

    # Test Case 3
    myKNeigh = MyKNeighborsClassifier(n_neighbors=5)
    x = [[0.8, 6.3], [1.4, 8.1], [2.1, 7.4], [2.6, 14.3], [6.8, 12.6], [8.8, 9.8], 
        [9.2, 11.6], [10.8, 9.6], [11.8, 9.9], [12.4, 6.5], [12.8, 1.1], [14, 19.9], 
        [14.2, 18.5], [15.6, 17.4], [15.8, 12.2], [16.6, 6.7], [17.4, 4.5], [18.2, 6.9], 
        [19, 3.4], [19.6, 11.1]]
    y = ["-","-","-","+","-","+","-","+","+","+","-","-","-","-","-","+","+","+","-","+"]
    myKNeigh.fit(x, y)
    x_test = [[9.1, 11]]
    pred_y = myKNeigh.predict(x_test)
    assert pred_y == ["+"]

def test_naive_bayes_classifier_fit():
    myNaiveBayes = MyNaiveBayesClassifier()

    # Test Case 1: Example traced out in class
    y_train, X_train = [], []
    for inst in attr_table:
        y_train.append(inst[-1])
        X_train.append(inst[:-1])
    myNaiveBayes.fit(X_train, y_train)
    # Assert against the priors and posteriors
    true_priors = [[(5/8)], [(3/8)]] # buys_iphone=yes, buys_iphone=no
    true_posteriors = [ [(4/5),(2/3)] , [(1/5),(1/3)] , [(2/5),(2/3)] , [(3/5),(1/3)] ]
    _, pred_priors = myNaiveBayes.priors.get_subtable(0, 'end', 1, 'end')
    _, pred_posteriors = myNaiveBayes.posteriors.get_subtable(0, 'end', 2, 'end')

    assert np.allclose(true_priors, pred_priors)
    assert np.allclose(true_posteriors, pred_posteriors) 

    # Test Case 2: RQ5 Example
    y_train, X_train = [], []
    for inst in iphone_table:
        y_train.append(inst[-1])
        X_train.append(inst[:-1])
    myNaiveBayes.fit(X_train, y_train)
    # Assert against the priors and posteriors
    true_priors = [[(5/15)], [(10/15)]] # buys_iphone=yes, buys_iphone=no
    true_posteriors = [[0.6, 0.2], [0.4, 0.8], [0.4, 0.3], [0.4, 0.4], [0.2, 0.3], [0.4, 0.7], [0.6, 0.3]]
    _, pred_priors = myNaiveBayes.priors.get_subtable(0, 'end', 1, 'end')
    _, pred_posteriors = myNaiveBayes.posteriors.get_subtable(0, 'end', 2, 'end')

    assert np.allclose(true_priors, pred_priors)
    assert np.allclose(true_posteriors, pred_posteriors) 

    # Test Case 3: Bramer 3.2 Train Dataset
    y_train, X_train = [], []
    for inst in train_table:
        y_train.append(inst[-1])
        X_train.append(inst[:-1])
    myNaiveBayes.fit(X_train, y_train)
    # Assert against the priors and posteriors
    true_priors = [[(14/20)],[(2/20)],[(3/20)],[(1/20)]] # buys_iphone=yes, buys_iphone=no
    true_posteriors = [[0.6428571428571429, 0.5, 1.0, 0.0], [0.14285714285714285, 0.5, 0.0, 1.0],
        [0.14285714285714285, 0.0, 0.0, 0.0], [0.07142857142857142, 0.0, 0.0, 0.0],
        [0.2857142857142857, 0.0, 0.0, 1.0], [0.14285714285714285, 1.0, 0.6666666666666666, 0.0],
        [0.42857142857142855, 0.0, 0.0, 0.0], [0.14285714285714285, 0.0, 0.3333333333333333, 0.0],
        [0.35714285714285715, 0.0, 0.0, 0.0], [0.2857142857142857, 0.5, 0.3333333333333333, 1.0],
        [0.35714285714285715, 0.5, 0.6666666666666666, 0.0], [0.35714285714285715, 0.5, 0.3333333333333333, 0.0],
        [0.5714285714285714, 0.0, 0.0, 0.0], [0.07142857142857142, 0.5, 0.6666666666666666, 1.0]]
    _, pred_priors = myNaiveBayes.priors.get_subtable(0, 'end', 1, 'end')
    _, pred_posteriors = myNaiveBayes.posteriors.get_subtable(0, 'end', 2, 'end')

    assert np.allclose(true_priors, pred_priors)
    assert np.allclose(true_posteriors, pred_posteriors) 

def test_naive_bayes_classifier_predict():
    myNaiveBayes = MyNaiveBayesClassifier()

    # Test Case 1: Example traced out in class
    y_train, X_train = [], []
    for inst in attr_table:
        y_train.append(inst[-1])
        X_train.append(inst[:-1])
    myNaiveBayes.fit(X_train, y_train)
    # Get the prediction for the given test value(s)
    X_test = [ [1,5] ]
    y_pred = myNaiveBayes.predict(X_test)
    assert y_pred ==[ "yes" ]

    # Test Case 2: RQ5 Example
    y_train, X_train = [], []
    for inst in iphone_table:
        y_train.append(inst[-1])
        X_train.append(inst[:-1])
    myNaiveBayes.fit(X_train, y_train)
    # Get the prediction for the given test value(s)
    X_test = [ [2, 2, "fair"], [1, 1, "excellent"] ]
    y_pred = myNaiveBayes.predict(X_test)
    assert y_pred == [ "yes", "no" ]

    # Test Case 3: Bramer 3.2 Train Dataset
    y_train, X_train = [], []
    for inst in train_table:
        y_train.append(inst[-1])
        X_train.append(inst[:-1])
    myNaiveBayes.fit(X_train, y_train)
    # Get the prediction for the given test value(s)
    X_test = [ ["weekday", "winter", "high", "heavy"], 
        ["weekday", "summer", "high", "heavy"], 
        ["sunday", "summer", "normal", "slight"] ]
    y_pred = myNaiveBayes.predict(X_test)
    assert y_pred == [ "very late", "on time", "on time" ]

def test_decision_tree_classifier_fit():
    # Test the fit for the Interview Dataset
    myDecisionTreeClassifier = MyDecisionTreeClassifier()
    y_train, X_train = [], []
    for inst in interview_table:
        y_train.append(inst[-1])
        X_train.append(inst[:-1])
    myDecisionTreeClassifier.fit(X_train, y_train)

    assert interview_tree == myDecisionTreeClassifier.tree

    # Test the result from Bramer
    y_train, X_train = [], []
    for inst in degrees_table:
        y_train.append(inst[-1])
        X_train.append(inst[:-1])
    myDecisionTreeClassifier.fit(X_train, y_train)

    assert degrees_tree == myDecisionTreeClassifier.tree

def test_decision_tree_classifier_predict():
    # Test the predict for the Interview Dataset
    myDecisionTreeClassifier = MyDecisionTreeClassifier()
    y_train, X_train = [], []
    for inst in interview_table:
        y_train.append(inst[-1])
        X_train.append(inst[:-1])
    myDecisionTreeClassifier.fit(X_train, y_train)

    X_test = [ ["Junior", "Java", "yes", "no"], ["Junior", "Java", "yes", "yes"]]
    y_pred = myDecisionTreeClassifier.predict(X_test)

    assert y_pred == ["True", "False"]

    # Test the result from Bramer
    y_train, X_train = [], []
    for inst in degrees_table:
        y_train.append(inst[-1])
        X_train.append(inst[:-1])
    myDecisionTreeClassifier.fit(X_train, y_train)

    X_test = [["B", "B", "B", "B", "B"], ["A", "A", "A", "A", "A"], ["A", "A", "A", "A", "B"]]
    y_pred = myDecisionTreeClassifier.predict(X_test)

    assert y_pred == ["SECOND", "FIRST", "FIRST"]

def test_My_Random_Forest_Classifier_fit():
    # Object Declarations
    # Tests with N = 3, M = 2, F = 2 and seed = 0
    rand_forest_test = MyRandomForestClassifier(3, 2, 2, 0)
    table = MyPyTable()
    
    # Variable Assignment and Declaration
    table.data = interview_table
    table.column_names = interview_header
    
    X_test = interview_table
    y_train = table.get_column("interviewed_well")
    
    # Tests on the Interview Dataset
    rand_forest_test.header = interview_header
    rand_forest_test.fit(X_test, y_train)

    trees = rand_forest_test.trees

    for tree in trees:
        print(tree)
        print("\n")

    assert True == False
    
def test_My_Random_Forest_Classifier_predict():
    # Object Declarations
    # Tests with N = 3, M = 2, F = 2 and seed = 1
    rand_forest_test = MyRandomForestClassifier(3, 2, 2, 1)
    table = MyPyTable()
    
    # Variable Assignment and Declaration
    table.data = interview_table
    table.column_names = interview_header
    
    y_train, X_train = [], []
    for inst in interview_table:
        y_train.append(inst[-1])
        X_train.append(inst[:-1])
    
    # Sets X_test
    X_test = [ ["Junior", "Java", "yes", "no"], ["Junior", "Java", "yes", "yes"]]
    
    # Tests on the Interview Dataset
    rand_forest_test.header = interview_header[:-1]
    rand_forest_test.fit(X_train, y_train)
    y_predicted = rand_forest_test.predict(X_test)

    print("y_predicted:", y_predicted)

    # Trace Test
    
    
    assert y_predicted == ['True', 'False']