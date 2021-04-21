"""
Programmer: Matthew Triebes
Class: 322-01, Spring 2021
Programming Assignment #3
2/21/21

This program is a utils file that contains the functions needed for VGSales.ipynb 
"""

# Imports 
import numpy as np
import matplotlib.pyplot as plt
from mysklearn.mypytable import MyPyTable

# ------------------------
# Creates A Bar Chart
# ------------------------
def create_bar_chart(x, y, x_labels, axis_labels, angle=90):
    """Creates a Bar Chart based on two passed in lists
    Args:
        x(list of ints): x axis values
        y(list of ints): y axis bar values
        x_labels(list of strs): x axis labels
        axis_labels(list of strs): labels for the graph (title, x label, y label)
    Returns:
        Print out of a bar chart
    """
    plt.figure()
    plt.bar(x, y)
    # Adds string values to x ticks
    plt.xticks(x, x_labels, rotation=angle, horizontalalignment="center")
    # Makes the xticks readable
    plt.tick_params(axis='x', which='major', labelsize=8)
    # Sets the graph labels
    plt.title(axis_labels[0])
    plt.xlabel(axis_labels[1])
    plt.ylabel(axis_labels[2])
    # Shows the final plot
    plt.show()

# ------------------------
# Counts the number of occurences in a Table
# ------------------------
def get_counts(table, col, col_num):
    """Counts the number of occurences of a specific value in a table
    Args:
        table(2D List): All contents in file
        col(list of values): values to be searched for
    Returns:
        Print out of a bar chart
    """
    count = 0
    counts = []
    
    # Gets a count of each value per element of col 
    for i in col:
        for j in range(len(table)):
            if(table[j][col_num] == i):
                count = count + 1
        counts.append(count)
        count = 0
    
    return counts

# ------------------------
# Removes the duplicates in a list of values
# ------------------------
def remove_duplicates(col):
    """Casts the list to a set and then casts back to a list to remove duplicates
    Args:
        col(list of values): values to be searched for
    Returns:
        col w/o the duplicate values
    """
    col = set(col) # Removes duplicate values
    col = list(col)
    
    return col

# ------------------------
# Creates a Pie chart
# ------------------------
def create_pie_chart(x, y, title):
    """ Creates A Pie chart with count values
    Args:
        x(list): Labels for each part of the pie chart
        y(list): Values that create the sections of the pie chart
    Returns:
        Print out of a pie chart
    """
    plt.figure()
    # Adds Title to Chart
    plt.title(title)
    # Plots the Pie Chart
    plt.pie(y, labels=x, autopct="%1.1f%%")
    # Displays the Pie Chart
    plt.show()

# ------------------------
# Adds together elems in a list
# ------------------------
def sum_list_values(arr):
    """ Sums elements in a list 
    Args:
        arr(list or ints or floats): List of values that are going to be added together
    Returns:
        summ(float): A sum of all the values in a list
    """
    summ = 0.0
    for i in range(len(arr)):
        summ += arr[i]
    return round(summ, 2)

# ------------------------
# Computes Bin Frequencies
# ------------------------
def compute_bin_frequencies(values, cutoffs):
    """ Computes the frequency of occurences per-cutoff
    Args:
        values(list): list of values to get frequencies for
        cutoffs(list): cutoff point for each value
    Returns:
        freqs(list of ints): A list of all the frequences per cutoff
    """
    freqs = [0 for _ in range(len(cutoffs) - 1)]

    for val in values:
        if val == max(values):
            freqs[-1] += 1
        else:
            for i in range(len(cutoffs) - 1):
                if cutoffs[i] <= val < cutoffs[i + 1]:
                    freqs[i] += 1
    return freqs

# ------------------------
# Computes cutoff values
# ------------------------
def compute_equal_width_cutoffs(values, num_bins):
    """ Computes the cutoff points for the list of values
    Args:
        values(list): list of values to get cutoffs for
        num_bins(int): number of bins needed to get cutoffs for
    Returns:
        cutoffs(list of floats): cutoff points for the dataset 
    """
    # first compute the range of the values
    values_range = max(values) - min(values)
    bin_width = values_range / num_bins 
    cutoffs = list(np.arange(min(values), max(values), bin_width)) 
    cutoffs.append(max(values))
    # optionally: might want to round
    cutoffs = [round(cutoff, 2) for cutoff in cutoffs]
    return cutoffs 

# ------------------------
# Assign MPG Ratings (Approach 1)
# ------------------------
def assign_mpg_ratings(mpg):
    """ Assigns rating to an MPG
    Args:
        mpg(float): car miles per gallon rating
    Returns:
        rating(int): a rating from 1-10 based on the MPG value
    """
    rating = 0
    if (mpg < 14.0):
        rating = 1
    elif (mpg >= 14.0 and mpg < 15.0):
        rating = 2
    elif (mpg >= 15.0 and mpg < 17.0):
        rating = 3
    elif (mpg >= 17.0 and mpg < 20.0):
        rating = 4
    elif (mpg >= 20.0 and mpg < 24.0):
        rating = 5
    elif (mpg >= 24.0 and mpg < 27.0):
        rating = 6
    elif (mpg >= 27.0 and mpg < 31.0):
        rating = 7
    elif (mpg >= 31.0 and mpg < 37.0):
        rating = 8
    elif (mpg >= 37.0 and mpg < 45.0):
        rating = 9
    elif  (mpg >= 45.0):
        rating = 10
    
    return rating

# ------------------------
# Creates Histogram
# ------------------------
def create_histogram(data, axis_labels):
    """ Creates and shows a histogram when called
    Args:
        data(list): 1D list of data values
        axis_labels(list of strs): List of values to labels x axis, y axis, and title
    """
    plt.figure()
    # Sets titles and labels of the plot
    plt.title(axis_labels[0])
    plt.xlabel(axis_labels[1])
    plt.ylabel(axis_labels[2])
    # Plots (w/ 10 bins)
    plt.hist(data, bins=10) # default is 10
    # Displays the graph
    plt.show()

# ------------------------
# Creates Scatter Plot
# ------------------------
def create_scatter_plot(x, y, axis_labels):
    """ Creates A Scatter Plot
    Args:
        x(1D List): x axis numerical values
        y(1D List): y axis numerical values
        axis_labels(1D List or str): title and labels for each axis
    """
    plt.figure()
    # Sets titles and labels of the plot
    plt.title(axis_labels[0])
    plt.xlabel(axis_labels[1])
    plt.ylabel(axis_labels[2])
    # Computes Regression Line
    m, b = compute_slope_intercept(x, y)
    # Creates Scatter Plot
    plt.scatter(x, y)
    # Plots the Linear Regression Line
    plt.plot([min(x), max(x)], [m * min(x) + b, m * max(x) + b], c="r", lw=3);
    # Displays scatter plot
    plt.show()

# ------------------------
# Creates Scatter Plot
# ------------------------
def compute_slope_intercept(x, y):
    """ computes the slope intercept values for lin reg line
    Args:
        x(1D List): x axis numerical values
        y(1D List): y axis numerical values
    Returns:
        m, b (touple): m in y = mx + b and b in y = mx + b
    """
    mean_x = np.mean(x)
    mean_y = np.mean(y)
    
    m = sum([(x[i] - mean_x) * (y[i] - mean_y) for i in range(len(x))]) / sum([(x[i] - mean_x) ** 2 for i in range(len(x))])
    # y = mx + b => b = y - mx
    b = mean_y - m * mean_x
    return m, b 

# ------------------------
# Gets movie count per streaming service
# ------------------------
def get_movie_counts(table):
    ''' Finds the streaming service with the most hosted TV shows
        Args: 
            table(2D string list): contains the values from the .csv file
        Returns:  
            counts(list of ints): returns the number of movies per streaming service
    '''
    netflix_count = 0
    hulu_count = 0
    disney_count = 0
    prime_count = 0

    for i in range (len(table)):
        if (table[i][7] == 1):
            netflix_count = netflix_count + 1
        if (table[i][8] == 1):
            hulu_count = hulu_count + 1
        if (table[i][9] == 1):
            prime_count = prime_count + 1
        if (table[i][10] == 1):
            disney_count = disney_count + 1
    
    counts = [netflix_count, hulu_count, prime_count, disney_count]

    return counts

# ------------------------
# Gets movie count per streaming service
# ------------------------
def create_box_plot(data, axis_labels, x_labels, angle=90):
    """ Creates A Box Plot
    Args:
        data(list of 1D Lists): list of ratings per genre
        axis_labels(1D List of str): title and labels for each axis
        x_labels(1D list of str): xtick labels
        angle(int): angle of xticks
    """
    plt.figure()
    # Plots the Box Plot
    plt.boxplot(data)
    # Labels the x markers
    plt.xticks(list(range(1, len(x_labels) + 1)), x_labels, rotation=angle, horizontalalignment="center")
    # Makes the xticks readable
    plt.tick_params(axis='x', which='major', labelsize=8)
    # Sets the graph labels
    plt.title(axis_labels[0])
    plt.xlabel(axis_labels[1])
    plt.ylabel(axis_labels[2])
    plt.tight_layout()
    # Displays the Box Plot
    plt.show()

# ------------------------
# Create Lists of Lists
# ------------------------
def task_2_helper(genres, genres_updated, ratings):
    ''' Returns list of lists based on genres
        Args: 
            genres(1D list str): contains the genres from the .csv file
            genres_updated(1D list str): contains a list of unique genres
            ratings(1D list): contains either IMDb or rotten tomatoes ratings
        Returns:  
            boglist(list of lists): list of ratings per genre
    '''
    arr = []
    biglist = []

    # Goes through each genre and makes a list of ratings
    for j in range(len(genres_updated)):
        for i in range(len(genres)):
            if (genres_updated[j] in genres[i]) and ratings[i] != '':
                arr.append(float(ratings[i]))
        biglist.append(arr)
        arr = []

    return biglist







