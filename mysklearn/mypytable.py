import copy
import csv 
from tabulate import tabulate 

##############################################
# Programmer: Scott Campbell
# Class: CPSC 322-02, Spring 2021
# Programming Assignment #2
# 2/17/2020
# 
# Description: This class does basic data table operations on a dataset for use in common
#   data science problems.         
##############################################

class MyPyTable:
    """Represents a 2D table of data with column names.

    Attributes:
        column_names(list of str): M column names
        data(list of list of obj): 2D data structure storing mixed type data. There are N rows by M columns.
    """

    def __init__(self, column_names=None, data=None):
        """Initializer for MyPyTable.

        Args:
            column_names(list of str): initial M column names (None if empty)
            data(list of list of obj): initial table data in shape NxM (None if empty)
        """
        if column_names is None:
            column_names = []
        self.column_names = copy.deepcopy(column_names)
        if data is None:
            data = []
        self.data = copy.deepcopy(data)

    def pretty_print(self):
        """Prints the table in a nicely formatted grid structure.
        """
        print(tabulate(self.data, headers=self.column_names, tablefmt="fancy_grid"))

    def get_shape(self):
        """Computes the dimension of the table (N x M).

        Returns:
            int: number of rows in the table (N)
            int: number of cols in the table (M)
        """
        if not self.data: # Empty data array -> return 0x0
            return 0, 0
        else:
            return len(self.data), len(self.data[0]) 

    def get_column(self, col_identifier, include_missing_values=True):
        """Extracts a column from the table data as a list.

        Args:
            col_identifier(str or int): string for a column name or int
                for a column index
            include_missing_values(bool): True if missing values ("NA")
                should be included in the column, False otherwise.

        Returns:
            list of obj: 1D list of values in the column

        Notes:
            Raise ValueError on invalid col_identifier
        """
        data_rows, data_cols = self.get_shape()

        col_index = -1
        if type(col_identifier) is str: # Get column index from its string name
            col_index = self.column_names.index(col_identifier)
        elif type(col_identifier) is int: # Get column index from its int pos
            col_index = col_identifier

        # Check if the table is empty -> return an empty col array
        if data_rows == 0 or data_cols == 0: # Empty table
            return []
        # Check that a valid index was able to be extracted
        if col_index < 0 or col_index >= data_cols: 
            raise ValueError("Column Identifier Unknown")
  
        col = []
        for row in self.data:
            col.append(row[col_index])
        return col 

    def get_columns_by_indices(self, col_identifier, row_indices):
        data_rows, data_cols = self.get_shape()

        col_index = -1
        if type(col_identifier) is str: # Get column index from its string name
            col_index = self.column_names.index(col_identifier)
        elif type(col_identifier) is int: # Get column index from its int pos
            col_index = col_identifier
        # Check if the table is empty -> return an empty col array
        if data_rows == 0 or data_cols == 0: # Empty table
            return []
        # Check that a valid index was able to be extracted
        if col_index < 0 or col_index >= data_cols: 
            raise ValueError("Column Identifier Unknown")
        # Get the column vals for the specified row indices
        col = []
        for row_index in row_indices:
            if row_index < data_rows:
                col.append(self.data[row_index][col_index])
        return col 

    def get_multiple_columns(self, col_names):
        data_rows, data_cols = self.get_shape()
        # Get the col indices for the names
        col_indices = [self.column_names.index(name) for name in col_names]

        # Check if the table is empty -> return an empty col array
        if data_rows == 0 or data_cols == 0: # Empty table
            return []
  
        data = []
        for row in self.data:
            selected_cols = []
            for col_index in col_indices:
                selected_cols.append(row[col_index])
            data.append(selected_cols)
            
        return data 
    
    def get_subtable(self, start_row, end_row, start_col, end_col):
        if end_row == "end":
            end_row = len(self.data)
        if end_col == "end":
            end_col = len(self.data[0])
        # Get the subtable
        subtable = []
        for row_index in range(start_row, end_row, 1):
            sub_row = []
            for col_index in range(start_col, end_col, 1):
                sub_row += [self.data[row_index][col_index]]
            subtable.append(sub_row)
        # Get the subheader associated with it
        subheader = []
        for col_index in range(start_col, end_col, 1):
            subheader.append(self.column_names[col_index])

        return subheader, subtable
        

    def get_instances(self, row_indices):
        data_rows, _ = self.get_shape()

        instances = []
        for row_index in row_indices:
            if row_index < data_rows:
                instances.append(self.data[row_index])
        
        return instances

    def convert_to_numeric(self):
        """Try to convert each value in the table to a numeric type (float).

        Notes:
            Leave values as is that cannot be converted to numeric.
        """
        data_rows, data_cols = self.get_shape()

        for row in range(data_rows):
            for col in range(data_cols):
                try: # converting the value to a numeric
                    self.data[row][col] = float(self.data[row][col])
                except: # If not numeric, leave as is
                    self.data[row][col] = self.data[row][col]

    def drop_rows(self, rows_to_drop):
        """Remove rows from the table data.

        Args:
            rows_to_drop(list of list of obj): list of rows to remove from the table data.
        """   
        new_table = []
        for row in self.data:
            shouldnt_remove = True
            for row_to_delete in rows_to_drop:
                if row == row_to_delete:
                    shouldnt_remove = False
            if shouldnt_remove:
                new_table.append(row)

        self.data = copy.deepcopy(new_table)

    def load_from_file(self, filename):
        """Load column names and data from a CSV file.

        Args:
            filename(str): relative path for the CSV file to open and load the contents of.

        Returns:
            MyPyTable: return self so the caller can write code like: table = MyPyTable().load_from_file(fname)
        
        Notes:
            Use the csv module.
            First row of CSV file is assumed to be the header.
            Calls convert_to_numeric() after load
        """
        new_table = []
        with open(filename, newline='') as csvfile:
            filereader = csv.reader(csvfile, delimiter=',', quotechar='"')
            for row in filereader:
                new_table.append(row)
        
        # Save the loaded data as a MyPyTable
        self.column_names = copy.deepcopy(new_table[0])
        self.data = copy.deepcopy(new_table[1:])
        # Convert the array to numeric values if possible
        self.convert_to_numeric()
        return self 

    def save_to_file(self, filename):
        """Save column names and data to a CSV file.

        Args:
            filename(str): relative path for the CSV file to save the contents to.

        Notes:
            Use the csv module.
        """
        with open(filename, 'w', newline='') as csvfile:
            filewriter = csv.writer(csvfile, delimiter=',',
                                    quotechar='"', quoting=csv.QUOTE_MINIMAL)
            # Write the column names first
            filewriter.writerow(self.column_names)
            # Write the data next
            for row in self.data:
                filewriter.writerow(row)
    
    def table_contains(self, key_column_names, row_to_check, table):
        """Returns whether or not the two rows are the same based on the values in the key_column_names indices

        Args:
            key_column_names(list of str): column names to use as row keys.
            row_to_check(list): the row to see if it is contained in the table
            table(list of list obj): the table of data

        Returns: 
            True/False: if the table does/doesn't contain the row based on the key column values
        """
        # Get the indices of the columns to check
        col_key_indices = [self.column_names.index(key_name) for key_name in key_column_names]
        
        for instance in table:
            # See if the instance and row_to_check are the same
            rows_match = True
            for kk in col_key_indices:
                if row_to_check[kk] != instance[kk]:
                    rows_match = False
                    break
            if rows_match: # If the instance and row_to_check are the same, then the row is contained in table
                return True
        # The row_to_check was not found anywhere in the table for the key column values, so it isn't contained in table
        return False

    def find_duplicates(self, key_column_names):
        """Returns a list of duplicates. Rows are identified uniquely baed on key_column_names.

        Args:
            key_column_names(list of str): column names to use as row keys.

        Returns: 
            list of list of obj: list of duplicate rows found
        
        Notes:
            Subsequent occurrence(s) of a row are considered the duplicate(s). The first instance is not
                considered a duplicate.
        """
        table_no_duplicates = [] # Holds only one instance of each row (even if it has duplicates)
        duplicates = []
        for row in self.data:
            # Check if the row is already is the no duplicate set
            if self.table_contains(key_column_names, row_to_check=row, table=table_no_duplicates):
                duplicates.append(row)
            else: # Not yet a duplicate, add to the no duplicates list
                table_no_duplicates.append(row)

        return duplicates

    def remove_rows_with_missing_values(self, missing_str_label="NA"):
        """Remove rows from the table data that contain a missing value ("NA").
        """
        new_table = []
        for row in self.data:        
            if not (missing_str_label in row):
                new_table.append(row)
        self.data = copy.deepcopy(new_table)
    
    def remove_rows_with_missing_values_in_col(self, col_name, missing_str_label="NA"):
        """Remove rows from the table data that contain a missing value ("NA") in the column with col_name.
        """
        col_index = self.column_names.index(col_name)
        new_table = []
        for row in self.data:        
            if not (row[col_index] == missing_str_label):
                new_table.append(row)
        self.data = copy.deepcopy(new_table)

    def replace_missing_values_with_column_average(self, col_name, rounding_place=None, missing_str_label="NA"):
        """For columns with continuous data, fill missing values in a column by the column's original average.

        Args:
            col_name(str): name of column to fill with the original average (of the column).
        """
        # Go through the column and compute the average (if possible)
        col_sum, col_count = 0, 0
        the_col = self.get_column(col_name)
        for row in the_col:
            if row != missing_str_label:
                col_sum += row
                col_count += 1

        if type(col_sum) is float or type(col_sum) is int:
            col_avg = col_sum / col_count # Compute the average of the column
            
            if rounding_place != None:
                col_avg = round(col_avg, rounding_place)
            
            col_index = self.column_names.index(col_name)
            # Replace any NA entry in the data table with the col average
            for kk in range(len(the_col)):
                if the_col[kk] == missing_str_label:
                    self.data[kk][col_index] = col_avg

    def compute_median(self, numeric_list):
        """Calculates the median of list of numeric values

        Args:
            numeric_list(list of numerics): a list of numeric values to compute the median on

        Returns:
            A float representing the median value of numeric_list
        """
        num_elements = len(numeric_list)
        numeric_list.sort()
        if num_elements % 2 == 0: # Even number of elements
            return (numeric_list[int((num_elements / 2) - 1)] + numeric_list[int(num_elements / 2)]) / 2
        else: # Odd number of elements
            return numeric_list[int(num_elements / 2)]

    def compute_summary_statistics(self, col_names):
        """Calculates summary stats for this MyPyTable and stores the stats in a new MyPyTable.

        Args:
            col_names(list of str): names of the continuous columns to compute summary stats for.

        Returns:
            MyPyTable: stores the summary stats computed. The column names and their order is as 
            follows: ["attribute", "min", "max", "mid", "avg", "median"]
        """
        
        # If the table is not empty, then compute the stats on it
        summary_stats = []
        for col_name in col_names:
            # Get the associated column
            column = self.get_column(col_name)
            # Make a new list of all numeric values
            num_column = []
            for row in column:
                if (type(row) is float) or (type(row) is int):
                    num_column.append(row)
            # Check first if the table is empty
            if num_column != []:
                # Compute the stats on the numeric values of the column
                col_stats = []
                col_stats.append(col_name)
                col_stats.append(min(num_column))
                col_stats.append(max(num_column))
                col_stats.append((max(num_column) + min(num_column)) / 2)
                col_stats.append(sum(num_column) / len(num_column))
                col_stats.append(self.compute_median(num_column))
                # Append the stats for the column in the summary table
                summary_stats.append(col_stats)

        return MyPyTable(column_names=["attribute", "min", "max", "mid", "avg", "median"], data=summary_stats)

    def check_rows_match(self, row1, row2, table1_key_indices, table2_key_indices):
        """Checks if two rows for a data table match based on the values in the key_indices

        Args:
            row1(list of obj): the row for the first datatable
            row2(list of obj): the row for the second datatable
            table1_key_indices(list of int): the positions in row1 to check for match
            table2_key_indices(list of int): the positions in row2 to check for match

        Returns:
            Boolean: True/False if the rows do/don't match
        """
        for kk in range(len(table1_key_indices)):
            if row1[table1_key_indices[kk]] != row2[table2_key_indices[kk]]:
                return False
        # If make it to this point, then all of the rows key values matched
        return True

    def merge_rows(self, row1, row2, header1, header2):
        """Merges two rows together, without duplicate attributes

        Args:
            row1(list of obj): the row for the first datatable
            row2(list of obj): the row for the second datatable
            header1(list of str): the column names for row1
            header2(list of str): the column names for row2

        Returns:
            list of obj: a list of the two rows joined together (without duplicate attributes)
        """
        joined_row = []
        for attr in row1:
            joined_row.append(attr)
        for kk in range(len(row2)):
            # See if the current attribute in row2 is already in row 1
            curr_attr_name = header2[kk]
            if not (curr_attr_name in header1):
                # the arrtibute has not yet been joined in. Do it now
                joined_row.append(row2[kk])
        return joined_row

    def merge_rows_with_NA(self, row1, row2, header1, header2, table1_key_indices, table2_key_indices):
        """Merges two rows together, without duplicate attributes and filling in missing values with NA

        Args:
            row1(list of obj): the row for the first datatable
            row2(list of obj): the row for the second datatable
            header1(list of str): the column names for row1
            header2(list of str): the column names for row2
            table1_key_indices(list of int): the positions in row1 of the keys
            table2_key_indices(list of int): the positions in row2 of the keys

        Returns:
            list of obj: a list of the two rows joined together (without duplicate attributes)
        """
        if row1 == []:
            row1 = ["NA" for x in range(len(header1))] # Fill the key values with their values, otherwise with NA
            for kk in range(len(table1_key_indices)):
                row1[table1_key_indices[kk]] = row2[table2_key_indices[kk]]
        if row2 == []:
            row2 = ["NA" for x in range(len(header2))] # Fill the key values with their values, otherwise with NA

        return self.merge_rows(row1, row2, header1, header2)

    def perform_inner_join(self, other_table, key_column_names):
        """Return a new MyPyTable that is this MyPyTable inner joined with other_table based on key_column_names.

        Args:
            other_table(MyPyTable): the second table to join this table with.
            key_column_names(list of str): column names to use as row keys.

        Returns:
            MyPyTable: the inner joined table.
        """
        # Get the indices of each key for each table
        table1_key_indices = [self.column_names.index(key) for key in key_column_names]
        table2_key_indices = [other_table.column_names.index(key) for key in key_column_names]

        joined_column_names = self.merge_rows(self.column_names, other_table.column_names, self.column_names, other_table.column_names)
        joined_data = []
        for table_1_row in self.data:
            for table_2_row in other_table.data:
                if self.check_rows_match(table_1_row, table_2_row, table1_key_indices, table2_key_indices):
                    # The rows match, so merge them and add them together
                    merged_row = self.merge_rows(table_1_row, table_2_row, self.column_names, other_table.column_names)
                    joined_data.append(merged_row)

        return MyPyTable(column_names=joined_column_names, data=joined_data) 

    def perform_full_outer_join(self, other_table, key_column_names):
        """Return a new MyPyTable that is this MyPyTable fully outer joined with other_table based on key_column_names.

        Args:
            other_table(MyPyTable): the second table to join this table with.
            key_column_names(list of str): column names to use as row keys.

        Returns:
            MyPyTable: the fully outer joined table.

        Notes:
            Pad the attributes with missing values with "NA".
        """
        # Join the column names/ headers together
        joined_column_names = self.merge_rows(self.column_names, other_table.column_names, self.column_names, other_table.column_names)
        
        # Get the indices of each key for each table
        table1_key_indices = [self.column_names.index(key) for key in key_column_names]
        table2_key_indices = [other_table.column_names.index(key) for key in key_column_names]     
        joined_key_indices = [joined_column_names.index(key) for key in key_column_names]

        # Begin the outer join process
        joined_data = []
        for table_1_row in self.data:
            # See if there is a match with a row in the other table
            has_match_in_table_2 = False
            for table_2_row in other_table.data:
                if self.check_rows_match(table_1_row, table_2_row, table1_key_indices, table2_key_indices):
                    # The rows match, so merge them and add them together
                    merged_row = self.merge_rows(table_1_row, table_2_row, self.column_names, other_table.column_names)
                    joined_data.append(merged_row)
                    has_match_in_table_2 = True
            if not has_match_in_table_2: # There was no match in other table, add in here with NAs
                merged_row_with_NA = self.merge_rows_with_NA(table_1_row, [], self.column_names, other_table.column_names, table1_key_indices, table2_key_indices)
                joined_data.append(merged_row_with_NA)
        # Now go through every row in the other table and add in if there was no match in first table
        for table_2_row in other_table.data:
            has_match_in_joined_table = False
            for joined_row in joined_data:
                if self.check_rows_match(joined_row, table_2_row, joined_key_indices, table2_key_indices):
                    has_match_in_joined_table = True
        
            if not has_match_in_joined_table:
                # This row has not yet been added to the table, so add it in with NAs
                merged_row_with_NA = self.merge_rows_with_NA([], table_2_row, self.column_names, other_table.column_names, table1_key_indices, table2_key_indices)
                joined_data.append(merged_row_with_NA)

        return MyPyTable(column_names=joined_column_names, data=joined_data)

    def group_by(self, group_by_col_name):
        """ Gets the subtables grouped by the values in a column

        Args:
            group_by_col_name(str): the name of the column to get the subtables from

        Returns:
            group_names: the names of the groups
            group_subtables: the subtables of the group
        """

        col = self.get_column(group_by_col_name)
        col_index = self.column_names.index(group_by_col_name)

        # We need the unique values for out group by column
        group_names = sorted(list(set(col))) # e.g. 74, 75, 76, 77
        group_subtables = [[] for _ in group_names] # [ [], [], [], [] ]

        # algorithm: walk thorugh each row and assign to the appropriate subtable based on its group_by_col_name
        for row in self.data:
            group_by_value = row[col_index]
            group_by_index = group_names.index(group_by_value)
            group_subtables[group_by_index].append(row)

        return group_names, group_subtables

    def categorical_group_by(self, group_by_col_name):
        """ Gets the subtables grouped by the values in a column for categorical data

        Args:
            group_by_col_name(str): the name of the column to get the subtables from

        Returns:
            group_names: the names of the groups
            group_subtables: the subtables of the group
        """

        col = self.get_column(group_by_col_name)
        col_index = self.column_names.index(group_by_col_name)

        # We need the unique values for out group by column
        group_names = list(set(col)) # e.g. A, D, C, B
        group_subtables = [[] for _ in group_names] # [ [], [], [], [] ]

        # algorithm: walk thorugh each row and assign to the appropriate subtable based on its group_by_col_name
        for row in self.data:
            group_by_value = row[col_index]
            group_by_index = group_names.index(group_by_value)
            group_subtables[group_by_index].append(row)

        return group_names, group_subtables

    def categorical_group_by_multiple_vals(self, group_by_col_name):
        """ Gets the subtables grouped by the values in a column for categorical data. 
            The group by column can have multiple attributes, with each value seperated 
            by a comma, (e.g  attribute value: "Animal, Human, Mammal")

        Args:
            group_by_col_name(str): the name of the column to get the subtables from

        Returns:
            group_names: the names of the groups
            group_subtables: the subtables of the group
        """

        col = self.get_column(group_by_col_name)
        col_index = self.column_names.index(group_by_col_name)

        # Get all the unique names of all attribute values
        group_names = []
        for instance in col:
            if not instance.strip() == "":
                vals = instance.split(",")
                for val in vals:
                    if not (val.strip() == ""):
                        group_names.append(val.strip())
        group_names = list(set(group_names)) # e.g. A, D, C, B
        group_subtables = [[] for _ in group_names] # [ [], [], [], [] ]

        # algorithm: walk thorugh each row and assign to the appropriate subtable based on its group_by_col_name
        for row in self.data:
            group_by_values = row[col_index].split(",")
            for val in group_by_values:
                if not (val.strip() == ""):
                    group_by_index = group_names.index(val.strip())
                    group_subtables[group_by_index].append(row)

        return group_names, group_subtables
    
    def average_by_group(self, group_by_column_name, average_over_column_name):
        """ Gets the averages for a column based on its unique values

        Args:
            group_by_col_name(str): the name of the column to get the subtables from

        Returns:
            group_names: the names of the groups
            group_averages: the averages of each group
        """

        group_names, group_subtables = self.group_by(group_by_column_name)

        avg_col_index = self.column_names.index(average_over_column_name)
        group_sum = [0 for _ in range(len(group_names))]
        group_count = [0 for _ in range(len(group_names))]

        for kk in range(len(group_subtables)):
            table = group_subtables[kk]
            for row in table:
                if row[avg_col_index] != "NA":
                    group_sum[kk] += row[avg_col_index]
                    group_count[kk] += 1
        
        group_averages = [group_sum[x] / group_count[x] for x in range(len(group_sum))]

        return group_names, group_averages
    
    def get_instance_from_key_pairs(self, key_column_names, key_column_values):
        """
            Returns the instances in the data table that has the key columns with the specifed values
        """
        key_indices = [self.column_names.index(key) for key in key_column_names]
        rows = []

        for row in self.data:
            hasMatchingKeys = True
            for kk in range(len(key_indices)):
                if not row[key_indices[kk]] == key_column_values[kk]:
                    hasMatchingKeys = False
            if hasMatchingKeys:
                rows.append(row)
        
        return rows
            