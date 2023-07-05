def gini_index(y):
    import numpy as np
    import pandas as pd
    '''
      Given a Pandas Series, it calculates the Gini Impurity. 
      y: variable with which calculate Gini Impurity.
      '''
    if isinstance(y, pd.Series):
        p = y.value_counts()/y.shape[0]
        gini = 1-np.sum(p**2)
        return(gini)

    else:
        raise('Object must be a Pandas Series.')

#Check purity
def check_purity(data):
    import numpy as np
    import pandas as pd
    label_column = data[:, -1]
    unique_classes = np.unique(label_column)

    if len(unique_classes) == 1:
        return True
    else:
        return False
def gini_index(y):
    import pandas as pd
    import numpy as np
    '''
  Given a Pandas Series, it calculates the Gini Impurity. 
  y: variable with which calculate Gini Impurity.
   '''
    if isinstance(y, pd.Series):
        p = y.value_counts()/y.shape[0]
        gini = 1-np.sum(p**2)
        return(gini)

    else:
        raise('Input needs to be a Pandas Series.')

def information_gain(y, mask, func=gini_index):
    '''
      It returns the Information Gain of a variable given a loss function.
      y: target variable.
      mask: split choice.
      func: function to be used to calculate Information Gain in case os classification.
    '''

    a = sum(mask)
    b = mask.shape[0] - a

    if(a == 0 or b ==0): 
        ig = 0

    else:
        if y.dtypes != 'O':
            ig = variance(y) - (a/(a+b)* variance(y[mask])) - (b/(a+b)*variance(y[-mask]))
        else:
            ig = func(y)-a/(a+b)*func(y[mask])-b/(a+b)*func(y[-mask])

    return ig


def get_best_split(y, data):
    '''
      Given a data, select the best split and return the variable, the value, the variable type and the information gain.
      y: name of the target variable
      data: dataframe where to find the best split.
    '''
    masks = data.drop(y, axis= 1).apply(max_information_gain_split, y = data[y])
    if sum(masks.loc[3,:]) == 0:
        return(None, None, None, None)

    else:
        # Get only masks that can be splitted
        masks = masks.loc[:,masks.loc[3,:]]

        # Get the results for split with highest IG
        split_variable = max(masks)
        #split_valid = masks[split_variable][]
        split_value = masks[split_variable][1] 
        split_ig = masks[split_variable][0]
        split_numeric = masks[split_variable][2]

    return(split_variable, split_value, split_ig, split_numeric)






import itertools
import pandas as pd
import numpy as np
import ast


def categorical_options(a):
  '''
  Creates all possible combinations from a Pandas Series.
  a: Pandas Series from where to get all possible combinations. 
  '''
  a = a.unique()

  options = []
  for L in range(0, len(a)+1):
      for subset in itertools.combinations(a, L):
          subset = list(subset)
          options.append(subset)

  return options[1:-1]

def max_information_gain_split(x, y, func=gini_index):
  '''
  Given a predictor & target variable, returns the best split, the error and the 
  type of variable based on a selected cost function.
  x: predictor variable as Pandas Series.
  y: target variable as Pandas Series.
  func: function to be used to calculate the best split.
  '''

  split_value = []
  ig = [] 

  numeric_variable = True if x.dtypes != 'O' else False

  # Create options according to variable type
  if numeric_variable:
    options = x.sort_values().unique()[1:]
  else:  
    options = categorical_options(x)

  # Calculate ig for all values
  for val in options:
    mask =   x < val if numeric_variable else x.isin(val)
    val_ig = information_gain(y, mask, func)
    # Append results
    ig.append(val_ig)
    split_value.append(val)

  # Check if there are more than 1 results if not, return False
  if len(ig) == 0:
    return(None,None,None, False)

  else:
  # Get results with highest IG
    best_ig = max(ig)
    best_ig_index = ig.index(best_ig)
    best_split = split_value[best_ig_index]
    return(best_ig,best_split,numeric_variable, True)


def maximum_information_gain_split(x, y, func=gini_index):
      '''
      Given a predictor & target variable, returns the best split, the error and the 
      type of variable based on a selected cost function.
      x: predictor variable as Pandas Series.
      y: target variable as Pandas Series.
      func: function to be used to calculate the best split.
      '''

      split_value = []
      ig = [] 

      numeric_variable = True if x.dtypes != 'O' else False

      # Create options according to variable type
      if numeric_variable:
        options = x.sort_values().unique()[1:]
      else:  
        options = categorical_options(x)

      # Calculate ig for all values
      for val in options:
        mask =   x < val if numeric_variable else x.isin(val)
        val_ig = information_gain(y, mask, func)
        # Append results
        ig.append(val_ig)
        split_value.append(val)

      # Check if there are more than 1 results if not, return False
      if len(ig) == 0:
        return(None,None,None, False)

      else:
      # Get results with highest IG
        best_ig = max(ig)
        best_ig_index = ig.index(best_ig)
        best_split = split_value[best_ig_index]
        return(best_ig,best_split,numeric_variable, True)





def get_best_split(y, data):
    '''
      Given a data, select the best split and return the variable, the value, the variable type and the information gain.
      y: name of the target variable
      data: dataframe where to find the best split.
    '''
    masks = data.drop(y, axis= 1).apply(max_information_gain_split, y = data[y])
    if sum(masks.loc[3,:]) == 0:
        return(None, None, None, None)

    else:
        # Get only masks that can be splitted
        masks = masks.loc[:,masks.loc[3,:]]

        # Get the results for split with highest IG
        split_variable = max(masks)
        #split_valid = masks[split_variable][]
        split_value = masks[split_variable][1] 
        split_ig = masks[split_variable][0]
        split_numeric = masks[split_variable][2]

    return(split_variable, split_value, split_ig, split_numeric)


def classify_data(data):
    import numpy as np
    label_column = data[:, -1]
    unique_classes, counts_unique_classes = np.unique(label_column, return_counts=True)
	
    index = counts_unique_classes.argmax()
    classification = unique_classes[index]
    
    return classification


def get_potential_splits(data):
    import numpy as np
    potential_splits = {}
    _, n_columns = data.shape
    for column_index in range(n_columns - 1):          # excluding the last column which is the label
        values = data[:, column_index]
        unique_values = np.unique(values)
        
        potential_splits[column_index] = unique_values
    
    return potential_splits


def split_data(data, split_column, split_value):
    import numpy as np
    split_column_values = data[:, split_column]

    type_of_feature = FEATURE_TYPES[split_column]
    if type_of_feature == "continuous":
        data_below = data[split_column_values <= split_value]
        data_above = data[split_column_values >  split_value]
    
    # feature is categorical   
    else:
        data_below = data[split_column_values == split_value]
        data_above = data[split_column_values != split_value]
    
    return data_below, data_above


def calculate_entropy(data):
    import numpy as np
    label_column = data[:, -1]
    _, counts = np.unique(label_column, return_counts=True)

    probabilities = counts / counts.sum()
    entropy = sum(probabilities * -np.log2(probabilities))
     
    return entropy

def calculate_overall_entropy(data_below, data_above):
    import numpy as np
    n = len(data_below) + len(data_above)
    p_data_below = len(data_below) / n
    p_data_above = len(data_above) / n

    overall_entropy =  (p_data_below * calculate_entropy(data_below) 
                      + p_data_above * calculate_entropy(data_above))
    
    return overall_entropy

def determine_best_split(data, potential_splits):
    import numpy as np
    overall_entropy = 9999
    for column_index in potential_splits:
        for value in potential_splits[column_index]:
            data_below, data_above = split_data(data, split_column=column_index, split_value=value)
            current_overall_entropy = calculate_overall_entropy(data_below, data_above)

            if current_overall_entropy <= overall_entropy:
                overall_entropy = current_overall_entropy
                best_split_column = column_index
                best_split_value = value
    
    return best_split_column, best_split_value


def determine_type_of_feature(df):
    import numpy as np
    feature_types = []
    n_unique_values_treshold = 15
    for feature in df.columns:
        if feature != "label":
            unique_values = df[feature].unique()
            example_value = unique_values[0]

            if (isinstance(example_value, str)) or (len(unique_values) <= n_unique_values_treshold):
                feature_types.append("categorical")
            else:
                feature_types.append("continuous")
    
    return feature_types

def DecisionTree(df, counter=0, min_samples=2, minfreq=5):
    
    # data preparations
    if counter == 0:
        global COLUMN_HEADERS, FEATURE_TYPES
        COLUMN_HEADERS = df.columns
        FEATURE_TYPES = determine_type_of_feature(df)
        data = df.values
    else:
        data = df           
    
    
    # base cases
    if (check_purity(data)) or (len(data) < min_samples) or (counter == minfreq):
        classification = classify_data(data)
        
        return classification

    
    # recursive part
    else:    
        counter += 1

        # helper functions 
        potential_splits = get_potential_splits(data)
        split_column, split_value = determine_best_split(data, potential_splits)
        data_below, data_above = split_data(data, split_column, split_value)
        
        # check for empty data
        if len(data_below) == 0 or len(data_above) == 0:
            classification = classify_data(data)
            return classification
        
        # determine question
        feature_name = COLUMN_HEADERS[split_column]
        type_of_feature = FEATURE_TYPES[split_column]
        if type_of_feature == "continuous":
            question = "n" + str(counter) + " <= {}".format(feature_name, split_value)
            
        # feature is categorical
        else:
            question = "n" + str(counter) +" " + "{} , {}".format(feature_name, split_value)
            
        
        # instantiate sub-tree
        sub_tree = {question: []}
        
        # find answers (recursion)
        yes_answer = DecisionTree(data_below, counter, min_samples, minfreq)
        no_answer = DecisionTree(data_above, counter, min_samples, minfreq)
        
        # If the answers are the same, then there is no point in asking the qestion.
        # This could happen when the data is classified even though it is not pure
        # yet (min_samples or minfreq base case).
        if yes_answer == no_answer:
            sub_tree = yes_answer
        else:
            sub_tree[question].append(yes_answer)
            sub_tree[question].append(no_answer)
        
        return sub_tree
#For classification
    
    
def DTree(df, counter=0, min_samples=2, minfreq=5):
    
    # data preparations
    if counter == 0:
        global COLUMN_HEADERS, FEATURE_TYPES
        COLUMN_HEADERS = df.columns
        FEATURE_TYPES = determine_type_of_feature(df)
        data = df.values
    else:
        data = df           
    
    
    # base cases
    if (check_purity(data)) or (len(data) < min_samples) or (counter == minfreq):
        classification = classify_data(data)
        
        return classification

    
    # recursive part
    else:    
        counter += 1

        # helper functions 
        potential_splits = get_potential_splits(data)
        split_column, split_value = determine_best_split(data, potential_splits)
        data_below, data_above = split_data(data, split_column, split_value)
        
        # check for empty data
        if len(data_below) == 0 or len(data_above) == 0:
            classification = classify_data(data)
            return classification
        
        # determine question
        feature_name = COLUMN_HEADERS[split_column]
        type_of_feature = FEATURE_TYPES[split_column]
        if type_of_feature == "continuous":
            question = "{} <= {}".format(feature_name, split_value)
            
        # feature is categorical
        else:
            question = "{} = {}".format(feature_name, split_value)
        
        # instantiate sub-tree
        sub_tree = {question: []}
        
        # find answers (recursion)
        yes_answer = DTree(data_below, counter, min_samples, minfreq)
        no_answer = DTree(data_above, counter, min_samples, minfreq)
        
        # If the answers are the same, then there is no point in asking the qestion.
        # This could happen when the data is classified even though it is not pure
        # yet (min_samples or minfreq base case).
        if yes_answer == no_answer:
            sub_tree = yes_answer
        else:
            sub_tree[question].append(yes_answer)
            sub_tree[question].append(no_answer)
        
        return sub_tree

def build(trainfile, modelfile, minfreq):
    import pandas as pd
    import numpy as np
    minfreq = minfreq
    from pprint import pprint
    train_df = pd.read_csv(trainfile)
    tree = DecisionTree(train_df, minfreq=minfreq)
    model = tree
    model = str(model)
    model = model.replace("[", "")
    model = model.replace("{", "")
    model = model.replace("'", "")
    model = model.replace("]", "")
    model = model.replace("}}", "\n")
    model = model.replace("}", "")
    model = model.replace(",", "")
    model = model.replace(":", " : ")
    import sys

    orig_stdout = sys.stdout
    f = open(modelfile, 'w')
    sys.stdout = f

    print(model)

    sys.stdout = orig_stdout
    f.close()
if __name__ == '__main__':
    import sys
    import numpy as np
    import pandas as pd
    trainfile = str(sys.argv[1])
    modelfile = str(sys.argv[2])
    minfreq = int(sys.argv[3])
    build(trainfile, modelfile, minfreq)