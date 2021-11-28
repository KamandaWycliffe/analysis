def data_prep(inp_file, out_file):
    import pandas as pd

    import numpy as np
    df = pd.read_csv(inp_file, header = None, encoding = 'ansi', skiprows = 1)
    #Define column names
    df.columns = ['age', 'workclass', 'fnlwgt', 'education', 'education-num',
       'marital-status', 'occupation', 'relationship', 'race', 'sex',
       'capital-gain', 'capital-loss', 'hours-per-week', 'native-country',
       'income']
    
    '''(1) remove instances that have any missing values'''
    df = df.replace(' ?',np.NaN)
    df = df.dropna()
    
    '''(2) remove attributes: fnlwgt, education-num, relationship'''
    
    df = df.drop(['fnlwgt', 'education-num', 'relationship'], axis = 1)
    
    '''(3) binarize the following attributes'''
    
    #capital-gain (yes: >0, no: =0)
    
    df.loc[df["capital-gain"] > 0, "capital-gain"] = 'yes'

    df.loc[df["capital-gain"] == 0, "capital-gain"] = 'no'
    
    #capital-loss (yes: >0, no: =0)
    
    df.loc[df["capital-loss"] > 0, "capital-loss"] = 'yes'

    df.loc[df["capital-loss"] == 0, "capital-loss"] = 'no'
    
    #native country (United-States, other)
    df.loc[df["native-country"] != ' United-States', "native-country"] = 'other'
    
    df.loc[df["native-country"] == ' United-States', "native-country"] = 'United-States'
    
    '''(4) discretize continuous attributes:'''
    
    #age: divide values of age to 4 levels: young (<=25), adult ([26,45]), senior ([46,65]), and old ([66,90]).
    df['age'] = pd.cut(df.age,bins=[0, 25, 45, 65, 90],
                     labels=["young", "adult", "senior", "old"])
    
    #hours-per-week: divide the values into 3 levels: part-time (<40), full-time (=40), over-time (>40)
    conditions = [
    (df['hours-per-week'] < 40),
    (df['hours-per-week'] == 40),
    (df['hours-per-week'] > 40 )]
    choices = ['part-time', 'full-time', 'over-time']


    df['hours-per-week'] = np.select(conditions, choices, default='black')
    
    '''(5) merge attribute values together/reassign attribute values:'''
    #workclass: create the following 4 values: gov (Federal-gov, Local-gov, State-gov), 
    #Not- working (Without-pay, Never-worked), Private, Self-employed (Self-emp-inc, Self-emp-not- inc)
    
    conditions = [
    (df['workclass'] == ' Federal-gov') | (df['workclass'] == ' Local-gov') | (df['workclass'] == ' State-gov'),
    (df['workclass'] == ' Without-pay') | (df['workclass'] == ' Never-worked'),
    (df['workclass'] == ' Private'), (df['workclass'] == ' Self-emp-inc') | (df['workclass'] == ' Self-emp-not-inc')]
    
    choices = ['gov', 'Not-working', 'Private', 'Self-employed']

    # Add new column based on conditions and choices:
    df['workclass'] = np.select(conditions, choices, default='black')
    
    
    
    #education: create the following 5 values: BeforeHS (Preschool, 1st-4th, 5th-6th, 7th-8th, 9th,
    #10th, 11th, 12th), HS-grad, AfterHS (Prof-school, Assoc-acdm, Assoc-voc, Some-college), UGrD, GrD (Masters, Doctorate).
    
    conditions = [
    (df['education'] == ' Preschool') | (df['education'] == ' 10th') | (df['education'] == ' 5th-6th') 
        | (df['education'] == ' 7th-8th') | (df['education'] == ' 9th') | (df['education'] == ' 1st-4th') 
        | (df['education'] == ' 11th') | (df['education'] == ' 12th'),
    (df['education'] == ' Prof-school') | (df['education'] == ' Assoc-acdm') | (df['education'] == ' Assoc-voc') |
        (df['education'] == ' Some-college'),
    (df['education'] == ' HS-grad'),(df['education'] == ' UGrD') | (df['education'] == ' Bachelors'), 
        (df['education'] == ' Masters') | (df['education'] == ' Doctorate')]
    
    choices = ['BeforeHS', 'AfterHS', 'HS-grad', 'UGrD', 'GrD']

    # Add new column based on conditions and choices:
    df['education'] = np.select(conditions, choices, default='black')
    
    
    #marital-status: create the following 3 values: Married (Married-AF-spouse, Married-civ-spouse), 
    #Never-married, Not-married (Married-spouse-absent, Separated, Divorced, Widowed)
    
    conditions = [
    (df['marital-status'] == ' Married-AF-spouse') | (df['marital-status'] == ' Married-civ-spouse'),
    (df['marital-status'] == ' Married-spouse-absent') | (df['marital-status'] == ' Separated') | 
        (df['marital-status'] == ' Divorced') | (df['marital-status'] == ' Widowed'),
    (df['marital-status'] == ' Never-married')]
    
    choices = ['Married', 'Not-married', 'Never-married']
    
    df['marital-status'] = np.select(conditions, choices, default='black')
    
    
    #occupation: create the following 5 values: Exec-managerial, Prof-specialty, Other (Tech-
    #support, Adm-clerical, Priv-house-serv, Protective-serv, Armed-Forces, Other-service),
    #ManualWork (Craft-repair, Farming-fishing, Handlers-cleaners, Machine-op-inspct, Transport-moving), Sales.
    
    conditions = [(df['occupation'] == ' Exec-managerial'), (df['occupation'] == ' Prof-specialty'), 
    (df['occupation'] == ' Tech-support') | (df['occupation'] == ' Adm-clerical') | (df['occupation'] == ' Priv-house-serv') 
        | (df['occupation'] == ' Protective-serv') | (df['occupation'] == ' Armed-Forces')
                  | (df['occupation'] == ' Other-service'),
    (df['occupation'] == ' Craft-repair') | (df['occupation'] == ' Farming-fishing') |
                  (df['occupation'] == ' Handlers-cleaners') |
        (df['occupation'] == ' Machine-op-inspct') | (df['occupation'] == ' Transport-moving'), 
                 (df['occupation'] == ' Sales')]
    
    choices = ['Exec-managerial', 'Prof-specialty', 'Other', 'ManualWork', 'Sales']

    # Add new column based on conditions and choices:
    df['occupation'] = np.select(conditions, choices, default='black')
    df['income'] = df.income.str.replace('.','', regex=True)
    #use this to output to a file
    df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)
    
    
    return df.to_csv(out_file, index = False)

if __name__ == '__main__':
    import sys
    inp_file = str(sys.argv[1])
    out_file = str(sys.argv[2])
    data_prep(inp_file, out_file)