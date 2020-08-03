import os
import pandas as pd
from collections import defaultdict

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import seaborn as sns
import test as t
os.chdir("C:\\Users\\NiceGaius\\Desktop\\UdacityLearning\\AugustMediumProject")
#df2011 = pd.read_csv("2011 Stack Overflow Survey Results.csv", encoding="latin-1")
#df2012=pd.read_csv("2012 Stack Overflow Survey Results.csv", encoding="latin-1")
#df2013=pd.read_csv("2013 Stack Overflow Survey Responses.csv", encoding="latin-1") #throws dtype warning on some columns
#df2014=pd.read_csv("2014 Stack Overflow Survey Responses.csv", encoding="latin-1")
#df2015=pd.read_csv("2015 Stack Overflow Developer Survey Responses.csv", encoding="latin-1") #throws dtype warning on some columns
#df2016=pd.read_csv("2016 Stack Overflow Survey Responses.csv", encoding="latin-1")
#df2017 = pd.read_csv("survey_results_public2017.csv")
df2018 = pd.read_csv("survey_results_public2018.csv",encoding="latin-1")
#df2019 = pd.read_csv("survey_results_public2019.csv")
#df2020 = pd.read_csv("survey_results_public2020.csv")    

#num_vars2017 = df2017[['Salary', 'CareerSatisfaction', 'HoursPerWeek','JobSatisfaction','StackOverflowSatisfaction']]
#drop_sal_df2017 = num_vars2017.dropna(subset=['Salary'], axis=0)
#fill_mean = lambda col: col.fillna(col.mean())

'''
fill_df2017 = drop_sal_df2017.apply(fill_mean, axis=0)
X = fill_df2017[['CareerSatisfaction','HoursPerWeek','JobSatisfaction','StackOverflowSatisfaction']]
y = fill_df2017['Salary']
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=.3,random_state=42)
lm_model = LienarRegression(normalize=True)
lm_model.fit(X_train,y_train)
y_test_preds = lm_model.predict(X_test)
'''

def clean_data(df):
    df = df.dropna(subset=['Salary'], axis=0)
    y = df['Salary']
    X = df.drop(['Respondent','Salary'], axis=1)
    #fill_mean = lambda col: col.fillna(col.mean())
    #fill_df = X.apply(fill_mean, axis=0)
    Xquants = X.select_dtypes(include=['float','int']).columns
    for col in Xquants:
        X[col].fillna((X[col].mean()),inplace=True)
    
    #Xquants = fill_mean(Xquants)
    Xobjectslist = X.select_dtypes(include='object').columns
    for col in Xobjectslist:
        X = pd.concat([X.drop(col,axis=1),pd.get_dummies(X[col],prefix=col,
                            prefix_sep='_',drop_first=True)], axis=1 )
   # Xcats = createdummy(X,Xobjectslist,dummy_na=False)
    return X,y
X,y = clean_data(df2018)    

'''def createdummy():
    for col in cat_cols:
        try:
            df=pd.concat([df.drop(col,axis=1),pd.get_dummies(df[col],prefix=col,
                            prefix_sep='_',drop_first=True,dummy_na=dummy_na)]), axis=1
        except:
            continue
        return df
'''
#import AllTogetherSolns as s
## Putting It All Together
#Helper functions
def clean_fit_linear_mod(df, response_col, test_size=.3, rand_state=42):
    '''
    INPUT:
    df - a dataframe holding all the variables of interest
    response_col - a string holding the name of the column
    test_size - a float between [0,1] about what proportion of data should be in the test dataset
    rand_state - an int that is provided as the random state for splitting the data into training and test

    OUTPUT:
    X - cleaned X matrix (dummy and mean imputation)
    y - cleaned response (just dropped na)
    test_score - float - r2 score on the test data
    train_score - float - r2 score on the test data
    lm_model - model object from sklearn
    X_train, X_test, y_train, y_test - output from sklearn train test split used for optimal model

    This function cleans the data and provides the necessary output for the rest of this notebook.
    '''
    #Dropping where the salary has missing values
    df  = df.dropna(subset=['Salary'], axis=0)

    #Drop columns with all NaN values
    df = df.dropna(how='all', axis=1)

    #Pull a list of the column names of the categorical variables
    cat_df = df.select_dtypes(include=['object'])
    cat_cols = cat_df.columns

    #dummy all the cat_cols
    for col in  cat_cols:
        df = pd.concat([df.drop(col, axis=1), pd.get_dummies(df[col], prefix=col, prefix_sep='_', drop_first=True, dummy_na=True)], axis=1)


    # Mean function
    fill_mean = lambda col: col.fillna(col.mean())
    # Fill the mean
    df = df.apply(fill_mean, axis=0)

    #Split into explanatory and response variables
    X = df.drop(response_col, axis=1)
    y = df[response_col]

    #Split into train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=rand_state)

    lm_model = LinearRegression(normalize=True) # Instantiate
    lm_model.fit(X_train, y_train) #Fit

    #Predict using your model
    y_test_preds = lm_model.predict(X_test)
    y_train_preds = lm_model.predict(X_train)

    #Score using your model
    test_score = r2_score(y_test, y_test_preds)
    train_score = r2_score(y_train, y_train_preds)

    return X, y, test_score, train_score, lm_model, X_train, X_test, y_train, y_test


def find_optimal_lm_mod(X, y, cutoffs, test_size = .30, random_state=42, plot=True):
    '''
    INPUT
    X - pandas dataframe, X matrix
    y - pandas dataframe, response variable
    cutoffs - list of ints, cutoff for number of non-zero values in dummy categorical vars
    test_size - float between 0 and 1, default 0.3, determines the proportion of data as test data
    random_state - int, default 42, controls random state for train_test_split
    plot - boolean, default 0.3, True to plot result

    OUTPUT
    r2_scores_test - list of floats of r2 scores on the test data
    r2_scores_train - list of floats of r2 scores on the train data
    lm_model - model object from sklearn
    X_train, X_test, y_train, y_test - output from sklearn train test split used for optimal model
    '''
    r2_scores_test, r2_scores_train, num_feats, results = [], [], [], dict()
    for cutoff in cutoffs:

        #reduce X matrix
        reduce_X = X.iloc[:, np.where((X.sum() > cutoff) == True)[0]]
        num_feats.append(reduce_X.shape[1])

        #split the data into train and test
        X_train, X_test, y_train, y_test = train_test_split(reduce_X, y, test_size = test_size, random_state=random_state)

        #fit the model and obtain pred response
        lm_model = LinearRegression(normalize=True)
        lm_model.fit(X_train, y_train)
        y_test_preds = lm_model.predict(X_test)
        y_train_preds = lm_model.predict(X_train)

        #append the r2 value from the test set
        r2_scores_test.append(r2_score(y_test, y_test_preds))
        r2_scores_train.append(r2_score(y_train, y_train_preds))
        results[str(cutoff)] = r2_score(y_test, y_test_preds)

    if plot:
        plt.plot(num_feats, r2_scores_test, label="Test", alpha=.5)
        plt.plot(num_feats, r2_scores_train, label="Train", alpha=.5)
        plt.xlabel('Number of Features')
        plt.ylabel('Rsquared')
        plt.title('Rsquared by Number of Features')
        plt.legend(loc=1)
        plt.show()

    best_cutoff = max(results, key=results.get)

    #reduce X matrix
    reduce_X = X.iloc[:, np.where((X.sum() > int(best_cutoff)) == True)[0]]
    num_feats.append(reduce_X.shape[1])

    #split the data into train and test
    X_train, X_test, y_train, y_test = train_test_split(reduce_X, y, test_size = test_size, random_state=random_state)

    #fit the model
    lm_model = LinearRegression(normalize=True)
    lm_model.fit(X_train, y_train)

    return r2_scores_test, r2_scores_train, lm_model, X_train, X_test, y_train, y_test


                          
cutoffs = [5000, 3500, 2500, 1000, 100, 50, 30, 25]
r2_scores_test, r2_scores_train, lm_model, X_train, X_test, y_train, y_test = find_optimal_lm_mod(X, y, cutoffs)

