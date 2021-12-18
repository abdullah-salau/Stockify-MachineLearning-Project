# -*- coding: utf-8 -*-
"""
Created on Sun Sep 12 02:34:40 2021

@author: abdul
"""
# Running this python file will print and plot the best scores that Stockify was able to obtain
# in Classification Analysis (Whether a stock will rise or drop) utilizing Random Forests.

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer
from sklearn import model_selection
import seaborn as sn
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
import scipy.spatial as sp
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.neighbors import KNeighborsClassifier
import sklearn
from numpy import mean
from numpy import std
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import RepeatedKFold
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import IsolationForest
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import sys

def main():
    # # Read the financial indicator data files
    #DELETE
    # df = pd.read_csv("StockDatasets/2014_Financial_Data.csv")
    # #print(df)
    # df = df.iloc[:,1:]
    # print(df)
    # #Names of Columns
    # colNames = list(df.columns)
    # #Separate inputCols and outputCol, return with df
    # inputCols = colNames
    # inputCols.remove("Class")
    # outputCol = "Class"
    df, inputCols, outputCol= readData() 
    
    print("Number of missing values for each variable \n",df.isnull().sum() )
    #inputDF, inputCols, outputSeries = preprocess(df, inputCols, outputCol)
    #demonstrateHelpers(df)
    buildAndTestModel(df, inputCols, outputCol)
    #return df, inputCols, outputCol
    #plotAccuracies()
    

def readData():
    df = pd.read_csv("StockDatasets/2014_Financial_Data.csv")
    df = df.iloc[:,1:]
    colNames = list(df.columns)
    inputCols = colNames
    inputCols.remove("Class")
    outputCol = "Class"
    return df, inputCols, outputCol
    #MIXING UP DATASET
    # Need to mix this up before doing CV
    #wineDF = wineDF.sample(frac=1, random_state=50).reset_index(drop=True)
    
def kFoldCVBuiltIn1(i, inputDF, outputSeries):
    #Should we include variable to test different solvers? (liblinear, sag, saga)
    model = LogisticRegression(max_iter=500)
    cvScores = model_selection.cross_val_score(model,
                                               inputDF, outputSeries, cv=i) 
    
    return cvScores.mean()

def kFoldCVBuiltIn2(i, inputDF, outputSeries):
    #Should we include variable to test different solvers? (liblinear, sag, saga)
    model = LogisticRegression(solver='liblinear',max_iter=500)
    cvScores = model_selection.cross_val_score(model,
                                               inputDF, outputSeries, cv=i) 
    
    return cvScores.mean()

def kFoldCVBuiltIn3(i, inputDF, outputSeries):
    #Should we include variable to test different solvers? (liblinear, sag, saga)
    model = LogisticRegression(solver='sag',max_iter=500)
    cvScores = model_selection.cross_val_score(model,
                                               inputDF, outputSeries, cv=i) 
    
    return cvScores.mean()

def kFoldCVBuiltIn4(i, inputDF, outputSeries):
    #Should we include variable to test different solvers? (liblinear, sag, saga)
    model = LogisticRegression(solver='saga',max_iter=500)
    cvScores = model_selection.cross_val_score(model,
                                               inputDF, outputSeries, cv=i) 
    
    return cvScores.mean()

def kFoldCVBuiltGradBoost(i, inputDF, outputSeries):
    #Should we include variable to test different solvers? (liblinear, sag, saga)
    model = GradientBoostingClassifier(learning_rate=0.05, loss='exponential', max_depth=2, n_estimators=50, 
                                       subsample=0.5)
    cv = RepeatedStratifiedKFold(n_splits=i, n_repeats=3, random_state=1)
    cvScores = model_selection.cross_val_score(model,
                                               inputDF, outputSeries, scoring='accuracy', n_jobs=-1, cv=cv) 
    
    return cvScores.mean()

def kFoldCVBuiltInRandomForest(i, inputDF, outputSeries):
    
    model = RandomForestClassifier(bootstrap=True, max_features='sqrt', max_samples=0.5, n_estimators=250)
    #evaluating randomForest Model 
    cv = RepeatedKFold(n_splits=i, n_repeats=3, random_state=1)
    n_scores = cross_val_score(model, inputDF, outputSeries, scoring='accuracy', cv=cv, n_jobs=-1)
    
    return n_scores.mean()


def demonstrateHelpers(trainDF):
    print("Attributes with missing values:", getAttrsWithMissingValues(trainDF), sep='\n')
    
    numericAttrs = getNumericAttrs(trainDF)
    print("Numeric attributes:", numericAttrs, sep='\n')
    
    nonnumericAttrs = getNonNumericAttrs(trainDF)
    print("Non-numeric attributes:", nonnumericAttrs, sep='\n')

    print("Values, for each non-numeric attribute:", getAttrToValuesDictionary(trainDF.loc[:, nonnumericAttrs]), sep='\n')


'''
Returns a dictionary mapping an attribute to the array of values for that attribute.
'''
def getAttrToValuesDictionary(df):
    attrToValues = {}
    for attr in df.columns.values:
        attrToValues[attr] = df.loc[:, attr].unique()

    return attrToValues

# ===============================================================================
'''
Returns the attributes with missing values.
'''
def getAttrsWithMissingValues(df):
    valueCountSeries = df.count(axis=0)  # 0 to count down the rows
    numCases = df.shape[0]  # Number of examples - number of rows in the data frame
    missingSeries = (numCases - valueCountSeries)  # A Series showing the number of missing values, for each attribute
    attrsWithMissingValues = missingSeries[missingSeries != 0].index
    return attrsWithMissingValues

# =============================================================================

'''
Returns the numeric attributes.
'''
def getNumericAttrs(df):
    return __getNumericHelper(df, True)

'''
Returns the non-numeric attributes.
'''
def getNonNumericAttrs(df):
    return __getNumericHelper(df, False)

def __getNumericHelper(df, findNumeric):
    isNumeric = df.applymap(np.isreal) # np.isreal is a function that takes a value and returns True (the value is real) or False
                                       # applymap applies the given function to the whole data frame
                                       # So this returns a DataFrame of True/False values indicating for each value in the original DataFrame whether it is real (numeric) or not

    isNumeric = isNumeric.all() # all: For each column, returns whether all elements are True
    attrs = isNumeric.loc[isNumeric==findNumeric].index # selects the values in isNumeric that are <findNumeric> (True or False)
    return attrs

# =============================================================================

def preprocess(df, inputCols, outputCol):
    #Divide DataFrame into inputDF and outputSeries
    inputDF = df.loc[:,inputCols]
    outputSeries = df.loc[:,outputCol]
    print(inputDF)
    print(outputSeries)
    
    print("LOOK HERE")
    #Advanced Preprocess (1)
    print("inputCols length is ",len(inputCols))
    inputCols, inputDF = removeMissing(inputCols, inputDF)
    # inputCols, inputDF = removeLowVar(inputCols, inputDF)
    #removeLowCorrWTV(inputCols, inputDF, outputSeries)
    print("After removeMissing, inputCols length is now ",len(inputCols))
    
    #Deleting columns with no values and Price var
    inputDF = inputDF.drop(columns=['operatingProfitMargin','2015 PRICE VAR [%]'])#,'cashConversionCycle','operatingCycle'
    #inputCols.remove('cashConversionCycle')
    #inputCols.remove('operatingCycle')
    inputCols.remove('operatingProfitMargin') #All values are 1 except missing values
    inputCols.remove('2015 PRICE VAR [%]')
    #Note: When advanced preprocessing 1 is not utilized, we manually remove cashConversionCyclel and operatingCycle
    
    #If the skewness is between -0.5 and 0.5, the data are fairly symmetrical. Therefore, we use mean to fill in missing values
    #Otherwise, use median
    # for attr in inputCols:
    #     print(attr,inputDF.loc[:,attr].skew(), sep=": ")
    numCols = inputCols.copy()
    numCols.remove('Sector')
    print(numCols)
        
    for attr in numCols:
        if(inputDF.loc[:,attr].skew()<1 and inputDF.loc[:,attr].skew()>-1):
            inputDF.loc[:,attr] = inputDF.loc[:,attr].fillna(inputDF.loc[:,attr].mean())
        else:
            inputDF.loc[:,attr] = inputDF.loc[:,attr].fillna(inputDF.loc[:,attr].median())
        
    print(inputDF)
    print("Attributes with missing values:", getAttrsWithMissingValues(inputDF), sep='\n')
    #--------------------------------------------------------------------------------------------
    
    # Advanced Preprocessing (2)
    #Removing Variables with low variation (just realized i named it wrongly)
    print("LOOK HERE")
    print("inputCols length is ",len(inputCols))
    #inputCols, inputDF = removeLowVar(inputCols, inputDF)
    print("After removeHighVar, inputCols length is now ",len(inputCols))
    
    # Advanced Preprocessing (3)
    #Removing Variables with low correlation with target variable
    print("LOOK HERE")
    print("inputCols length is ",len(inputCols))
    inputCols, inputDF, outputSeries = removeLowCorrWTV(inputCols, inputDF, outputSeries)
    print("After removeLowCorrWTV, inputCols length is now ",len(inputCols))
    #One Hot Encoding the Sector Attribute
    
    #Sector Labels
    #print(inputDF.loc[:,'Sector'].unique())
    
    #Unprocessed inputDF with no missing values, but Sector dtype remains string
    uInputDF = inputDF.copy()
    
    print(inputDF.loc[:,'Sector'])
    #Convert sector dtype to float
    uInputDF.loc[:, "Sector"] = uInputDF.loc[:, "Sector"].map(lambda v: 0 if v=="Basic Materials" else v)
    uInputDF.loc[:, "Sector"] = uInputDF.loc[:, "Sector"].map(lambda v: 1 if v=="Communication Services" else v)
    uInputDF.loc[:, "Sector"] = uInputDF.loc[:, "Sector"].map(lambda v: 2 if v=="Consumer Cyclical" else v)
    uInputDF.loc[:, "Sector"] = uInputDF.loc[:, "Sector"].map(lambda v: 3 if v=="Consumer Defensive" else v)
    uInputDF.loc[:, "Sector"] = uInputDF.loc[:, "Sector"].map(lambda v: 4 if v=="Energy" else v)
    uInputDF.loc[:, "Sector"] = uInputDF.loc[:, "Sector"].map(lambda v: 5 if v=="Financial Services" else v)
    uInputDF.loc[:, "Sector"] = uInputDF.loc[:, "Sector"].map(lambda v: 6 if v=="Healthcare" else v)
    uInputDF.loc[:, "Sector"] = uInputDF.loc[:, "Sector"].map(lambda v: 7 if v=="Industrials" else v)
    uInputDF.loc[:, "Sector"] = uInputDF.loc[:, "Sector"].map(lambda v: 8 if v=="Real Estate" else v)
    uInputDF.loc[:, "Sector"] = uInputDF.loc[:, "Sector"].map(lambda v: 9 if v=="Technology" else v)
    uInputDF.loc[:, "Sector"] = uInputDF.loc[:, "Sector"].map(lambda v: 10 if v=="Utilities" else v)
    
    print(inputDF.loc[:,'Sector'])
    print(uInputDF.loc[:,'Sector'])
    #Value Counts
    print(inputDF.loc[:,'Sector'].value_counts())
    
    #Advanced Preprocessing (4)
    #Log Transformations: we center all numerical variables with skew greater than |10|
    # We use log1p for varaibles with positive skewness, and we use exponential logarithm for 
    # variables with negative skewness
    numCols = inputCols.copy()
    numCols.remove('Sector')
    print("LOOK HERE FOR SKEW VALUES")
    print("Attributes with missing values:", getAttrsWithMissingValues(inputDF), sep='\n')
    for attr in numCols:
        print(attr, " ", inputDF.loc[:,attr].skew())
        if(inputDF.loc[:,attr].skew()>10):
            var_copy = inputDF.loc[:,attr]
            var_copy = np.log1p(var_copy)
            if(pd.isna(var_copy.skew())):
                print("doing nothing")
            else:
                inputDF.loc[:,attr] = inputDF.loc[:,attr].map(lambda x: np.log1p(x + -(min(inputDF.loc[:,attr])) + 1) 
                                                              if min(inputDF.loc[:,attr]) < 0 else np.log1p(x))
                #inputDF.loc[:,attr] = np.log1p(inputDF.loc[:,attr])
        elif(inputDF.loc[:,attr].skew()<-10):
            var_copy = inputDF.loc[:,attr]
            var_copy = np.log(var_copy)
            if(pd.isna(var_copy.skew())):
                print("doing nothing")
            else:
                inputDF.loc[:,attr] = inputDF.loc[:,attr].map(lambda x: np.log(x + -(min(inputDF.loc[:,attr])) + 1) 
                                                              if min(inputDF.loc[:,attr]) < 0 else np.log(x))
                #inputDF.loc[:,attr] = np.log(inputDF.loc[:,attr])
        print(attr, " ", inputDF.loc[:,attr].skew()) 
    
    
    
    
    
    #One-Hot encoding the Sectors Attribute using get_dummies()
    # inputDF = pd.get_dummies(inputDF, columns = ['Sector'])
    # print(inputDF.iloc[:,120:]) #At 210: , Itll print empty dataframe as dimensionality has been reduced to less than 150 cols
    # #Update input column names
    # print("printing inputDF.columns: ", inputDF.columns)
    # inputCols = list(inputDF.columns)
    # print("printing inputCols", inputCols)
    #print(inputDF.std())
    
    #Advanced Preprocessing (5)
    #Hash Encoding
    
    import category_encoders as ce
    hash_enc = ce.HashingEncoder(cols='Sector', n_components=8)
    hash_enc_data = hash_enc.fit_transform(inputDF)
    inputDF = hash_enc_data
    print(inputDF.iloc[:,120:])
    print("printing inputDF.columns: ", inputDF.columns)
    inputCols = list(inputDF.columns)
    print("printing inputCols", inputCols)
    #print("Hashed data: ", hash_enc_data)
    
    #Advanced Preprocessing (6): Isolation Forest (After log transformation)
    # iforest = IsolationForest(n_estimators = 100).fit(inputDF)

    # scores = iforest.score_samples(inputDF)
    # print("Printing scores for isolation forest: ")
    # print(scores)
    # #Converting scores into a dataframe and exporting it to excel
    # sampleDF = pd.Series(scores)
    # #sampleDF.to_excel(r'C:\Users\abdul\OneDrive - DePauw University\Desktop\Desktop\School\Year 4\Senior Project\afterMath\test2.xlsx', index = False)
    # print(sampleDF)
    
    # #Removing datapoints with score less than -0.5
    # print("Start examining HERE")
    # print(sampleDF.shape)
    # print(inputDF.shape)
    # inputDF = inputDF.loc[sampleDF>-0.45,]
    # inputDF = inputDF.reset_index(drop=True)
    # print(inputDF.shape)
    
    # #Updating output Series and resetting its index
    # outputSeries = outputSeries.loc[sampleDF>-0.45]
    # outputSeries = outputSeries.reset_index(drop=True)
    # print("output series shape is ", outputSeries.shape)
    
    #Finding attribute with 0 standard deviation
    # for attr in numCols:
    #     if(inputDF.loc[:,attr].std()==0):
    #         print(attr)
    ##Update input column names
    # inputCols = list(inputDF.columns)
    # print(inputCols)
    
    #Delete
    #Data Regularization: 
    #We go with Normalization Firstly, then compare with Standardization
    #First formula for normalization will make range [0,1]
    #Formula1 = (x-xmin)/(xmax-xmin)
    # for attr in inputCols:
    #     min = inputDF.loc[:,attr].min()
    #     max = inputDF.loc[:,attr].max()
    #     inputDF.loc[:,attr] = inputDF.loc[:,attr].map(lambda x: x if max-min==0 else (x-min)/(max-min))
        
    #normalize(inputDF,inputCols)
    #normalize2(inputDF,inputCols)
    standardize(inputDF,inputCols)
    print(inputDF)
    return(inputDF, inputCols, outputSeries, uInputDF)
    
    
def buildAndTestModel(df, inputCols, outputCol):
    #For now, we will input missing values based on the whole of the dataset, which will cause data leakage during our 
    #K Folds Cross Validation process
    
    inputDF, inputCols, outputSeries, uInputDF = preprocess(df, inputCols, outputCol)
    print("printing inputCols AGAIN: ", inputCols)
    print("Checking output Series: ", outputSeries)    
    print("length of inputDF: ", len(inputDF.iloc[1,:]))
    print("length of inputCols: ", len(inputCols))
    #gridSearchGradBoost(inputDF, outputSeries)
    
    
    #Building model, will DELETE later
    model = LogisticRegression()
    # cvScores = model_selection.cross_val_score(model,
    #                                            inputDF, outputSeries, cv=3) #exclude: scorer=alg.scorer
    # print("Preprocessed Score",cvScores.mean())
    
    # randomForestModel = RandomForestClassifier()
    # #evaluating randomForest Model
    # cv = RepeatedKFold(n_splits=20, n_repeats=3, random_state=1)
    # n_scores = cross_val_score(randomForestModel, inputDF, outputSeries, scoring='accuracy', cv=cv, n_jobs=-1)
    # # report performance
    # print('mean is %.3f' % (n_scores.mean()))
    # sys.exit()
    accuracies1 = []
    accuracies2 = []
    #Accuracy testing for different cv values 
    cvValues = [2,3,4,5,10,12,14,16,18,20,24,28,32,40,50,60,70,80]
    for n in cvValues:
        cv = RepeatedKFold(n_splits=n, n_repeats=3, random_state=1)
        # evaluate the model on the dataset
        n_scores = cross_val_score(model, inputDF, outputSeries, scoring='accuracy', cv=n, n_jobs=-1)
        # report performance
        accuracies1.append(mean(n_scores))
        print('For cv= %.3f, mean is %.3f' % (n, mean(n_scores)))
    cvValues = [2,3,4,5,10,12,14,16,18,20,24,28,32,40,50,60,70,80]
    for n in cvValues:
        cv = RepeatedKFold(n_splits=n, n_repeats=3, random_state=1)
        # evaluate the model on the dataset
        n_scores = cross_val_score(model, uInputDF, outputSeries, scoring='accuracy', cv=n, n_jobs=-1)
        # report performance
        accuracies2.append(mean(n_scores))
        print('For cv= %.3f, mean is %.3f' % (n, mean(n_scores)))
    plt.plot(range(0,18), accuracies1, label = "Processed Score")
    plt.plot(range(0,18), accuracies2, label = "Unprocessed Score")
    labels = [2,3,4,5,10,12,14,16,18,20,24,28,32,40,50,60,70,80]
    plt.xticks(range(0,18), labels, rotation='horizontal')
    plt.show()
    sys.exit()
    
    #'solver=liblinear' gives us cross val score of 0.5861382575405025
    #Trying sag and saga algorithms
    #Sag (requiring max_iter to be 500) scored 0.5874516276481988
    #Saga (also requiring max_iter to be 500) scored 0.5866636055835811
    #Both higher than liblinear but not significantly at all
    #No Scorer parameter shows the same score as sag
    
    #Test model with unprocessed dataset
    # model2 = LogisticRegression(solver='saga')
    # cvScores2 = model_selection.cross_val_score(model2,
    #                                            uInputDF, outputSeries, cv=3) #exclude: scorer=alg.scorer
    # print("Unpreprocessed Score",cvScores2.mean())
    #Scores for Unprocessed DataSet
    #No scorer parameter gives us a score of 0.5477950894436068
    #'solver=liblinear' gives us cross val score of 0.5672302782483158
    # Sag scored 0.5181089952408431
    # Saga scored 0.5186347569437981
    #plotAccuracies(df, inputDF, outputSeries)
    #plotAccuracies(df, uInputDF, outputSeries)
    
    # #Correlation Matrix (before multicollinearity test)
    # inputDF, inputCols = corrMatrix(inputDF, inputCols)
    # print('After correlation selection: \n',inputDF)
    
    #Hyperparameter testing with different values of K in select K Best
    hyperparameterTesting(inputCols,inputDF,outputSeries)
    sys.exit()
    #Trying out different values for K
    #We stick with select60Best Features (turns out select60best features is best with log transformation, 120 otherwise)
    
    
    # select = SelectKBest(score_func=f_classif, k=130)
    # z = select.fit_transform(inputDF,outputSeries)
 
    # print("After selecting best 130 features:", z.shape) 
 
    # #After selecting best 3 (60) features: (150, 3) 
    # filter = select.get_support()
  
    # #Selectin 60 best predictors
    # test = SelectKBest(score_func=f_classif, k=130)
    # fit = test.fit_transform(inputDF, outputSeries)
    # print(fit)
    
    
    # filter = test.get_support()
    # #features = array(inputCols)
    
    # print(filter)
    # print(len(filter))
    # print(len(inputCols))
    
    # inputCols = pd.Series(inputCols)
    # inputCols = inputCols.loc[filter]
    # #print(fit.transform(inputDF)) DELETE
    
    # #inputCols = inputCols[filter]
    # print(inputCols)
    # print("Before Selection: \n",inputDF)
    # inputDF = inputDF.loc[:,inputCols]
    # print("After Selection: \n",inputDF)
    # print("Attributes with missing values:", getAttrsWithMissingValues(inputDF), sep='\n')
    
    # #Correlation Matrix (after multicollinearity test)
    # #inputCols in corrMatrix must be List
    # inputCols = inputCols.tolist()# DELETE if doing multicollinearity BEFORE selectKBest
    # inputDF, inputCols = corrMatrix(inputDF, inputCols)
    # print('After correlation selection: \n',inputDF)
    
    # gridSearchRandomForest(inputDF, outputSeries)
    # sys.exit()
    #randomForests(inputDF, outputSeries)
    #Plot New CorrMatrix
    #newCorrMatrix = inputDF.corr()
    #sn.heatmap(newCorrMatrix, annot=True)
    #plt.show()
    
    #print("Start looking here \n")
    #plotAccuracies(df, inputDF, outputSeries)
    #hyperparameterTesting(inputCols,inputDF,outputSeries)
    
    # create data
    # x = range(0,20)#[10,20,30,40,50]
    # y1 = [0.549632,0.504712,0.529674,0.523100,0.519956,0.509191,0.543067,0.519985,0.546785,
    #      0.528887,0.536502,0.535494,0.533659,0.539154,0.541492,0.542542,0.542569,0.541729,
    #      0.540298,0.540896]#[30,30,30,30,30]
    # y2 = [0.546744,0.559610,0.520221,0.498695,0.599516,0.447742,0.558298,0.587687,0.618449,
    #      0.614496,0.616334,0.617461,0.619273,0.617319,0.619223,0.619223,0.623999,0.627163,
    #      0.625343,0.624928]
    # y3 = [0.514706,0.587714,0.484244,0.479773,0.543854,0.351891,0.522584,0.529146,0.562736,
    #      0.580357,0.581408,0.569042,0.577440,0.564288,0.571954,0.576155,0.577930,0.579258,
    #      0.582172,0.588281]
    
  
    # plot lines
    # plt.plot(x, y1, label = "Unprocessed")
    # plt.plot(x, y3, label = "Preprocessed")
    # plt.legend()
    # labels = [2,3, 4, 5, 6,7,8, 10, 12, 14,16,18, 20, 24, 28,32,40, 50, 60, 80]
    # plt.xticks(x, labels, rotation='horizontal')
    # plt.show()
    
    #50 Fold Cross Validation test result
    #print(kFoldCVBuiltIn4(50, inputDF, outputSeries))
    #plotAccuracies(df, inputDF, outputSeries)
    # alg = kNNClassifier(k=1)
    # cvScores1 = model_selection.cross_val_score(alg,
    #                                             inputDF, outputSeries, cv=50, scoring=alg.scorer)
    # result1 = sum(cvScores1)/len(cvScores1)
    # print("Selected 60 Best dataset, 1NN, accuracy: ", result1)
    
    # alg = kNNClassifier(k=5)
    # cvScores1 = model_selection.cross_val_score(alg,
    #                                             inputDF, outputSeries, cv=50, scoring=alg.scorer)
    # result1 = sum(cvScores1)/len(cvScores1)
    # print("Selected 60 Best dataset, 5NN, accuracy: ", result1)
    
    # alg = kNNClassifier(k=10)
    # cvScores1 = model_selection.cross_val_score(alg,
    #                                             inputDF, outputSeries, cv=50, scoring=alg.scorer)
    # result1 = sum(cvScores1)/len(cvScores1)
    # print("Selected 60 Best dataset, 10NN, accuracy: ", result1)
    
    # alg = kNNClassifier(k=20)
    # cvScores1 = model_selection.cross_val_score(alg,
    #                                             inputDF, outputSeries, cv=50, scoring=alg.scorer)
    # result1 = sum(cvScores1)/len(cvScores1)
    # print("Selected 60 Best dataset, 20NN, accuracy: ", result1)
    
    # alg = kNNClassifier(k=50)
    # cvScores1 = model_selection.cross_val_score(alg,
    #                                             inputDF, outputSeries, cv=50, scoring=alg.scorer)
    # result1 = sum(cvScores1)/len(cvScores1)
    # print("Selected 60 Best dataset, 50NN, accuracy: ", result1)
    
    
    # alg = kNNClassifier(k=30)
    # cvScores1 = model_selection.cross_val_score(alg,
    #                                             inputDF, outputSeries, cv=50, scoring=alg.scorer)
    # result1 = sum(cvScores1)/len(cvScores1)
    # print("Selected 60 Best dataset, 30NN, accuracy: ", result1)
    
    # alg = kNNClassifier(k=70)
    # cvScores1 = model_selection.cross_val_score(alg,
    #                                             inputDF, outputSeries, cv=50, scoring=alg.scorer)
    # result1 = sum(cvScores1)/len(cvScores1)
    # print("Selected 60 Best dataset, 70NN, accuracy: ", result1)
    
    # alg = kNNClassifier(k=100)
    # cvScores1 = model_selection.cross_val_score(alg,
    #                                             inputDF, outputSeries, cv=50, scoring=alg.scorer)
    # result1 = sum(cvScores1)/len(cvScores1)
    # print("Selected 60 Best dataset, 100NN, accuracy: ", result1)
    
    # alg = kNNClassifier(k=150)
    # cvScores1 = model_selection.cross_val_score(alg,
    #                                             inputDF, outputSeries, cv=50, scoring=alg.scorer)
    # result1 = sum(cvScores1)/len(cvScores1)
    # print("Selected 60 Best dataset, 150NN, accuracy: ", result1)
    
    
    #Gradient Boosting Algorithm
    #gradBoostAlg(inputDF, outputSeries) #number of estimators hyperparameter testing
    #gradBoostAlg2(inputDF, outputSeries)# subsample hyperparameter testing
    #gradBoostAlg3(inputDF, outputSeries)# learning rate hyperparameter testing
    #gradBoostAlg4(inputDF, outputSeries)# tree depth hyperparameter testing
    #XGBoostAlg(inputDF, outputSeries)# XGBoost Classification
    #lightGBM(inputDF, outputSeries)# lightGBM Classification
    
    
    
    
    
    
    
def normalize(inputDF,inputCols):
    #Data Regularization: 
    #We go with Normalization Firstly, then compare with Standardization
    #First formula for normalization will make range [0,1]
    #Formula1 = (x-xmin)/(xmax-xmin)
    for attr in inputCols:
        min = inputDF.loc[:,attr].min()
        max = inputDF.loc[:,attr].max()
        inputDF.loc[:,attr] = inputDF.loc[:,attr].map(lambda x: x if max-min==0 else (x-min)/(max-min))
    #return(inputDF)
    
def normalize2(inputDF,inputCols):
    #Second formula for normalization will make range [-1,1]
    #Formula2 = (x-xavg)/(xmax-xmin)
    for attr in inputCols:
        min = inputDF.loc[:,attr].min()
        max = inputDF.loc[:,attr].max()
        avg = inputDF.loc[:,attr].mean()
        inputDF.loc[:,attr] = inputDF.loc[:,attr].map(lambda x: x if max-min==0 else (x-avg)/(max-min))
    #return(inputDF)
    
def standardize(inputDF, inputCols):
    #Std formula = (x-xavg)/x.std 
    for attr in inputCols:
        print("inputCols not empty afterall")
        avg = inputDF.loc[:,attr].mean()
        if(inputDF.loc[:,attr].std()!=0):
            inputDF.loc[:,attr] = inputDF.loc[:,attr].map(lambda x: (x-avg)/(inputDF.loc[:,attr].std()))
        
    
        


def plotAccuracies(df,inputDF,outputSeries):
    KFoldList = pd.Series([2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 18, 20, 24, 28, 32, 40, 50, 60, 80])
   
    norm_inputDF = inputDF
    
    
    accuracies = KFoldList.map(lambda x: kFoldCVBuiltIn4(x, norm_inputDF, outputSeries))
    
    print(accuracies)
    
    # df, inputCols, outputCol = readData()
    
    #inputDF = df.loc[:, inputCols]
    #outputSeries = df.loc[:, outputCol]
    
    
    
    #normalize(norm_df, inputCols)
    #standardize(stan_df, outputCol)
    #Will you use norminputDF and ...?
    #norm_inputDF = norm_df.loc[:, inputCols]
    #norm_outputSeries = norm_df.loc[:, outputCol]
    
    
    # plt.plot(neighborList, accuracies)
    # plt.xlabel('Neighbors')
    # plt.ylabel('Accuracy')
    
    # print(neighborList.loc[accuracies.idxmax()])
    
#Correlation Selection: using multicollinearity as a guide for feature selection 
def corrMatrix(inputDF, inputCols):
    corrMatrix = inputDF.corr()
    #corrMatrix.to_excel("correlationMatrix.xlsx")
    print(corrMatrix)
    upper_tri = corrMatrix.where(np.triu(np.ones(corrMatrix.shape),k=1).astype(np.bool))
    print(upper_tri)
    to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > 0.8)]
    print(); print(to_drop)
    print(len(inputCols))
    
    for i in to_drop:
        inputCols.remove(i)
    
    #inputCols.remove(to_drop[:]) DELETE
    
    
    print(len(inputCols))
    inputDF = inputDF.drop(inputDF.loc[:,to_drop],axis=1)
    print(); print(inputDF)
    
    return(inputDF, inputCols)
    #sn.heatmap(corrMatrix, annot=True)
    #plt.show()
    
#Testing different values of K in SeclectKBest using 50 Folds Cross Validation (4)
def hyperparameterTesting(inputCols,inputDF,outputSeries):
    KList = [ 3,4,5,6,7,8,10,12,14,16,18,20,24,28,32,40, 50, 60, 80, 100, 120, 130, 140, 150, 160,170,180,190] 
   
    accuracies = []
    norm_inputDF = inputDF
    
    #Trial to fix error: Looks like inputDF is changed after
    #first iteration, must make it new for each iteration in the for loop
    for i in KList:
        temp_inputCols = inputCols
        temp_inputDF = inputDF.copy()
        temp_outputSeries = outputSeries
        select = SelectKBest(score_func=f_classif, k=i)
        z = select.fit_transform(temp_inputDF,temp_outputSeries)
     
        print("After selecting best 50 features:", z.shape) 
     
        #After selecting best 3 (50) features: (150, 3) 
        filter = select.get_support()
      
        #Selectin 50 best predictors
        test = SelectKBest(score_func=f_classif, k=i)
        fit = test.fit_transform(temp_inputDF, temp_outputSeries)
        print(fit)
        
        
        filter = test.get_support()
        #features = array(inputCols)
        
        print(filter)
        print(len(filter))
        print(len(temp_inputCols))
        
        temp_inputCols = pd.Series(temp_inputCols)
        temp_inputCols = temp_inputCols.loc[filter]
        #print(fit.transform(inputDF)) DELETE
        
        #inputCols = inputCols[filter]
        print(temp_inputCols)
        print("Before Selection: \n",temp_inputDF)
        temp_inputDF = temp_inputDF.loc[:,temp_inputCols]
        print("After Selection: \n",temp_inputDF)
        
        #Correlation Matrix (after multicollinearity test)
        #inputCols in corrMatrix must be List
        print('Length before correlation testing (expecting ',i,'): ',len(temp_inputCols))
        temp_inputCols = temp_inputCols.tolist()# DELETE if doing multicollinearity BEFORE selectKBest
        temp_inputDF, temp_inputCols = corrMatrix(temp_inputDF, temp_inputCols)
        print('Length after correlation testing (not expecting ',i,'): ',len(temp_inputCols))
        print('After correlation selection: \n',temp_inputDF)
        
        #Plot New CorrMatrix
        #newCorrMatrix = temp_inputDF.corr()
        #sn.heatmap(newCorrMatrix, annot=True)
        #plt.show()
        iFeatureAccuracy = kFoldCVBuiltIn4(20, temp_inputDF, temp_outputSeries)
        accuracies.append(iFeatureAccuracy)
        
        
    
    
    
    
    print("Here are the accuracies for different hypwerParameter values for the feature selection method selectKBest: \n")
    print(accuracies)
    
    # plot lines
    plt.plot(range(0,28), accuracies, label = "K-Cross Validation Score")
    labels = [ 3,4,5,6,7,8,10,12,14,16,18,20,24,28,32, 40, 50, 60, 80, 100, 120, 130, 140, 150, 160, 170, 180, 190]
    plt.xticks(range(0,28), labels, rotation='horizontal')
    plt.show()
    
    
    
def removeMissing(inputCols,inputDF):
    dropList = []
    inputCols_copy = inputCols.copy()
    inputCols_copy.remove('Sector')
    for i in inputCols_copy:
        score = inputDF.loc[:,i].isnull().sum()/inputDF.loc[:,i].size
        #series.isnull().sum()/series.size
        if(score>=0.3):
            dropList.append(i)
        #series.isnull().sum()/series.size
    print("Droplist length is ", len(dropList))
    print("current inputCols length is ", len(inputCols))
    inputCols = [x for x in inputCols if x not in dropList]
    print("updated inputCols length is ", len(inputCols))
    #Drop columns from inputDF
    inputDF = inputDF.drop(columns=dropList)
    if ('Sector' in inputCols):
        print ("Element Exists")
    
    
    return inputCols, inputDF
        
        
def removeLowVar(inputCols,inputDF):
    dropList = []
    inputCols_copy = inputCols.copy()
    inputCols_copy.remove('Sector')
    print(inputCols)
    print(inputCols_copy)
    for i in inputCols_copy:
        if(inputDF.loc[:,i].var()<0.1 and inputDF.loc[:,i].var()>-0.1):
            dropList.append(i)
        #series.isnull().sum()/series.size
    print("Droplist length is ", len(dropList))
    print("current inputCols length is ", len(inputCols))
    inputCols = [x for x in inputCols if x not in dropList]
    print("updated inputCols length is ", len(inputCols))
    #Drop columns from inputDF
    inputDF = inputDF.drop(columns=dropList)
    
    return inputCols, inputDF

def removeLowCorrWTV(inputCols, inputDF, outputSeries):
    dropList = []
    inputCols_copy = inputCols.copy()
    inputCols_copy.remove('Sector')
    inputDF_copy = inputDF.loc[:,inputCols_copy]
    corrSeries = inputDF_copy.apply(lambda x: x.corr(outputSeries))
    #print(corrSeries)
    #print(max(corrSeries))
    #print(min(corrSeries))
    for i in corrSeries.index:
        if(corrSeries[i]>-0.1 and corrSeries[i]<0.1):
            dropList.append(i)
            print(i)
    
    print("Droplist length is ", len(dropList))
    print("current inputCols length is ", len(inputCols))
    inputCols = [x for x in inputCols if x not in dropList]
    print("updated inputCols length is ", len(inputCols))
    #Drop columns from inputDF
    inputDF = inputDF.drop(columns=dropList)
    
    return inputCols, inputDF, outputSeries

#KNNClassifier
def findKNearestHOF(df, testRow, k):
    KList = list(range(0,k))
    nearestKseriesIdx = pd.Series(KList)
    distSeries = df.apply(lambda row: sp.distance.euclidean(testRow, row),axis=1)
    for i in range(k):
        nearestKseriesIdx.iloc[i] = distSeries.idxmin()
        distSeries = distSeries.drop([distSeries.idxmin()])

    return nearestKseriesIdx

def findNearestHOF(df, testRow):
    distSeries = df.apply(lambda row: sp.distance.euclidean(testRow, row),axis=1)
    return distSeries.idxmin()

def accuracyOfActualVsPredicted(actualOutputSeries, predOutputSeries):
        compare = (actualOutputSeries == predOutputSeries).value_counts()
        
        # actualOutputSeries == predOutputSeries makes a Series of Boolean values.
        # So in this case, value_counts() makes a Series with just two elements:
        # - with index "False" is the number of times False appears in the Series
        # - with index "True" is the number of times True appears in the Series
    
        # print("compare:", compare, type(compare), sep='\n', end='\n\n')
        
        # if there are no Trues in compare, then compare[True] throws an error. So we have to check:
        if (True in compare):
            accuracy = compare[True] / actualOutputSeries.size
        else:
            accuracy = 0
        
        return accuracy
    
class kNNClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, k=1):
        self.k = k
        self.inputsDF = None
        self.outputSeries = None
        self.scorer = make_scorer(accuracyOfActualVsPredicted,
                                  greater_is_better=True)
    def fit(self, inputsDF, outputSeries):
        self.inputsDF = inputsDF
        self.outputSeries = outputSeries
        return self
    
    def predict(self, testInput):
        if isinstance(testInput, pd.core.series.Series):
            if(self.k>1):
                #return self.outputSeries.loc[findKNearestHOF(self.inputsDF,testInput, self.k).mode().iloc[0]]#self.outputSeries.loc[findNearestHOF(self.inputsDF, testInput)]
                return self.outputSeries.loc[findKNearestHOF(self.inputsDF,testInput, self.k)].mode().iloc[0]
            else:
                return self.outputSeries.loc[findNearestHOF(self.inputsDF, testInput)]
        else:
            #return testInput.apply(lambda row: self.outputSeries.loc[findKNearestHOF(self.inputsDF,row, self.k).mode().iloc[0]] if self.k>1 else self.outputSeries.loc[findNearestHOF(self.inputsDF, row)], axis = 1)
            return testInput.apply(lambda row: self.outputSeries.loc[findKNearestHOF(self.inputsDF,row, self.k)].mode().iloc[0] 
                                   if self.k>1 else self.outputSeries.loc[findNearestHOF(self.inputsDF, row)], axis = 1)
            #testInput is a DataFrame, so predict for every row in it
            #testInput.apply(lambda row: self.outputSeries.loc[findNearestHOF(self.inputsDF, row)], axis=1)
            #old: self.outputSeries.loc[testInput.apply(lambda row: self.outputSeries.loc[findNearestHOF(self.inputsDF, row), axis=1)]
            
def kFoldCVBuiltIn(i, inputDF, outputSeries):
    alg = kNNClassifier(k=i)
    cvScores = model_selection.cross_val_score(alg,
                                               inputDF, outputSeries, cv=10, scoring=alg.scorer)
    return sum(cvScores)/len(cvScores)
    

#Gradient Boosting
#X: InputDF, Y: outputSeries

def gradBoostAlg(inputDF, outputSeries):
    #HyperParameter Testing with different number of Estimators
    estNumList = [5,10,20,30,40,50,70,100,200,500,1000]
    for n in estNumList:
        # define the model
        model = GradientBoostingClassifier(n_estimators=n)
        # define the evaluation method
        cv = RepeatedStratifiedKFold(n_splits=50, n_repeats=3, random_state=1)
        # evaluate the model on the dataset
        n_scores = cross_val_score(model, inputDF, outputSeries, scoring='accuracy', cv=cv, n_jobs=-1)
        # report performance
        print('For %.3f estimators, Mean Accuracy: %.3f (%.3f)' % (n, mean(n_scores), std(n_scores)))
    
#def LGBMAlg(inputDF, outputSeries):
    
def gradBoostAlg2(inputDF, outputSeries):#subsample hyperparameter stesting
    #HyperParameter Testing with different proportions of subsample of Estimators
    estNumList = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
    for n in estNumList:
        # define the model
        model = GradientBoostingClassifier(n_estimators=50,subsample=n)
        # define the evaluation method
        cv = RepeatedStratifiedKFold(n_splits=50, n_repeats=3, random_state=1)
        # evaluate the model on the dataset
        n_scores = cross_val_score(model, inputDF, outputSeries, scoring='accuracy', cv=cv, n_jobs=-1)
        # report performance
        print('For 50 estimators, and %.3f percent of subsample, Mean Accuracy: %.3f (%.3f)' % (n, mean(n_scores), std(n_scores)))
      
def gradBoostAlg3(inputDF, outputSeries):#learning rate hyperparameter stesting
    #HyperParameter Testing with different learning rate values
    estNumList = [ 0.05, 0.1,0.2,0.3,0.4,0.5, 1.0]
    for n in estNumList:
        # define the model
        model = GradientBoostingClassifier(n_estimators=50,subsample=0.5, learning_rate=n)
        # define the evaluation method
        cv = RepeatedStratifiedKFold(n_splits=50, n_repeats=3, random_state=1)
        # evaluate the model on the dataset
        n_scores = cross_val_score(model, inputDF, outputSeries, scoring='accuracy', cv=cv, n_jobs=-1)
        # report performance
        print('50 estimators, 0.5 of subsample, and %.3f learning rate, Mean Accuracy: %.3f (%.3f)' % (n, mean(n_scores), std(n_scores)))
  
def gradBoostAlg4(inputDF, outputSeries):#Tree Depth hyperparameter stesting
    #HyperParameter Testing with different maximum tree depth values
    estNumList = [ 1,2,3,4,5,6,7,8,9,10]
    for n in estNumList:
        # define the model
        model = GradientBoostingClassifier(n_estimators=50,subsample=0.5, learning_rate=0.05, max_depth=n)
        # define the evaluation method
        cv = RepeatedStratifiedKFold(n_splits=50, n_repeats=3, random_state=1)
        # evaluate the model on the dataset
        n_scores = cross_val_score(model, inputDF, outputSeries, scoring='accuracy', cv=cv, n_jobs=-1)
        # report performance
        print('50 estimators, 0.5 of subsample, 0.05 learning rate, and %.3f max tree depth, Mean Accuracy: %.3f (%.3f)' % (n, mean(n_scores), std(n_scores)))


def gridSearchGradBoost(inputDF, outputSeries):
    model=GradientBoostingClassifier() 
    cv = RepeatedKFold(n_splits=50, n_repeats=3, random_state=1)
    space = dict()
    #space['solver'] = ['svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga']
    space['loss'] = ['deviance', 'exponential']
    space['learning_rate'] = [ 0.05]#, 0.01]
    space['n_estimators'] = [50]#, 100, 500]
    space['subsample'] = [0.5]#, 0.7]
    space['max_depth'] = [2]#,3]#,7,9]
    
    search = GridSearchCV(model, space, scoring='accuracy', n_jobs=-1, cv=cv)
    
    result = search.fit(inputDF, outputSeries)
    
    print('Best Score: %s' % result.best_score_)
    print('Best Hyperparameters: %s' % result.best_params_)
    
def gridSearchRandomForest(inputDF, outputSeries):
    model=RandomForestClassifier() 
    cv = RepeatedKFold(n_splits=20, n_repeats=3, random_state=1)
    space = dict()
    #space['solver'] = ['svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga']
    space['n_estimators'] = [200, 250, 300]
    space['max_features'] = ['sqrt']
    space['max_samples'] = [0.1, 0.25, 0.5, 0.75]
    space['bootstrap'] = [True]
    
    
    search = GridSearchCV(model, space, scoring='accuracy', n_jobs=-1, cv=cv)
    
    result = search.fit(inputDF, outputSeries)
    
    print('Best Score: %s' % result.best_score_)
    print('Best Hyperparameters: %s' % result.best_params_)




def XGBoostAlg(inputDF, outputSeries):
    # #HyperParameter Testing with different number of Estimators
    # estNumList = [5,10,20,30,40,50,70,100,200,500,1000]
    # for n in estNumList:
    # define the model
    model = XGBClassifier()
    # define the evaluation method
    cv = RepeatedStratifiedKFold(n_splits=50, n_repeats=3, random_state=1)
    # evaluate the model on the dataset
    n_scores = cross_val_score(model, inputDF, outputSeries, scoring='accuracy', cv=cv, n_jobs=-1)
    # report performance
    print('Mean Accuracy: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))
    
    
def lightGBM(inputDF, outputSeries):
    # #HyperParameter Testing with different number of Estimators
    # estNumList = [5,10,20,30,40,50,70,100,200,500,1000]
    # for n in estNumList:
    # define the model
    model = LGBMClassifier()
    # define the evaluation method
    cv = RepeatedStratifiedKFold(n_splits=50, n_repeats=3, random_state=1)
    # evaluate the model on the dataset
    n_scores = cross_val_score(model, inputDF, outputSeries, scoring='accuracy', cv=cv, n_jobs=-1)
    # report performance
    print('Mean Accuracy: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))
    
    
def randomForests(inputDF, outputSeries):
    # #HyperParameter Testing with different number of Estimators
    estNumList = [5,10,20,30,40,50,70,100,200,500,1000]
    for n in estNumList:
    #define the model
        model = RandomForestClassifier(n_estimators=n)
        # define the evaluation method
        cv = RepeatedStratifiedKFold(n_splits=50, n_repeats=3, random_state=1)
        # evaluate the model on the dataset
        n_scores = cross_val_score(model, inputDF, outputSeries, scoring='accuracy', cv=cv, n_jobs=-1)
        # report performance
        print('Mean Accuracy: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))
    
    
if __name__ == "__main__":
    main()
    
    