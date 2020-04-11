import pandas as pd #took kit for dataframes

titanic = pd.read_csv("titanic_train.csv")

titanic.head(5)

print(titanic.describe()) #.decribe= pandas's tool for calculating some statistical data like percentile, mean and std of the numerical values of the Series or DataFrame
#notice that count of ages < 891, meaning values are missing 

titanic["Age"]=titanic["Age"].fillna(titanic["Age"].median()) #.fillna fills in null values. here it filles with the median of age

print(titanic.describe()) #notice that age now has count of 891

print(titanic["Sex"].unique()) #list all the unique values of sex

titanic.loc[titanic["Sex"]=="male", "Sex"]=0 #use loc to set value for all items matching [male, sex]to be 0

titanic.loc[titanic["Sex"]=="female", "Sex"]=1 #use loc to set value for all items matching [female, sex] to be 1

print(titanic["Embarked"].unique())

#set null values of embarked to S, because S is the most frequent 
titanic["Embarked"]=titanic["Embarked"].fillna("S")

print(titanic["Embarked"].unique())


titanic.loc[titanic["Embarked"]=='S', "Embarked"]=0 #set value for all items matching [S, embarked] to be 0

titanic.loc[titanic["Embarked"]=="C", "Embarked"]=1 #set value for all items matching [C, embarked] to be 1

titanic.loc[titanic["Embarked"]=="Q", "Embarked"]=2 #set value for all items matching [Q, embarked] to be 2

from sklearn.linear_model import LogisticRegression #first to try with logistic regression 
from sklearn.model_selection import StratifiedKFold,cross_val_score 

predictors=["Pclass", "Sex","Age","SibSp", "Parch", "Fare", "Embarked"]

alg=LogisticRegression()

strKFold = StratifiedKFold(n_splits=3,shuffle=False,random_state=0) # use StratifiedKFold as the method of cross validation

scores = cross_val_score(alg,titanic[predictors],titanic["Survived"],cv=strKFold)

print("straitified cross validation scores:{}".format(scores)) #.format(values) put values into strings

print("Mean score of straitified cross validation:{:.2f}".format(scores.mean()))


titanic_test=pd.read_csv("test.csv")

titanic_test["Age"]=titanic_test["Age"].fillna(titanic_test["Age"].median()) #fill in age null with median of age

titanic_test["Fare"]=titanic_test["Fare"].fillna(titanic_test["Fare"].median()) #fill in fare null with median of age

titanic_test.loc[titanic_test["Sex"]=="male", "Sex"]=0 #set value for all males in SEX to be 0

titanic_test.loc[titanic_test["Sex"]=="female", "Sex"]=1 #set value for all males in SEX to be 1

titanic_test["Embarked"]=titanic_test["Embarked"].fillna("S")

titanic_test.loc[titanic_test["Embarked"]=="S", "Embarked"]=0
titanic_test.loc[titanic_test["Embarked"]=="C", "Embarked"]=1
titanic_test.loc[titanic_test["Embarked"]=="Q", "Embarked"]=2

from sklearn.model_selection import cross_validate
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold,cross_val_score 

predictors=["Pclass", "Sex","Age","SibSp", "Parch", "Fare", "Embarked"] #notice that predictors are not filtered. Pclass and Fare are atucatlly highly related.

alg=RandomForestClassifier(random_state=1, n_estimators=10, min_samples_split=2, min_samples_leaf=1) #roughly set the parameters of RandomForestClassifier
strKFold = StratifiedKFold(n_splits=3,shuffle=False,random_state=0)

scores=cross_val_score(alg, titanic[predictors],titanic["Survived"], cv=strKFold)

print(scores.mean()) #the result is not improved compared to the result of logisticregresion, meaning the parameters need to be adjusted 


alg=RandomForestClassifier(random_state=1, n_estimators=100, min_samples_split=4, min_samples_leaf=2) # increase #of estimator and limit the depth of the trees

strKFold = StratifiedKFold(n_splits=3,shuffle=False,random_state=0)

scores=cross_val_score(alg, titanic[predictors],titanic["Survived"], cv=strKFold)

print(scores.mean()) 

alg.fit(titanic[predictors], titanic["Survived"])
predictions = alg.predict(titanic_test[predictors])

result = pd.DataFrame({'PassengerId':titanic_test['PassengerId'].as_matrix(), 'Survived':predictions.astype(np.int32)})
result.to_csv("random_forest_predictions.csv", index=False)
print(pd.read_csv("random_forest_predictions.csv"))

#now make some new predictors 

titanic["FamilySize"]=titanic["SibSp"]+titanic["Parch"]  #generate a falimysize column

titanic["NameLength"]=titanic["Name"].apply(lambda x: len(x)) #generate a namelength column # The .apply method generates a new series of len(x) where x is the names

#now get the titles and count 
#re is useful method to extract contents from text
import re 

#first, define a function that extract title from x
def get_title(name):
    title_search= re.search('([A-Za-z]+)\.', name)
    if title_search:
        return title_search.group(1) #.group(1) resturns The first parenthesized subgroup.
    return ""

titles=titanic["Name"].apply(get_title) #apply the get_title function to "Name" colume

print(pd.value_counts(titles)) #check the frequency of each title

#generate a dictionary of title_mapping {"k":v}  where k= keys; v=values 
title_mapping = {"Mr":1, "Miss": 2, "Mrs": 3, "Master": 4, "Dr": 5, "Rev": 6, "Major": 7, "Col": 7, "Mlle": 8, "Mme": 8, "Don": 9, "Lady": 10, "Countess": 10, "Jonkheer": 10, "Sir": 9, "Capt": 7, "Ms": 2}
for k, v in title_mapping.items(): 
    titles[titles==k]=v
    
print(pd.value_counts(titles)) #check that all the titles are converted to values

titanic["Title"]=titles #add the converted titles as a column to the dataframe
    
import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif #selectkbest=Select features according to the k highest scores. f_classif=Compute the ANOVA F-value for the provided sample.
import matplotlib.pyplot as plt

#print(titanic.describe()) to verify the column names are correct
predictors=["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked", "FamilySize", "Title", "NameLength"]
selector=SelectKBest(f_classif, k=5)
selector.fit(titanic[predictors], titanic["Survived"]) #.fit(X,y)

print(selector)

#in SKlearn, .pvalues_ can get you p value
scores=-np.log10(selector.pvalues_) #0<=pvalues<=1, therefore log10(pvalues)<0, therefore need "-"

plt.bar(range(len(predictors)), scores)
plt.xticks(range(len(predictors)), predictors, rotation="vertical")
plt.show()

#pick the best predictors
predictors=["Pclass", "Sex", "Fare", "Title"]

alg=RandomForestClassifier(random_state=1, n_estimators=50, min_samples_split=8, min_samples_leaf=4)

alg.fit(titanic[predictors], titanic["Survived"])
predictions = alg.predict(titanic_test[predictors])
result = pd.DataFrame({'PassengerId':titanic_test['PassengerId'].as_matrix(), 'Survived':predictions.astype(np.int32)})
result.to_csv("random_forest_predictions.csv", index=False)
print(pd.read_csv("random_forest_predictions.csv"))
