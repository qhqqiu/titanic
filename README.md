# titanic
Codes for Kaggle titanic contest: https://www.kaggle.com/c/titanic

This was my first time completing the process of handling and manipulating data, implementing machine learning algorithms and making predictions.

For my own purpose of studying, I wanted to be absolutely clear about every line I wrote, thus I memoed a lot (maybe too much) throughout the codes.

I would like to roughly list out the steps I've taken here.

# Steps
1. Loading train data
* Why not loading in test data as well? Because at that time I was worried model training would steal insights from test data if I miss pre-processing. So I decided to deal with them seperately.

2. Pre-processing train data
* Filling in nulls with median values
* Encoding string vlues to numerics. (I simply assigned  0, 1, 2 to the strings. Back then I was not aware of encoding methods such as one-hoc)
* Selecting predictors from columns 

3. Implementing logistic regression on train data
* Why? The wanted prediction is to predict if a passenger on Titanic survived or not. Logistic regression fits classification problem well like this case.
* To improve the training accuracy, I used cross validation (by StratifiedKFold from SKlearn)
* Mean score of straitified cross validations was 0.80

3. Implemeting RandomForestClassifier on train data
* Why? Randomforest also works well on classification problems and I wanted to try tunning parameters such as n_estimators, min_samples_split, min_samples_leaf.
* First roughly set the parameters
* The result is not improved compared to the result of logisticregresion, meaning the parameters need to be adjusted 

4. Increase the number of estimator and limit the depth of the trees
* Te result improved and was even better than the result I got from logistic regression. Thus I decided to use this model.

5. Loading test data

6. Preprocessing test data 
* same as step 2

7. Implementing model from step 4 on test data
8. Loading prediction results to a csv file
