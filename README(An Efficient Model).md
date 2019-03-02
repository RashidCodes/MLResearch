In our last post, we built a Logistic Regression Model with our data. Our model didn't do well on our data but as amateur ML engineers, we want that to happen so we'll have to make a few changes. Afterall, what did Einstein say :).

Firstly, let's review some of the things that we did wrongly and some stuff that could have also been done.

## DEDUCTIONS FROM OUR PREVIOUS RESEARCH.
- With k-fold cross validation, we only achieved an accuracy of 80% on our test set.
- With the learning curve, we were able to tell that adding more training samples won't do us any good.
- With the learning curve, we were also able to tell that our model totally underfits our data.
- With the validation curve, we were also able to tell that our model didn't do too well with variations in regularization strength.
- We noticed that there was large imbalance in our classes with about an 80:20 ratio.


Looks like we have a few issues we might have to address. There are some assumptions we have to make before using a logistic regression classifier.

## SOME ASSUMPTIONS ABOUT LOGISTICREGRESSION 
- Logisitic regression does not require a linear relationship between the dependent and independent variable.
- Homoscedasticity is not required unlike in linear regression.
- The dependent variable is not measured on an interval.
- Observations must be independent of each other, in other words, observations should not come from repeated measurements or     matched data.
- Logistic regression assumes linearity of independent variables and log odds.
- The dependent variable should be dichotomous in nature.
- It works well when there are no outliers in your data.
- It also works well when there is no multicollinearity in our predictors. 
- The dataset must be linearly seperable.

Previously, we did not consider some of the assumptions aforementioned. So we'll start to make a better logistic regression model by following some of the assumptions. We'll try to tackle each assumptions subsequently.

## TACKLING ASSUMPTIONS
- Our data shows that there isn't a **linear relationship** between the dependent and independent variable. We can prove this easily using matplotlib. 

- **Homoscedasticity** won't be a necessity in the absence of linearity. The dependent variable was not measured on an **interval or in any sequential order** and also, the all **observations** are **independent** of each other.

- The **dependent** variable is dichotomous in nature so that's good.

- In the preparation of data for our last model, we did not check for **Outliers** in our data. In a post from ***statisticssolutions.com***, this problem can be handled by removing values below -3.29 or above 3.29. We will try this method and see how the model does.

- We also did not check for **multicollinearity** in our data. For any pair of highly correlated features, we'll drop one, and we'll assess how model performance for our final dataset.

- We will also grab two features for visualization purposes - checking for **linear seperability** between these features.

## CLASS IMBALANCE
From our previous research, we noticed that there is a large imbalance in our classes. We will try a selected number of methods on our dataset and evaluate our model's performance accordingly.

The first assumption we're going to tackle is ***linear seperability***. We'll take take 2 features - Age, Tenure (no specific criteria) and plot these observations. 

```html
>>> import pandas as pd 
>>> import numpy as np 
>>> from matplotlib import pyplot as plt

## we'll first grab our data
>>> try:
...    data = pd.read_csv('Churn_Modelling1.csv')
... except:
...   print('The data was not found')

## grabbing 2 features
>>> X = data[['Age', 'Tenure']]
>>> X = np.array(X)

# targets
>>> y = data['Exited']

## all samples where Exited == 0
>>> X_0 = X[y == 0, :]

## all samples where Exited == 1
>>> X_1 = X[y == 1, :]

## We can now plot the data (50 samples)
>>> plt.scatter(X_0[:50, 0], X_0[:50, 1], color='blue', marker='x', label='Samples with class 0')
>>> plt.scatter(X_1[:50, 0], X_1[:50, 1], color='red', marker='o', label='Samples with class 1')
>>> plt.xlabel('Age')
>>> plt.ylabel('Tenure')
>>> plt.legend(loc='upper left')
>>> plt.show()
```

Our plot shows that the 2 features are not ***linearly seperable***. 

Next let's explore the assumption that there is no linear relationship between any of the features and the dependent variable. We'll randomly select any feature and plot it against the target.

```html
>>> try:
...    data = pd.read_csv('Churn_Modelling.csv')
... except:
...    print('The data was not found')

>>> X = data['CreditScore']
>>> y = data['Exited']

>>> plt.scatter(np.array(X)[:50], y[:50], color='green', marker='o', label='CreditScore vs Target')
>>> plt.xlabel('CreditScore')
>>> plt.ylabel('Exited')
>>> plt.legend(loc='center right')
>>> plt.show()
```

Let's investigate ***collinearity*** between features. We'll plot the correlation matrix to summarise the linear relationship between the features.

```html
>>> data = data.drop(['Exited'], axis=1)
>>> data.columns = ['CreditScore', 'Geography', 'Gender', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard'             ...                    'IsActiveMember', 'EstimatedSalary']
>>> cols = data.columns
>>> cm = np.corrcoef(data[cols].values.T)
>>> sns.set(font_scale=1.5)
>>> hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size' : 15}, yticklabels=cols,             ...      xticklabels=cols)
>>> plt.show()
```
The strength of the correlation between features is generally weak. Later, we'll use decision trees to assess feature importance.

We have to deal with outliers now, there are different ways of dealing with outliers, however we'll use the Tukey IQR.

We'll use this simple function to remove the outliers in our data. (Credit to **vishalkuo** on github)

```html
def removeOutliers(x, outlierConstant):
    a = np.array(x)
    upper_quartile = np.percentile(a, 75)
    lower_quartile = np.percentile(a, 25)
    IQR = (upper_quartile - lower_quartile) * outlierConstant
    quartileSet = (lower_quartile - IQR, upper_quartile + IQR)
    
    result = a[(a > quartileSet[0]) & (a < quartileSet[1])]
    
    return result.tolist()
```

## DEALING WITH CLASS IMBALANCE
Let's check the distribution of the classes in our dataset.
```html






