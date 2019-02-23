# DATA ANALYSIS
I run some of this code in the terminal so you may not find my script complete


## LIST OF FEATURES
```html
>>> import pandas as pd
>>> data = pd.read_csv('Churn_Modelling.csv')
>>> data.columns
```

- RowNumber 
- CustomerId
- Surname
- CreditScore
- Geography
- Gender
- Age
- Tenure
- Balance
- NumOfProducts
- HasCrCard
- IsActiveMember
- EstimateSalary
- Exited


## IMPORTANT FEATURES FROM MY PERSPECTIVE
- CreditScore
- Geography
- Gender
- Age
- Tenure
- Balance
- NumOfProducts
- HasCrCard
- IsActiveMember
- EstimateSalary

So let's drop a few features (vertical)
```html
>>> data.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1)
```

## TARGET
- Exited


## MISSING VALUES
```html
>>> data.isnull().sum()
```
As we can see, there are no missing values. Awesome!!


## HANDLING CATEGORICAL DATA
As we can see, we have some categorical data(nominal) under a few features, typically the Gender and Geography fields. Let's do something about it.

Firstly, let's grab all the unique values in the Geography.
```html
>>> import numpy as np
>>> unique_geography = np.unique(data['Geography'].values)
```

Next, let's assign each string a label

```html
>>> geography_dict = {val : idx for idx, val in enumerate(unique_geography)}
>>> geography_dict
{'France' : 0, 'Germany' : 1, 'Spain' : 2}

>>> data['Geography'] = data['Geography'].map(dict)
>>> data['Geography'].head()
0 0 
1 2
2 0
3 0
4 2
Name: Gender, dtype: int64
```

Let's do the same thing for the gender feature 

```html
>>> gender_dict = {val:idx for idx, val in enumerate(np.unique(data['Gender'].values))}
>>> data['Gender'] = data['Gender'].map(gender_dict)
9995 1
9996 1
9997 0
9998 1
9999 0
Name: Gender, dtype: int64
```

### So we're cool with the data now, let's see what the data looks like.
Let's see what the distribution of the **Age** looks like.
``` html
>>> from matplotlib import pyplot as plt
>>> ages = data['Age'].values
>>> plt.hist(ages, bins=7, histtype='step', align='mid')
>>> plt.xlabel('Ages')
>>> plt.ylabel('Number of Peeps')
>>> plt.title('Distribution of Ages')
>>> plt.show()
```

### We can see that there are a lot of people between the ages of 30 - 40.

### hmm, just a thought, I want to know the number of people above 40 that have credit cards.
``` html
>>> peopleAbove40 = hasCrCard[ages > 40] # grab all the data of the people above 40
>>> # let's now grab all the peeps with credit cards
...
>>> peopleAbove40 = len(peopleAbove40[peopleAbove40 == 1])
```

There, all done.

I also wanna know if there's some kinda relationship between a customer's age and his salary.
```html
>>> ages = data['Age'].values
>>> estimated_salary = data['EstimatedSalary'].values
>>> plt.scatter(ages, estimated salary)
>>> plt.xlabel('Ages')
>>> plt.ylabel('Estimated Salary')
>>> plt.title('Estimated Salaries with Age')
>>> plt.show()
```

Well, turns out there is, looks like the average salary of an age group reduces with an increase in the age. Cool Insight. I'm sure I can go on and on about the insights but let's get onto the Machine Learning wagon since that's what this Research is about.


Now let's get to the real deal which iiiisss....
# MACHINE LEARNING !!!!!

As I am still an amateur, I want to disect these binary classifiers. We start with simple LOGISTIC REGRESSION, nothing complicated here.

## LOGISTIC REGRESSION

Our focus for now, is going to be on just ***Logistic Regression***. We won't compress our data using any **dimensionality reduction techniques** for now, we'll just use the data as is, and we'll try to evaluate the model we come up with.

the data should now have only numeric values with the first 3 columns dropped. We'll add them later to see how our model changes.

```html
>>> X = data[:, :-1] ## all columns with the exception of the last one
>>> y = data[:, -1] ## the last column (target)
```


Let's try splitting the data and Importing LogisticRegression from sklearn
```html
>>> import sklearn.model_selection import train_test_split
>>> X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=123)
```

Many learning algorithms require input features on the same scale for optimal performance. So we have to standardize our data
```html
>>> from sklearn.preprocessing import StandardScaler
>>> sc = StandardScaler()
>>> X_train_std = sc.fit_transform(X_train)
>>> X_test_std = sc.transform(X_test)
```

We can now train our model
```html
import sklearn.linear_model import LogisticRegression
lr = LogisticRegression(random_state=123, solver='lbfgs', C=100, n_jobs=1).fit(X_train, y_train)
```

Let's make a few predictions
```html
>>> probas = lr.predict_proba(X_test[:5, :])
>>> print(probas)
```
We have something interesting going on. Maybe, just maybe, there's a large imbalance in the distribution of our targets, we'll explore this later as well. Let's Evaluate our Model using different techniques for now.

We'll start with ***k-fold cross validation with k=10***. I hear it's the best k to choose :)

```html
>>> kfold = StratifiedKFold(n_splits=10, random_state=1).split(X_train_std, y_train) # indices for splits
```

Now we have the indices for the training and test(validation) data
```html
scores = []
>>> for k, (train, test) in enumerate(kfold):
...    lr = lr.fit(X_train[train], y_train[train])
...    score = lr.score(X_train[test], y_train[test])
...    scores.append(score)
...    print("Fold %2d, Class Distribution.: %s, Acc %.3f" % (k+1, np.bincount(y_train[train].astype(int)), score))

print('\nCV Accuracy: %.3f +/- %.3f' % (np.mean(scores), np.std(scores)))
```
```html
Fold  1, Class Distribution.: [5016 1283], Acc 0.809
Fold  2, Class Distribution.: [5016 1283], Acc 0.822
Fold  3, Class Distribution.: [5016 1283], Acc 0.803
Fold  4, Class Distribution.: [5016 1283], Acc 0.815
Fold  5, Class Distribution.: [5017 1283], Acc 0.821
Fold  6, Class Distribution.: [5017 1283], Acc 0.797
Fold  7, Class Distribution.: [5017 1284], Acc 0.821
Fold  8, Class Distribution.: [5017 1284], Acc 0.794
Fold  9, Class Distribution.: [5017 1284], Acc 0.811
Fold 10, Class Distribution.: [5017 1284], Acc 0.797

CV Accuracy: 0.809 +/- 0.010
```

so we have the average cross validation score to assess our model. In this context, 80% accuracy doesn't seem too bad, but of course, we have to increase this number somehow!!.


Let's further evaluate our model with different criteria
# LEARNING AND VALIDATION CURVES





