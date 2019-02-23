import pandas as pd 
from sklearn.model_selection import train_test_split 
from matplotlib import pyplot as plt
import numpy as np 


## Let's read in the data
data = pd.read_csv("Churn_Modelling.csv")

## Let's drop a few columns for little relevance for now
data = data.drop(['RowNumber', 'CustomerId', 'Surname', 'CreditScore'], axis=1)

## Dealing with the 2 nominal features
## Geography
unique_geography = np.unique(data['Geography'].values)
geography_dict = {val : idx for idx, val in enumerate(unique_geography)}
data['Geography'] = data['Geography'].map(geography_dict)


## Gender
gender_dict = {val : idx for idx, val in enumerate(np.unique(data['Gender'].values))}
data['Gender'] = data['Gender'].map(gender_dict)

## THE EXCITING MACHINE LEARNING PART

## let's grab our DataFrame and make it a numpy.array() object
data = np.array(data)

## Now we have our data to play with.
X = data[:, :-1]
y = data[:, -1]

## Now let's split them into our training and test sets 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=123)

## Let's standardize our data for better results
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_train_std = sc.fit_transform(X_train)
X_test_std = sc.transform(X_test)

## Great, so far so good, we can now move on to training our Logistic Regression Model
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(n_jobs=1, random_state=123, C=100, solver='lbfgs').fit(X_train_std, y_train)

## we can now make some predictions
probas = lr.predict(X_test[:5, :])

## let's test our the accuracy
y_pred = lr.predict(X_test)
print('Test Accuracy: %.3f' % lr.score(X_test_std, y_test)) ## 80.1% -- not bad, but let's make it better

## MODEL EVALUATION
## We'll start by using K-fold Cross Validation
## Note that we're still using the training data, we'll use the test != validation data to check our generalization performance.
from sklearn.model_selection import StratifiedKFold 
kfold = StratifiedKFold(n_splits=10, random_state=1).split(X_train_std, y_train) ## we now have our indices for the splits

scores = []
for k, (train, test) in enumerate(kfold):
    lr =lr.fit(X_train_std[train], y_train[train])
    score = lr.score(X_train_std[test], y_train[test])
    scores.append(score)
    print("Fold %2d, Class Distribution.: %s, Acc %.3f" % (k+1, np.bincount(y_train[train].astype(int)), score))

print('\nCV Accuracy: %.3f +/- %.3f' % (np.mean(scores), np.std(scores)))