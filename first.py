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

## let's test the accuracy
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


## Now let's plot our learning curves to check for overfitting or underfitting
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve

train_sizes, train_scores, test_scores = learning_curve(estimator=lr, X=X_train_std, y=y_train, train_sizes=np.linspace(0.1, 1.0, 10), cv=10, n_jobs=1)

train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)

plt.plot(train_sizes, train_mean, color='magenta', marker='o', markersize=5, label='training accuracy')

plt.fill_between(train_sizes, train_mean + train_std, train_mean - train_std, alpha=0.15, color='magenta')

plt.plot(train_sizes, test_mean, color='green', linestyle='--', marker='s', markersize=5, label='validation accuracy')

plt.fill_between(train_sizes, test_mean + test_std, test_mean - test_std, color='green', alpha=0.15)

plt.grid(True)
plt.xlabel('Number of training samples')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.ylim([0.7, 1])
plt.show()

## Let's grab the validation curve as well, we'll only be playing the parameter C.
from sklearn.model_selection import validation_curve

param_range = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0, 100000.0]
train_scores, test_scores = validation_curve(estimator=lr, X=X_train_std, y=y_train, param_name='C', param_range=param_range, cv=10)

train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)

plt.plot(param_range, train_mean, color='magenta', marker='o', markersize=5, label='training accuracy')

plt.fill_between(param_range, train_mean + train_std, train_mean - train_std, alpha=0.15, color='magenta')

plt.plot(param_range, test_mean, color='green', linestyle='--', marker='s', markersize=5, label='validation accuracy')

plt.fill_between(param_range, test_mean + test_std, test_mean - test_std, color='green', alpha=0.15)

plt.grid(True)
plt.xscale('log')
plt.xlabel('Parameter C')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.show()

## We still have more metrics to explore. Let's check out the famous confusion matrix
from sklearn.metrics import confusion_matrix 
y_predicted = lr.predict(X_test_std)
confmat = confusion_matrix(y_true = y_test, y_pred = y_predicted)

## and we have our confusion matrix, let's make it look a little bit prettier
fig, ax = plt.subplots(figsize=(2.5, 2.5))
ax.matshow(confmat, cmap=plt.cm.Greens, alpha=0.3)
for i in range(confmat.shape[0]):
    for j in range(confmat.shape[1]):
        ax.text(x=j, y=i, s=confmat[i, j], va='center', ha='center')
plt.xlabel('predicted label')
plt.ylabel('true label')
plt.show()

## Let's grab more evidence that our model absolutely sucks using the precision_score and recall_score methods from sklearn.metrics module.
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score, f1_score
print('Precision Score: %.3f' % precision_score(y_true=y_test, y_pred=y_predicted))
print('Recall: %.3f' % recall_score(y_true=y_test, y_pred=y_predicted))
print('F1: %.3f' % f1_score(y_true=y_test, y_pred=y_predicted))


## RECEIVER OPERATING CHARACTERISTICS
from sklearn.metrics import roc_curve, auc
from scipy import interp

## let's create a list of our cross_validation splits
cv = list(StratifiedKFold(n_splits=3, random_state=1).split(X_train_std, y_train))

## let's create a new figure for our curve
fig = plt.figure(figsize=(7, 5))

mean_tpr = 0.0
mean_fpr = np.linspace(0, 1, 100)

## plot the ROC curves for each fold
for i, (train, test) in enumerate(cv):
    probas = lr.fit(X_train_std[train], y_train[train]).predict_proba(X_train_std[test])
    fpr, tpr, thresholds = roc_curve(y_train[test], probas[:, 1], pos_label=1)
    mean_tpr += interp(mean_fpr, fpr, tpr)
    mean_tpr[0] = 0.0
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label='ROC fold %d (area = %0.2f)' % (i+1, roc_auc))


## Let's plot the curve for random guessing
plt.plot([0, 1], [0, 1], linestyle='--', color=(0.5, 0.5, 0.5), label='random guessing')


## grabbing the mean_tpr over k=3 splits
mean_tpr = mean_tpr / len(cv)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
plt.plot(mean_fpr, mean_tpr, color='black', linestyle='--', label='mean ROC (area = %0.2f' % mean_auc, linewidth=2)

## let's plot the curve for perfect performance
plt.plot([0, 0, 1], [0, 1, 1], 'k:', label='perfect performance')

# Some Annotation
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='lower right')
plt.show()

