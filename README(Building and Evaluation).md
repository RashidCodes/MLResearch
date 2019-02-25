# MLResearch
Predicting customer churn (Research)

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

### Deductions
So we have the average cross validation score to assess our model. In this context, 80% accuracy doesn't seem too bad, but of course, we have to increase this number somehow!!.


Let's further evaluate our model with different criteria
# DEBUGGING WITH LEARNING AND VALIDATION CURVES
### LEARNING CURVE
We all know the 2 biggest problems that Machine learning algorithms face; **overfitting** and **underfitting**. So let's see how our model is doing in this regard.

By plotting the model training and validation accuracies as functions of the training set size, we can easily detect whether our model suffers from high variance (overfitting) or high bias(underfitting). 

So without further ado, let's jump right into it.

We're going to use sklearn's learning_curve method. Before so though, usually I'll read a little bit about the function using the inbuilt help() function in python just to have an idea about other **kwargs.

```html
>>> import matplotlib as plt
>>> from sklearn.model_selection import learning_curve

>>> train_sizes, train_scores, test_scores = learning_curve(estimator=lr, X=X_train_std, y=y_train, train_sizes=np.linspace(0.1, 1.0, 10), cv=10, n_jobs=1)

>>> train_mean = np.mean(train_scores, axis=1)
>>> train_std = np.std(train_scores, axis=1)
>>> test_mean = np.mean(test_scores, axis=1)
>>> test_std = np.std(test_scores, axis=1)

>>> plt.plot(train_sizes, train_mean, color='magenta', marker='o', markersize=5, label='training accuracy')

>>> plt.fill_between(train_sizes, train_mean + train_std, train_mean - train_std, alpha=0.15, color='magenta')

>>> plt.plot(train_sizes, test_mean, color='green', linestyle='--', marker='s', markersize=5, label='validation accuracy')

>>> plt.fill_between(train_sizes, test_mean + test_std, test_mean - test_std, color='green', alpha=0.15)

>>> plt.grid(True)
>>> plt.xlabel('Number of training samples')
>>> plt.ylabel('Accuracy')
>>> plt.legend(loc='lower right')
>>> plt.ylim([0.7, 1.0])
>>> plt.show()

```

The ***train_sizes*** parameter tells us the number of training samples that are used to generate the learning_curves.


### Deductions
Phew!, our model underfits the data, too bad. In the event where our model underfits some data, we can explore any of these options:
- Reduce the strength of regularization.
- We can also add more training samples.

But looking at the curve, adding more training samples will probably not do us any good, we have to explore greater options.

Later in our project, we'll explore these options. For now, C=100, perhaps we should consider increasing C later. Moving on....let's play with validation curves as well.

### VALIDATION CURVE
Validation curves are also used to improve the performance of a model by addressing overfitting or underfitting. They are quite similar to learning_curves with the only difference being that the model parameters are varied, like C in logisticRegression.

```html
>>> import matplotlib as plt
>>> from sklearn.model_selection import validation_curve

>>> param_range = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
>>> train_scores, test_scores = validation_curve(estimator=lr, X=X_train_std, y=y_train, param_name='logisticregression__C', param_range=param_range, cv=10)

>>> train_mean = np.mean(train_scores, axis=1)
>>> train_std = np.std(train_scores, axis=1)
>>> test_mean = np.mean(test_scores, axis=1)
>>> test_std = np.std(test_scores, axis=1)

>>> plt.plot(param_range, train_mean, color='magenta', marker='o', markersize=5, label='training accuracy')

>>> plt.fill_between(param_range, train_mean + train_std, train_mean - train_std, alpha=0.15, color='magenta')

>>> plt.plot(param_range, test_mean, color='green', linestyle='--', marker='s', markersize=5, label='validation accuracy')

>>> plt.fill_between(param_range, test_mean + test_std, test_mean - test_std, color='green', alpha=0.15)

>>> plt.grid(True)
>>> plt.xscale('log')
>>> plt.xlabel('Parameter C')
>>> plt.ylabel('Accuracy')
>>> plt.legend(loc='lower right')
>>> plt.show()

```
Okaayy, thank you Mr. Validation Curve. He gave us a few insights
- Our validation accuracy is higher than our training accuracy after C=2.92
- Our model still underfits the data because the Accuracy is nowhere near 1.

So I decided to vary C and observe the performance of our model. Nothing much happened :(. Perhaps we can apply a few **dimensionality reduction techniques** to produce higher accuracies, we'll see.

## THE CONFUSION MATRIX
### Another evaluation method often used is the Confusion Matrix, let's see what it has to say about our data.
Firstly, let us predict some targets with our test set

```html
>>> from sklearn.metrics import confusion_matrix
>>> 
>>> y_predicted = lr.predict(X_test_std)
>>> confusion_matrix = confusion_matrix(y_true=y_test, y_pred=y_predicted)
```

Confusion matrix is all done, let's make it look a little prettier
```html
fig, ax = plt.subplots(figsize=(2.5, 2.5))
ax.matshow(confmat, cmap=plt.cm.Greens, alpha=0.3)
for i in range(confmat.shape[0]):
    for j in range(confmat.shape[1]):
        ax.text(x=j, y=i, s=confmat[i, j], va='center', ha='center')
plt.xlabel('predicted label')
plt.ylabel('true label')
plt.show()
```

There, all done. Damn, our model really sucks haha. Let's grab more evidence.
```html
>>> from sklearn.metrics import precision_score
>>> from sklearn.metrics import recall_score, f1_score
>>> print('Precision Score: %.3f' % precision_score(y_true=y_test, y_pred=y_predicted))

>>> print('Recall: %.3f' % recall_score(y_true=y_test, y_pred=y_predicted))

>>> print('F1: %.3f' % f1_score(y_true=y_test, y_pred=y_predicted))
```

Great, before we deal with **class imbalance**, let's plot the ROC curve for our model. They tell us how our model is doing with respect to TPR and FPR. A perfect classifier of course has a TPR = 1 and FPR = 0.

Before we write the code, let's explain to ourselves what's supposed to happen.

**Things we need to know.**
- The area under the roc curve(ROC AUC) for a perfect model is 1.00.

And now, we are going to plot ***fpr*** and ***tpr*** for each fold(k=3). And after this, we want to grab the area under our mean_fpr and mean_tpr. This is what is of main importance.

```html
>>> from sklearn.metrics import roc_curve, auc
>>> from scipy import interp

>>> cv = list(StratifiedKFold(n_splits=3, random_state=1).split(X_train_std, y_train)) ## k=3

>>> fig = plt.figure(figsize=(7, 5))

>>> mean_tpr = 0.0
>>> mean_fpr = np.linspace(0, 1, 100)
>>> all_tpr = []

## If ever you forget, just run help(roc_curve)
>>> for i, (train, test) in enumerate(cv):
...     probas = lr.fit(X_train_std[train], y_train[train]).predict_proba(X_train_std[test])
...     fpr, tpr, thresholds = roc_curve(y_train[test], probas[:, 1], pos_label=1)

>>> mean_tpr += interp(mean_fpr, fpr, tpr)
>>> mean_tpr[0] = 0.0 # always set our first value to 0

## we calculate the area for each split using auc
>>> roc_auc = auc(fpr, tpr)
>>> plt.plot(fpr, tpr, label='ROC Fold %d (area=%0.2f)' % (i+1, roc_auc))

## Now let's plot the curve for random guessing
>>> plt.plot([0, 1], [0, 1], linestyle='--', color=(0.5, 0.5, 0.5), label='random guessing')

## Next, our mean_fpr-mean_tpr curve --->  very important
>>> mean_tpr = mean_tpr / len(cv)

## We want the last value to have a value of 1.
>>> mean_tpr[-1] = 1.0

mean_auc = auc(mean_fpr, mean_tpr)
>>> plt.plot(mean_fpr, mean_tpr, color='black', linestyle=':', label='mean ROC (area = %0.2f)' % mean_auc, linewidth=2)

## Lastly we wanna plot the curve for a perfect model
>>> plt.plot([0, 0, 1], [0, 1, 1], linestyle='--', color='black', label='pefecto performance)

## Little editting
>>> plt.xlabel('False Positive Rate')
>>> plt.ylabel('True Positive Rate')
>>> plt.legend(loc='lower right')
>>> plt.show()
```

All done. Our ROC curve doesn't look bad at all, hopefully we'll be able to make our mean hit the corner someday. :)

# CLASS IMBALANCE
The last thing that I want to explore in our dataset is class imbalance. 
So I want to know the number of samples in our training set that have a class of 0 and 1.

```html
>>> class_0 = X_train_std[y_train == 0].shape[0]
>>> class_1 = X_train_std[y_train == 1].shape[0]
>>> print("Percentage of class 0 samples %.3f%%" % (class_0/7000*100)) ## 79.629%
>>> print("Percentage of class 1 samples %.3f%%" % (class_1/7000*100)) ## 20.371%
```
Haa, there's a large imbalance in our data, quite interesting.
### There are a few things we can do.
- We can resample the dataset so we have a balance.
- We can assign a large penalty to wrong predictions of the minority class which is class_1

This brings us to the end of our Evaluation of our simple logisticRegression model. Our model didn't do too well on our data but we obviously want it to, so we have to make a few changes and just hope for the best :).




