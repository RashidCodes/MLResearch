import pandas as pd 
import numpy as np 

## let's import our data
try:
    data = pd.read_csv('Churn_Modelling.csv')
except(FileNotFoundError):
    print('The file was not found')

## now let's selected the features we need ( Age, Tenure)
X = data[['Age', 'Tenure']]
X = np.array(X)

## we also need to select the targets
y = data['Exited']
y = np.array(y)

## Before we go ahead onto model fitting, let's first see if our features are linearly seperable
plt.scatter(X[:, 0], X[:, 1])

## Now let's prepare our data for training, we'll start by splitting 
from sklearn.model_selection import train_test_split 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=1)

## Then standardizing 
from sklearn.preprocessing import StandardScaler 
sc = StandardScaler()
X_train_std = sc.fit_transform(X_train)
X_test_std = sc.transform(X_test)

## Next let's write a function that'll plot us our decision regions for VISUALIZATION purposes.
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

def plot_decision_regions(X, y, classifier, test_idx=None, resolution=0.02):
    #set up marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])
    
    
    #plot the decision surface
    x1_min, x1_max = X[:, 0].min() -1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() -1, X[:, 1].max() + 1


    """the meshgrid is supposed to create cartesian coordinate system, like in your graph book"""
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution), np.arange(x2_min, x2_max, resolution))
   
    
    
    """the classifier is simply the classifier or (model) used"""    
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
    
    
    #plot class samples
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y==cl, 0], y=X[y==cl, 1], alpha=0.8, c=colors[idx], marker=markers[idx], label=cl, edgecolor='black')
        
        
        """highlight test samples"""
        if test_idx:
            """plot all samples"""
            X_test = X[test_idx, :]
            
            plt.scatter(X_test[:, 0], X_test[:, 1], c='', edgecolor='black', alpha=0.1, linewidth=1, marker='o', s=100, label='test set')
        

## let's train our model using KFold cross validation
from sklearn.model_selection import StratifiedKFold 
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(penalty='l2', random_state=1, solver='lbfgs')
kfold = StratifiedKFold(n_splits=10, random_state=1).split(X_train_std, y_train)

scores = []
for k, (train, test) in enumerate(kfold):
    lr.fit(X_train_std[train], y_train[train])
    score = lr.score(X_train_std[test], y_train[test])
    scores.append(score)
    print("Fold %2d, Class Distribution.: %s, Acc %.3f" % (k+1, np.bincount(y_train[train].astype(int)), score))

print('\nCV Accuracy: %.3f +/- %.3f' % (np.mean(scores), np.std(scores)))


## Now that we're done training, let's plot the decision regions 
X_combined_std = np.vstack((X_train_std, X_test_std))
y_combined = np.hstack((y_train, y_test))
plot_decision_regions(X=X_combined_std, y=y_combined, classifier=lr, test_idx=range(7000, 10000))
plt.xlabel('Age [standardized]')
plt.ylabel('Tenure [standardized]')
plt.legend(loc='upper left')
plt.show()
