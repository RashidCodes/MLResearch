import pandas as pd 
import numpy as np 
from matplotlib import pyplot as plt

## we'll first grab our data
try:
    data = pd.read_csv('Churn_Modelling1.csv')
except:
    print('The data was not found')


## grabbing 2 features
X = data[['Age', 'Tenure']]
X = np.array(X)

# targets
y = data['Exited']

## all samples where Exited == 0
X_0 = X[y == 0, :]

## all samples where Exited == 1
X_1 = X[y == 1, :]

# We can now plot the data 
plt.scatter(X_0[:50, 0], X_0[:50, 1], color='blue', marker='x', label='Samples with class 0')
plt.scatter(X_1[:50, 0], X_1[:50, 1], color='red', marker='o', label='Samples with class 1')
plt.xlabel('Age')
plt.ylabel('Tenure')
plt.legend(loc='upper left')
plt.show()


## to get the summary of the relationship between featuers, we can use seaborn,
## however this didn't help much
# import seaborn as sns 
# data = data.drop(['Exited'], axis=1)
# data.columns = ['CreditScore', 'Geography', 'Gender', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary']
# cols = data.columns
# sns.pairplot(data[cols], height=2.5)
# plt.tight_layout()
# plt.show()




