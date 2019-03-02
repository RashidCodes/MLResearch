import seaborn as sns 
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt

try:
    data = pd.read_csv('Churn_Modelling1.csv')
except:
    print('The data was not found')

data = data.drop(['Exited'], axis=1)
data.columns = ['CreditScore', 'Geography', 'Gender', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary']
cols = data.columns
cm = np.corrcoef(data[cols].values.T)
sns.set(font_scale=1.5)
hmm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size' : 15}, yticklabels=cols, xticklabels=cols)
plt.show()