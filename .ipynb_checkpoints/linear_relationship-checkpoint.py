import pandas as pd 
import numpy as np 
from matplotlib import pyplot as plt

## we'll first grab our data
try:
    data = pd.read_csv('Churn_Modelling.csv')
except:
    print('The data was not found')

X = data['CreditScore']
y = data['Exited']

plt.scatter(np.array(X)[:50], y[:50], color='green', marker='o', label='CreditScore vs Target')
plt.xlabel('CreditScore')
plt.ylabel('Exited')
plt.legend(loc='center right')
plt.show()