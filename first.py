import pandas as pd 
from sklearn.model_selection import train_test_split 
from matplotlib import pyplot as plt

data = pd.read_csv("Churn_Modelling.csv")
data = data.drop(['RowNumber', 'CustomerId', 'Surname', 'CreditScore'], axis=1)
ages = data['Age'].values
plt.hist(ages, bins=7, histtype='step', align='mid')
plt.xlabel('$Ages$')
plt.ylabel('$Number$ $of$ $Peeps$')
plt.title('Distribution of Ages')
plt.show()