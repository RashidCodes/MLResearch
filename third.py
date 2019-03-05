import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

employee_df = pd.read_csv('Churn_Modelling.csv')
employee_df = employee_df.drop(['RowNumber', 'CustomerId', 'Surname', 'Exited'], axis=1)

## encoding categorical features
geo_dict = {val:idx for idx, val in enumerate(np.unique(employee_df['Geography'].values))}
gen_dict = { val : idx for idx, val in enumerate(np.unique(employee_df['Gender'].values))}

employee_df['Geography'] = employee_df['Geography'].map(geo_dict)
employee_df['Gender'] = employee_df['Gender'].map(gen_dict)

green_diamond = dict(markerfacecolor='g', marker='o')
plt.boxplot(employee_df[['CreditScore', 'Age', 'Tenure', 'Balance', 'EstimatedSalary']], flierprops=green_diamond)
plt.title('Box Plot for selected features')
plt.show()