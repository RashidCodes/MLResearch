In our last post, we built a Logistic Regression Model with our data. Our model didn't do too well but as an amateur ML engineers, we want that to happen so we'll have to make a few changes. Afterall, what did Einstein say :).

Firstly, let's review some of the things that we did wrongly and some stuff that could have also been done.

## DEDUCTIONS FROM OUR PREVIOUS RESEARCH.
- With k-fold cross validation, we only achieved an accuracy of 80% on our test set.
- With the learning curve, we were able to tell that adding more training samples won't do us any good.
- With the learning curve, we were also able to tell that our model totally underfits our data.
- With the validation curve, we were also able to tell that our model didn't do too well with variations in regularization strength.
- We noticed that there was large imbalance in our classes with about an 80:20 ratio.


Looks like we have a few issues we might have to address. There are some assumptions we have to make before using a logistic regression classifier.

## SOME ASSUMPTIONS ABOUT LOGISTICREGRESSION 
- Logisitic regression does not require a linear relationship between the dependent and independent variable.
- Homoscedasticity is not required unlike in linear regression.
- The dependent variable is not measured on an interval.
- Observations must be independent of each other, in other words, observations should not come from repeated measurements or     matched data.
- Logistic regression assumes linearity of independent variables and log odds.
- The dependent variable should be dichotomous in nature.
- It works well when there are no outliers in your data.
- It also works well when there is no multicollinearity in our predictors. 
- The dataset must be linearly seperable.

Previously, we did not consider some of the assumptions aforementioned. So we'll start to make a better logistic regression model by following some of the assumptions. We'll try to tackle each assumptions subsequently.

## TACKLING ASSUMPTIONS
- Our data shows that there isn't a **linear relationship** between the dependent and independent variable. We can prove this easily using matplotlib. 

- **Homoscedasticity** won't be a necessity in the absence of linearity. The dependent variable was not measured on an **interval or in any sequential order** and also, the all **observations** are **independent** of each other.

- The **dependent** variable is dichotomous in nature so that's good.

- In the preparation of data for our last model, we did not check for **Outliers** in our data. In a post from statisticssolutions.com, this problem can be handled by removing values below -3.29 or 3.29. We will try this method and see how the model does.

- We also did not check for **multicollinearity** in our data. For any pair of highly correlated features, we'll drop one, and we'll assess how model performance for our final dataset.

- We will also grab two features for visualization purposes - checking for **linear seperability** between these features.

A model will be trained with the final dataset that follows all of the assumptions listed above.








