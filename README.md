# Module 19 Challenge

## Employee Attrition

This notebook assess an employee's risk of attrition.

### Requirements

#### Preprocessing 

* Import the data.

``` python
attrition_df = pd.read_csv('https://static.bc-edx.com/ai/ail-v-1-0/m19/lms/datasets/attrition.csv')
attrition_df.head()
```

* Create y_df with the attrition and department columns.

``` python
y_df = attrition_df[['Attrition','Department']]
```

* Choose 10 columns for X.

``` python
names = ["Age","DistanceFromHome","EnvironmentSatisfaction","JobInvolvement","JobSatisfaction","PerformanceRating","WorkLifeBalance","YearsSinceLastPromotion","RelationshipSatisfaction","NumCompaniesWorked"]
```

* Show the data types of the X columns.

``` python
X_df = attrition_df[names]
X_df.dtypes
```

* Split the data into training and testing sets.

``` python
X_train, X_test, y_train, y_test = train_test_split(X_df, y_df, random_state=42)
```

* Encode all X data to numeric types.

``` python
X_train
```

* Scale the X data.

``` python
s = StandardScaler()
s.fit(X_train)
X_train_scaled = s.transform(X_train)
X_test_scaled = s.transform(X_test)
```

* Encode all y data to numeric types.

``` python
ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False, dtype='int')
y_train_department_encoded = pd.DataFrame(data=ohe.fit_transform(pd.DataFrame(y_train["Department"])), columns=ohe.get_feature_names_out())
y_test_department_encoded = pd.DataFrame(data=ohe.fit_transform(pd.DataFrame(y_test["Department"])), columns=ohe.get_feature_names_out())

ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False, dtype='int')
y_train_attrition_encoded = pd.DataFrame(data=ohe.fit_transform(pd.DataFrame(y_train["Attrition"])), columns=ohe.get_feature_names_out())
y_test_attrition_encoded = pd.DataFrame(data=ohe.fit_transform(pd.DataFrame(y_test["Attrition"])), columns=ohe.get_feature_names_out())
```

#### Model

* Find the number of columns in the X training data.

``` python
X_train.shape[1]
```

* Create an input layer.

``` python
input_layer = layers.Input(shape=(X_train.shape[1],), name='input_layer')
```

* Create at least two shared hidden layers.

``` python
shared_layer1 = layers.Dense(21, activation='relu')(input_layer)
shared_layer2 = layers.Dense(10, activation='relu')(shared_layer1)
shared_layer3 = layers.Dense(5, activation='relu')(shared_layer2)
```

* Create an output branch for the department column.

``` python
department_dense = layers.Dense(5, activation='relu')(shared_layer3)
department_output = layers.Dense(3,activation='softmax',name='department_output')(department_dense)
```

* Create an output branch for the attrition column.

``` python
attrition_dense = layers.Dense(5, activation='relu')(shared_layer3)
attrition_output = layers.Dense(2,activation='sigmoid',name='attrition_output')(attrition_dense)
```

#### Summary

* Answer the questions briefly.
* Show understanding of the concepts in your answers.

``` python
"""
1. Accuracy is not a good metric for this data because both the y columns are imbalanced. The attrition column has 1233 No values and 237 Yes values. Likewise, the department column has 961 R&D, 446 Sales, and 63 HR. Accuracy does not do well with imbalanced data because a model can do well simply by siding with the majority class.
2. For the department output layer I chose softmax because we need only 1 output that is most likely, meaning the result cannot belong to more than one class. For the attrition output layer I chose sigmoid becuase it is a binary classification. 
3. Hyperparameter tuning of the epochs, batch size, activation functions, optimizers, number of layers, number of nodes in each layer, and balancing the data.
"""
```

* The reloaded model was used to make binary predictions on the testing data.

``` python
predictions = nn_imported.predict(X_test)
p = pd.DataFrame({"Predictions":list(predictions)})
p['Predictions'] = [round(x[0]) for x in p["Predictions"]]
p
```

* A classification report is generated for the predictions and the testing data.

``` python
pp=list(p["Predictions"])
print(f"Classification Report:\n {classification_report(y_test,pp)}")
```

#### Discuss creating a recommendation system for student loans

##### Question 1

* The response describes the data that should be collected to build a recommendation system for student loan options.
* The response explains why they think that data should be collected.
* The type of data described is appropriate for a recommendation system for student loan options.

``` python
'''
The data that should be collected is the data that is the most closely related to financial habits. I would focus more on features such as current debt (credit and other), income (scholarships, job, etc), delinquent debt, credit history, and employment history. These features are better suited for loan repayment calculations because they are more closely tied with an individuals financial habits. Current debt and income will give a good idea about what other loan obligations does this individual have and will they be able to cover this new loan on top of the other debt. Delinquent debt payments and credit history will show the individual's ability to pay on time and consistency. Employment history will show if this individual has a stable source of income that can be relied on to pay back the loan for the life of the loan.
'''
```

##### Question 2

* The response chose a filtering method.
* The student justified the choice of their filtering method.
* The choice of filtering method was appropriate for the data selected in the previous question.

``` python
'''
The model would use collaborative filtering. This is because we are looking for similarities between users on what each feature is and what interest rate they took and if they repay it or not. It is reasonable to think that users with similar financial habits would make better candidtates for repaying the loan and so we can offer better rates to sell the loan. In other words, there is a higher probability that a new applicant will pay back the loan if their features are similar to the existing students who hve paid back their loans.
'''
```

##### Question 3

* The response lists two real-world challenges with building a recommendation system for student loans. (4 points)
* The response explains why these challenges would be of concern for a student loan recommendation system. (6 points)

``` python
'''
One challenge would be the inherent unpredictability of a student's situation to remain a good candidate for the life of the loan. Meaning, even though all the features at the time of the application indicated they were a good candidate, something could happen that dramatically changes their features and causes them to become delinquent on the loan. Another challenge is the quality of the data. This is very confidential information that is required and there is a risk of false data which could adversly effect other applicants' ability to qualify for a better loan.
'''
```