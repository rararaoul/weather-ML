# weather-ML
ML - EDA

import pandas as pd

weather=pd.read_csv('weatherAUS.csv')

weather.dropna()
weather.Date=pd.to_datetime(weather.Date)
weather.Location.value_counts().plot(kind="pie")
weather.groupby(["year"]).RainToday.count()
weather.groupby(["year"]).RainToday.value_counts().plot(kind="bar")
weather.RainToday.value_counts(normalize=True).plot(kind="bar")
temp = weather.copy()
temp.RainToday= temp.RainToday.map({"No":0, "Yes":1})
temp.groupby("Location").RainToday.mean().sort_values().plot(kind="bar")

weather.groupby(["RainToday"]).Rainfall.count().plot(kind="bar")


MACHINE LEARNING

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.compose import make_column_transformer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# X & y
X=weather1
y=weather1.RainTomorrow.str.replace("Yes", "1").str.replace("No", "0").astype(float)

# train_test_split

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test=train_test_split(X, y, random_state=42)

classifiers=[
    LogisticRegression(),
    DecisionTreeClassifier(),
    RandomForestClassifier()
]
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline

scaler=StandardScaler()

num_cols=['MinTemp', 'MaxTemp', 'Rainfall', 'Evaporation', 'Sunshine', 'WindGustSpeed', 'WindSpeed3pm', 'Humidity9am', 'Humidity3pm', 'Pressure9am', 'Pressure3pm', 'Cloud9am', 'Cloud3pm', 'Temp9am', 'Temp3pm']
cat_cols=['Location', 'WindGustDir', 'WindDir9am', 'RainToday']

num_trans = StandardScaler()
cat_trans = OneHotEncoder(drop='if_binary')

preproc = make_column_transformer(
    (num_trans, num_cols), 
    (cat_trans, cat_cols)
)


for clf_model in classifiers:
    pipe=make_pipeline(preproc, clf_model)
    
    grid=GridSearchCV(pipe, param_grid={}, cv=5, scoring="accuracy")
    grid.fit(X_train, y_train)
    
    print(f"Train score for {clf_model}: {grid.best_score_}")
    
    Train score for LogisticRegression(): 0.8561503013115915
Train score for DecisionTreeClassifier(): 0.7974949781401395
Train score for RandomForestClassifier(): 0.8612785064397966

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline

scaler=StandardScaler()

num_cols=['MinTemp', 'MaxTemp', 'Rainfall', 'Evaporation', 'Sunshine', 'WindGustSpeed', 'WindSpeed3pm', 'Humidity9am', 'Humidity3pm', 'Pressure9am', 'Pressure3pm', 'Cloud9am', 'Cloud3pm', 'Temp9am', 'Temp3pm']
cat_cols=['Location', 'WindGustDir', 'WindDir9am', 'RainToday']

num_trans = StandardScaler()
cat_trans = OneHotEncoder(drop='if_binary')

preproc = make_column_transformer(
    (num_trans, num_cols), 
    (cat_trans, cat_cols)
)

reg = LogisticRegression()

pipe = make_pipeline(preproc, reg)

param_grid=[{"logisticregression__C":[0.001, 0.01, 0.1, 1, 10], "logisticregression__solver": ["lbfgs", "liblinear"]}]

grid=GridSearchCV(pipe, param_grid=param_grid, scoring="accuracy", cv=5)

grid.fit(X_train, y_train)

GridSearchCV(cv=5,
             estimator=Pipeline(steps=[('columntransformer',
                                        ColumnTransformer(transformers=[('standardscaler',
                                                                         StandardScaler(),
                                                                         ['MinTemp',
                                                                          'MaxTemp',
                                                                          'Rainfall',
                                                                          'Evaporation',
                                                                          'Sunshine',
                                                                          'WindGustSpeed',
                                                                          'WindSpeed3pm',
                                                                          'Humidity9am',
                                                                          'Humidity3pm',
                                                                          'Pressure9am',
                                                                          'Pressure3pm',
                                                                          'Cloud9am',
                                                                          'Cloud3pm',
                                                                          'Temp9am',
                                                                          'Temp3pm']),
                                                                        ('onehotencoder',
                                                                         OneHotEncoder(drop='if_binary'),
                                                                         ['Location',
                                                                          'WindGustDir',
                                                                          'WindDir9am',
                                                                          'RainToday'])])),
                                       ('logisticregression',
                                        LogisticRegression())]),
             param_grid=[{'logisticregression__C': [0.001, 0.01, 0.1, 1, 10],
                          'logisticregression__solver': ['lbfgs',
                                                         'liblinear']}],
             scoring='accuracy')
             
             
 grid.best_score_
 0.8562920950017723
grid.best_params_
{'logisticregression__C': 10, 'logisticregression__solver': 'lbfgs'}
grid.best_estimator_
             
Pipeline(steps=[('columntransformer',
                 ColumnTransformer(transformers=[('standardscaler',
                                                  StandardScaler(),
                                                  ['MinTemp', 'MaxTemp',
                                                   'Rainfall', 'Evaporation',
                                                   'Sunshine', 'WindGustSpeed',
                                                   'WindSpeed3pm',
                                                   'Humidity9am', 'Humidity3pm',
                                                   'Pressure9am', 'Pressure3pm',
                                                   'Cloud9am', 'Cloud3pm',
                                                   'Temp9am', 'Temp3pm']),
                                                 ('onehotencoder',
                                                  OneHotEncoder(drop='if_binary'),
                                                  ['Location', 'WindGustDir',
                                                   'WindDir9am',
                                                   'RainToday'])])),
                ('logisticregression', LogisticRegression(C=10))])
                
best_model=grid.best_estimator_
best_model.predict(X_test)
'array([0., 0., 0., ..., 0., 0., 0.])
print(f"{grid.best_score_=}\n{grid.best_params_}")
grid.best_score_=0.8562920950017723
{'logisticregression__C': 10, 'logisticregression__solver': 'lbfgs'}

from sklearn.metrics import plot_confusion_matrix
plot_confusion_matrix(grid, X_train, y_train)
grid.score(X_test, y_test)
0.8577100319035803
from sklearn.metrics import plot_confusion_matrix
plot_confusion_matrix(grid, X_test, y_test)
