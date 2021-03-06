import pandas
import xgboost
import sklearn

# Common machine learning regression algorithms
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso

dataframe = pandas.read_csv('https://modcom.co.ke/bigdata/datasets/Advertising.csv')
print(dataframe)
# Pick the relevant predicting columns
subset = dataframe[(['TV', 'Radio', 'Newspaper', 'Sales'])]
print(subset)
# Obtain the values of the subset
array = subset.values
print(array)
# Define the features
X = array[:, 0:3]
print(X)
# Define the label
y = array[:, 3]
print(y)
# Define the training and testing data set
from sklearn import model_selection
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.30, random_state=7)
# Pick an algorithm
model = GradientBoostingRegressor()
# Fit the training data set
model.fit(X_train, y_train)
print('Training done!!')
# Get the trained model to predict
prediction = model.predict(X_test)
print(prediction)
# Check the accuracy
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
print('Accuracy: ', r2_score(prediction, y_test))
print('Mean squared error: ', mean_squared_error(prediction, y_test))

# Give the model random data to predict
Television = float(input('Input the TV cost: '))
Radio = float(input('Input the radio cost: '))
Newspaper = float(input('Input the newspaper cost: '))
predict = model.predict([(Television, Radio, Newspaper)])
print('The predicted sales is: ', predict)
