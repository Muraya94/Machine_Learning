# REGRESSION MODEL
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

pandas.set_option('display.max_columns', 7)
dataframe = pandas.read_csv('https://modcom.co.ke/bigdata/datasets/Customers.csv')
print(dataframe)
# Create a subset that contains only numerical columns
subset = dataframe[(['Avg. Session Length', 'Time on App', 'Time on Website', 'Length of Membership', 'Yearly Amount Spent'])]
print(subset)
# Obtain values of the subset
array = subset.values
print(array)
# Identify the features
X = array[:, 0:4]
print(X)
# Identify the label
y = array[:, 4]
print(y)
# Identify the training and testing sets
from sklearn import model_selection
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.30, random_state=7)
# Pick an algorithm
model = LinearRegression()
# Fit the training data set
model.fit(X_train, y_train)
print('Training done!!')
# Get the trained model to predict
prediction = model.predict(X_test)
print(prediction)

# Check the accuracy of the predictions
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
print('Accuracy: ', r2_score(prediction, y_test))
print('Mean squared error: ', mean_squared_error(prediction, y_test))

# Give the model random data to predict
AvgSessionLength = float(input('Enter the avg. session length :'))
TimeOnApp = float(input('Enter the time on app :'))
TimeOnWebsite = float(input('Enter the time on website :'))
LengthOfMembership = float(input('Enter the length of membership :'))
predict = model.predict([(AvgSessionLength, TimeOnApp, TimeOnWebsite, LengthOfMembership)])
print('The predicted yearly amount spent: ', predict)
