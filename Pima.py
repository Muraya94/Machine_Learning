# CLASSIFICATION MODEL
import pandas
import sklearn
import xgboost

# Common machine learning algorithms
from xgboost import XGBClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# Read contents of the file
dataframe = pandas.read_csv('https://modcom.co.ke/bigdata/datasets/pima.csv')
pandas.set_option('display.max_columns', 9)
print(dataframe)

array = dataframe.values
print(array)
X = array[:, 0:8]
print(X)
y = array[:, 8]
print(y)
# Identify features that won't be good predictors
from sklearn.feature_selection import RFE
rfc = RandomForestClassifier(n_estimators=40)
rfe = RFE(rfc, 5)
fitted = rfe.fit(X, y)
print('Selected columns: ', fitted.support_)
# Create a new dataset for the best predictors
subset = dataframe[(['Glucose', 'BloodPressure', 'BMI', 'DiabetesPedigreeFunction', 'Age'])]
print(subset)
# Obtain the values of the new dataset
subsetArray = subset.values
Xnew = subsetArray[:, 0:5]
print(Xnew)
# Establish the training and testing sets
from sklearn import model_selection
X_train, X_test, y_train, y_test = model_selection.train_test_split(Xnew, y, test_size=0.10, random_state=7)
# Pick an algorithm
model = LinearDiscriminantAnalysis()
# Fit the training dataset
model.fit(X_train, y_train)
print('Training done!!')
# Get the trained model to predict
prediction = model.predict(X_test)
print(prediction)
# Check the accuracy of the prediction
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
print('Accuracy: ', accuracy_score(prediction, y_test))
print(confusion_matrix(prediction, y_test))
print(classification_report(prediction, y_test))

# Prediction from users input
Glucose = int(input('Enter glucose concentration: '))
BloodPressure = int(input('Enter blood pressure: '))
BMI = float(input('Enter BMI: '))
DiabetesPedigreeFunction = float(input('Enter diabetes pedigree function: '))
Age = int(input('Enter age: '))
predicted = model.predict([(Glucose, BloodPressure, BMI, DiabetesPedigreeFunction, Age)])
print('The predicted diabetes outcome is: ', predicted)
