# CLASSIFICATION MODEL
import pandas
import sklearn
import xgboost
# Read the csv file
dataframe = pandas.read_csv('https://modcom.co.ke/bigdata/datasets/iris.csv')
print(dataframe)
# Obtain values from the dataframe
array = dataframe.values
print(array)
# Identify the features
X = array[:, 0:4]
print(X)
# Identify the label
y = array[:, 4]
print(y)

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

# Divide the data into training and testing datasets
from sklearn import model_selection
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.30, random_state=7)
# Pick a suitable algorithm
model = LinearDiscriminantAnalysis()
# Fit the training dataset
model.fit(X_train, y_train)
print('Training done!!')
# Get the trained model to predict
prediction = model.predict(X_test)
print(prediction)
# Check the accuracy of your prediction
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
print('Accuracy: ', accuracy_score(prediction, y_test))
print(confusion_matrix(prediction, y_test))
print(classification_report(prediction, y_test))
# Test the model
Sepallength = float(input('Enter sepallength: '))
Sepalwidth = float(input('Enter sepalwidth: '))
Petallength = float(input('Enter petallength: '))
Petalwidth = float(input('Enter petalwidth: '))
predictor = model.predict([(Sepallength, Sepalwidth, Petallength, Petalwidth)])
print('The predicted flower class is: ', predictor)
