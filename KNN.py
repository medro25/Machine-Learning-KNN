import sklearn
from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np
from sklearn import linear_model,preprocessing
# read the data
data =pd.read_csv("car.data")
print(data.head())
# transform our data into numerical data
# take the labels and encode them into appropriate integer value
le= preprocessing.LabelEncoder()
# create an  numpy array in Pandas of each column of the data frame. collect each column and collect them into list as integer values
buying=le.fit_transform(list(data["buying"]))
maint=le.fit_transform(list(data["maint"]))
doors=le.fit_transform(list(data["doors"]))
persons=le.fit_transform(list(data["persons"]))
lug_boot=le.fit_transform(list(data["lug_boot"]))
safety=le.fit_transform(list(data["safety"]))
cls=le.fit_transform(list(data["class"]))
print(safety)
# our main target to predict class of the car
predict="class"
# x axe will take features
X=list(zip(buying,maint,doors,persons,lug_boot,safety))
# y ax will take labels
y=list(cls)
# split the data into train and test data
X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.1)
#implement our model to train the data
model= KNeighborsClassifier(n_neighbors=9)
# find the best fit
model.fit(X_train,y_train)
# test the accuracy
acc=model.score(X_test,y_test)
print(acc)
# predict the data with the data that contains input values for testing
predicted= model.predict(X_test)
# give numbers names to get the actual value
names= ["unacc","acc","good","vgood"]
for x in range (len(predicted)):
 print("predicted: ",names[predicted[x]], "data :",X_test[x], "actual : ", names[y_test[x]])