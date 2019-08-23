import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
#% matplotlib inline

data=pd.read_csv("D:\VIT-2ND SEM\PROJECT\cloud\Disease-Prediction-System-master\DiabetesData.csv")
print(data.head(10))
print("# no of passenger in the data set:", +(len(data)))

##Analysis data

#sns.countplot(x="diastolic bp",data=data)
sns.countplot(x="diastolic bp",hue="age", data=data)
plt.show()
# sns.countplot(x="loans",hue="homeowner",data=bank_data)
# plt.show()
data["age"].plot.hist()
plt.show()
#


## TRAIN MY DATASET

x=data.drop("class",axis=1)
y=data["class"]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=1)
from sklearn.linear_model import LogisticRegression
logmodel=LogisticRegression()
print(logmodel.fit(x_train,y_train))

predictions=logmodel.predict(x_test)
from sklearn.metrics import classification_report
print(classification_report(y_test,predictions))  #generate classification report

##generate  accrucy

from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_test,predictions))

from sklearn.metrics import accuracy_score

print( "The Accuracy of the prediction is: ",accuracy_score(y_test,predictions))

