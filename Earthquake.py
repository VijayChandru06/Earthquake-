import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

data = pd.read_csv('earthquake_data.csv')

x = data.drop('earthquake_occured',axis=1)
y = data['earthquake_occured']

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)
clf = RandomForestClassifier()

clf.fit(x_train,y_train)

y_pred = clf.predict(x_test)

accuracy = accuracy_score(y_test,y_pred)
print(f"Accuracy:{accuracy}")